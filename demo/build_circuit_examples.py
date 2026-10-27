"""
Pre-compute the "Circuit Tracer" demo page (demo/pages/1_Circuit_Tracer.py).

For 10 HaluEval examples this saves, per example:
    - the prompt, references, the model's (A2) answer and its provisional label
    - every detector from B3-B6 (SelfCheck-NLI per sentence is computed
      here; semantic entropy, SEP and log-prob are read from data/b4..b6)
    - per-token log-prob / entropy / top alternatives
    - for the 3 most suspicious answer tokens, a circuit-tracer-style
      attribution graph and a logit lens (see `attribution_graph`)

Why not Anthropic's circuit-tracer itself: it needs transcoders trained for
the exact model, and none exist for Qwen2.5-0.5B-Instruct (the model that
produced every answer in this project). The graph here is a DIRECT-EFFECT
attribution graph on the real model: the logit difference
    logit(generated token) - logit(best alternative)
at the predicting position is split exactly (final RMSNorm scale frozen)
into the embedding, every MLP output, and every attention head x source
token. Nodes are model components (layer, head, source token), not
interpretable features.

Run on the server (GPU, a few minutes) from the repository root:
    python demo/build_circuit_examples.py
    python demo/build_circuit_examples.py --uids halueval-qa-00012 halueval-dialogue-00451
Output: demo/circuit_examples.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.common import (  # noqa: E402
    CALIBRATION_PATH,
    DATA_DIR,
    LABELS_PATH,
    LLM_DIR,
    MAIN_ANSWERS_PATH,
    MODELS_DIR,
    SAMPLES_PATH,
    NLIModel,
    chat_messages,
    git_commit_sha,
    split_sentences,
)
from src.signals.selfcheck import selfcheck_nli_batch  # noqa: E402


OUTPUT_PATH = PROJECT_ROOT / "demo" / "circuit_examples.json"
NLI_PATH = MODELS_DIR / "deberta-v3-large-mnli"

# How many examples of each kind (subset, category) to pick.
PLAN = [
    ("qa", "detected hallucination", 3),
    ("qa", "correct and confident", 2),
    ("qa", "confident hallucination (missed)", 1),
    ("dialogue", "detected hallucination", 2),
    ("dialogue", "correct and confident", 1),
    ("dialogue", "confident hallucination (missed)", 1),
]

# Detectors that do NOT use the labels; their mean percentile is the "risk"
# used to pick detected vs missed examples. (The accuracy probe is trained
# on the labels, so it is shown but not used for selection.)
RISK_COLUMNS = ["semantic_entropy", "logprob_neg_mean", "sep_tbg"]

N_TARGETS = 3          # suspicious tokens per example that get a graph
NODE_MASS = 0.80       # keep nodes until they explain 80% of |attribution|
MAX_NODES = 40
TOP_K = 5              # alternatives / promoted tokens shown per item

STOPWORDS = set("""
a an the of in on at to for from by with and or but is are was were be been
being it its this that these those as he she they them his her their we you i
not no do does did has have had will would can could should may might also
there here than then so such which who whom what when where how all any some
""".split())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uids", nargs="*", default=None,
                        help="Use these uids instead of the automatic selection.")
    parser.add_argument("--llm", default=str(LLM_DIR))
    parser.add_argument("--nli-model", default=str(NLI_PATH))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


# -------------------------------------------------------------------
# 1. Data
# -------------------------------------------------------------------

def load_frame() -> pd.DataFrame:
    main = pd.read_parquet(MAIN_ANSWERS_PATH, columns=["uid", "subset", "prompt", "generated_answer"])
    main = main.rename(columns={"generated_answer": "main_answer"})
    refs = pd.read_parquet(CALIBRATION_PATH, columns=["uid", "context", "reference_correct", "reference_incorrect"])
    labels = pd.read_parquet(LABELS_PATH).drop(columns=["subset"])
    b4 = pd.read_parquet(DATA_DIR / "b4" / "scores.parquet",
                         columns=["uid", "semantic_entropy", "discrete_semantic_entropy",
                                  "n_clusters", "n_answers", "cluster_ids"])
    b5 = pd.read_parquet(DATA_DIR / "b5" / "scores.parquet",
                         columns=["uid", "sep_tbg", "sep_slt", "accprobe_tbg"])
    b6 = pd.read_parquet(DATA_DIR / "b6" / "scores.parquet", columns=["uid", "logprob_neg_mean"])

    frame = main
    for other in (refs, labels, b4, b5, b6):
        other["uid"] = other["uid"].astype(str)
        frame = frame.merge(other, on="uid", how="left")

    # Percentile of every detector over all 24,652 answers (for display).
    for column in ["semantic_entropy", "logprob_neg_mean", "sep_tbg", "sep_slt", "accprobe_tbg"]:
        frame[f"{column}_pct"] = frame[column].rank(pct=True)
    frame["risk"] = frame[[f"{c}_pct" for c in RISK_COLUMNS]].mean(axis=1)
    return frame


def select_examples(frame: pd.DataFrame) -> list[tuple[str, str]]:
    words = frame["main_answer"].str.split().str.len()
    pool = frame[
        frame["subset"].isin(["qa", "dialogue"])
        & ~frame["refusal"].astype(bool)
        & frame["hallucinated"].notna()
        & words.between(12, 90)
        & frame["risk"].notna()
    ]

    chosen: list[tuple[str, str]] = []
    used: set[str] = set()
    for subset, category, count in PLAN:
        part = pool[pool["subset"] == subset]
        if category == "correct and confident":
            part = part[part["hallucinated"] == 0]
            if subset == "qa":  # reference answer literally in the answer
                part = part[part["correct_score"] >= 0.99]
            part = part.sort_values(["risk", "uid"])
        else:
            part = part[part["hallucinated"] == 1]
            if subset == "qa":  # clearly does not contain the reference answer
                part = part[part["correct_score"] < 0.05]
            ascending = category.startswith("confident")
            part = part.sort_values(["risk", "uid"], ascending=[ascending, True])
        picked = [u for u in part["uid"] if u not in used][:count]
        used.update(picked)
        chosen.extend((u, category) for u in picked)
        if len(picked) < count:
            print(f"[WARNING] only {len(picked)} examples for {subset} / {category}")
    return chosen


def load_samples(uids: list[str]) -> dict[str, list[str]]:
    extra = pd.read_parquet(SAMPLES_PATH, columns=["uid", "response_index", "generated_answer"])
    extra["uid"] = extra["uid"].astype(str)
    extra = extra[extra["uid"].isin(uids)].sort_values(["uid", "response_index"])
    return extra.groupby("uid")["generated_answer"].agg(list).to_dict()


def b6_youden_threshold() -> float | None:
    path = DATA_DIR / "b6" / "threshold_summary.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path)
    row = t[(t["score"] == "logprob_neg_mean") & (t["rule"] == "youden")]
    return float(row["threshold_mean"].iloc[0]) if len(row) else None


# -------------------------------------------------------------------
# 2. Model with the hooks the attribution needs
# -------------------------------------------------------------------

class TracedModel:
    def __init__(self, path: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        # eager attention: we need the attention probabilities themselves
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager"
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)  # pure analysis: no autograd anywhere

        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.layers = self.model.model.layers
        self.final_norm = self.model.model.norm
        self.W_U = self.model.lm_head.weight  # (V, D)
        self.norm_w = self.final_norm.weight
        self.norm_eps = getattr(self.final_norm, "variance_epsilon", getattr(self.final_norm, "eps", 1e-6))

        self.im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def prompt_ids(self, prompt: str) -> list[int]:
        encoded = self.tok.apply_chat_template(chat_messages(prompt), tokenize=True, add_generation_prompt=True)
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        return list(encoded)

    def decode(self, token_id: int) -> str:
        return self.tok.decode([int(token_id)], clean_up_tokenization_spaces=False)

    def roles(self, ids: list[int], prompt_len: int) -> list[str]:
        """'user' for the user's prompt text, 'template' for chat scaffolding, 'answer' after."""
        roles = ["template"] * prompt_len
        i = 0
        while i < prompt_len:
            if ids[i] == self.im_start and i + 1 < prompt_len:
                role = self.decode(ids[i + 1]).strip()
                j = i + 3  # <|im_start|> role \n
                while j < prompt_len and ids[j] != self.im_end:
                    if role == "user":
                        roles[j] = "user"
                    j += 1
                i = j
            i += 1
        return roles + ["answer"] * (len(ids) - prompt_len)

    @torch.inference_mode()
    def run(self, ids: list[int]) -> dict:
        captured = {"emb": None, "v": {}, "mlp": {}, "resid": {}}

        def first(o):
            t = o[0] if isinstance(o, tuple) else o
            return t[0].detach()

        handles = [self.model.model.embed_tokens.register_forward_hook(
            lambda m, i, o: captured.__setitem__("emb", o[0].detach()))]
        for l, layer in enumerate(self.layers):
            handles.append(layer.self_attn.v_proj.register_forward_hook(
                lambda m, i, o, l=l: captured["v"].__setitem__(l, o[0].detach())))
            handles.append(layer.mlp.register_forward_hook(
                lambda m, i, o, l=l: captured["mlp"].__setitem__(l, o[0].detach())))
            handles.append(layer.register_forward_hook(
                lambda m, i, o, l=l: captured["resid"].__setitem__(l, first(o))))
        try:
            out = self.model(
                input_ids=torch.tensor([ids], device=self.device),
                output_attentions=True,
                use_cache=False,
            )
        finally:
            for h in handles:
                h.remove()

        if out.attentions is None or out.attentions[0] is None:
            raise RuntimeError("No attention probabilities returned; eager attention is required.")
        captured["attn"] = [a[0].detach() for a in out.attentions]  # (H, T, T)
        captured["logits"] = out.logits[0].detach().float()          # (T, V)
        return captured

    def norm_scale(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.pow(2).mean() + self.norm_eps)

    def promoted(self, vec: torch.Tensor, rms: torch.Tensor) -> dict:
        """Tokens this residual-stream vector pushes up / down (like a feature's output card)."""
        logits = self.W_U @ (self.norm_w * vec / rms)
        top = torch.topk(logits, TOP_K)
        bottom = torch.topk(-logits, TOP_K)
        return {
            "up": [[self.decode(i), float(v)] for v, i in zip(top.values, top.indices)],
            "down": [[self.decode(i), float(-v)] for v, i in zip(bottom.values, bottom.indices)],
        }


# -------------------------------------------------------------------
# 3. Attribution graph + logit lens for one answer token
# -------------------------------------------------------------------

def attribution_graph(tm: TracedModel, cap: dict, ids: list[int], prompt_len: int, answer_index: int) -> dict:
    q = prompt_len + answer_index - 1         # position whose output predicts this token
    token = ids[prompt_len + answer_index]
    logits_q = cap["logits"][q]
    order = torch.argsort(logits_q, descending=True)
    alt = int(next(i for i in order.tolist() if i != token))
    probs_q = torch.softmax(logits_q, dim=-1)

    x_final = cap["resid"][tm.n_layers - 1][q]
    rms = tm.norm_scale(x_final)
    u = tm.norm_w * (tm.W_U[token] - tm.W_U[alt]) / rms   # logit-diff direction, (D,)

    T = len(ids)
    n_rep = tm.n_heads // tm.n_kv
    kv_of_head = torch.arange(tm.n_heads, device=u.device) // n_rep

    candidates = []
    emb_val = float(cap["emb"][q] @ u)
    candidates.append({"type": "embed", "layer": -1, "pos": q, "value": emb_val})
    total = emb_val
    attn_store = {}

    for l, layer in enumerate(tm.layers):
        A = cap["attn"][l][:, q, :T]                                   # (H, T)
        v = cap["v"][l].view(T, tm.n_kv, tm.head_dim)[:, kv_of_head, :]  # (T, H, hd)
        Wo = layer.self_attn.o_proj.weight                               # (D, H*hd)
        g = (Wo.T @ u).view(tm.n_heads, tm.head_dim)                    # (H, hd)
        per_head_src = A * torch.einsum("thd,hd->ht", v, g)             # (H, T)
        src = per_head_src.sum(0)
        total += float(src.sum())
        attn_store[l] = (A, v, Wo, per_head_src)
        for s in range(T):
            candidates.append({"type": "attn", "layer": l, "pos": s, "value": float(src[s])})

        mlp_val = float(cap["mlp"][l][q] @ u)
        total += mlp_val
        candidates.append({"type": "mlp", "layer": l, "pos": q, "value": mlp_val})

    target_diff = float(logits_q[token] - logits_q[alt])

    # Prune: biggest |value| first until NODE_MASS of the total |attribution|.
    candidates.sort(key=lambda c: -abs(c["value"]))
    mass = sum(abs(c["value"]) for c in candidates) or 1.0
    nodes, running = [], 0.0
    for c in candidates:
        if len(nodes) >= MAX_NODES or running / mass >= NODE_MASS:
            break
        nodes.append(c)
        running += abs(c["value"])

    for n, c in enumerate(nodes):
        c["id"] = n
        if c["type"] == "attn":
            A, v, Wo, per_head_src = attn_store[c["layer"]]
            s = c["pos"]
            heads = per_head_src[:, s]
            top_h = torch.argsort(heads.abs(), descending=True)[:3]
            c["heads"] = [[int(h), float(heads[h]), float(A[h, s])] for h in top_h]
            z = (A[:, s, None] * v[s]).reshape(-1)        # (H*hd) head outputs from this source
            vec = Wo @ z
        elif c["type"] == "mlp":
            vec = cap["mlp"][c["layer"]][q]
        else:
            vec = cap["emb"][q]
        c["promotes"] = tm.promoted(vec, rms)

    lens = []
    for l in range(tm.n_layers):
        x = cap["resid"][l][q]
        lp = torch.softmax(tm.model.lm_head(tm.final_norm(x[None]))[0].float(), dim=-1)
        lens.append({
            "layer": l,
            "p_token": float(lp[token]),
            "p_alt": float(lp[alt]),
            "rank": int((lp > lp[token]).sum()) + 1,
        })

    top_next = torch.topk(probs_q, TOP_K)
    return {
        "answer_index": answer_index,
        "position_q": q,
        "token": tm.decode(token),
        "alt": tm.decode(alt),
        "p_token": float(probs_q[token]),
        "p_alt": float(probs_q[alt]),
        "logit_diff": target_diff,
        "reconstructed": total,          # should equal logit_diff (sanity check)
        "covered": running / mass,
        "top_next": [[tm.decode(i), float(p)] for p, i in zip(top_next.values, top_next.indices)],
        "nodes": nodes,
        "lens": lens,
    }


# -------------------------------------------------------------------
# 4. One example
# -------------------------------------------------------------------

def token_sentences(answer: str, sentences: list[str], token_texts: list[str]) -> list[int]:
    starts, cursor = [], 0
    for sentence in sentences:
        found = answer.find(sentence, cursor)
        if found < 0:
            return [0] * len(token_texts)
        starts.append(found)
        cursor = found + len(sentence)
    offsets = np.cumsum([0] + [len(t) for t in token_texts[:-1]])
    first_char = offsets + np.array([len(t) - len(t.lstrip()) for t in token_texts])
    which = np.searchsorted(np.array(starts), first_char, side="right") - 1
    return np.clip(which, 0, len(starts) - 1).astype(int).tolist()


def is_content(text: str) -> bool:
    t = text.strip()
    if any(ch.isdigit() for ch in t):
        return True
    return len(t) >= 2 and any(ch.isalpha() for ch in t) and t.lower() not in STOPWORDS


def build_example(row, category: str, samples: list[str], tm: TracedModel, nli: NLIModel, youden: float | None) -> dict:
    p_ids = tm.prompt_ids(row.prompt)
    a_ids = tm.tok(row.main_answer, add_special_tokens=False)["input_ids"]
    ids = p_ids + a_ids
    P = len(p_ids)
    cap = tm.run(ids)

    # per-token stats for the answer
    logits = cap["logits"][P - 1 : P - 1 + len(a_ids)]
    logp = torch.log_softmax(logits, dim=-1)
    probs = logp.exp()
    entropy = -(probs * logp).sum(-1)
    chosen = logp.gather(1, torch.tensor(a_ids, device=logp.device)[:, None])[:, 0]
    top = torch.topk(probs, TOP_K, dim=-1)
    a_texts = [tm.decode(i) for i in a_ids]

    sentences = split_sentences(row.main_answer)
    sent_of_token = token_sentences(row.main_answer, sentences, a_texts)
    nli_matrix = selfcheck_nli_batch(nli, [(sentences, samples)])[0]   # (n_samples, n_sentences)
    sent_scores = nli_matrix.mean(axis=0)

    answer_tokens = []
    for i, text in enumerate(a_texts):
        answer_tokens.append({
            "text": text,
            "logprob": float(chosen[i]),
            "entropy": float(entropy[i]),
            "sentence": sent_of_token[i],
            "top": [[tm.decode(t), float(p)] for p, t in zip(top.values[i], top.indices[i])],
        })

    # Targets: most surprising content tokens of the sentence SelfCheck distrusts most.
    worst = int(np.argmax(sent_scores))
    surprisal = (-chosen).tolist()
    pool = [i for i in range(len(a_ids)) if i > 0 and is_content(a_texts[i])]
    in_worst = sorted([i for i in pool if sent_of_token[i] == worst], key=lambda i: -surprisal[i])
    rest = sorted([i for i in pool if sent_of_token[i] != worst], key=lambda i: -surprisal[i])
    targets = (in_worst + rest)[:N_TARGETS]

    graphs = [attribution_graph(tm, cap, ids, P, i) for i in sorted(targets)]

    cluster_ids = list(row.cluster_ids) if row.cluster_ids is not None else []
    clusters = {}
    for k, cid in enumerate(cluster_ids):
        if k < len(samples):
            clusters.setdefault(int(cid), []).append(samples[k][:400])
    clusters = sorted(clusters.values(), key=len, reverse=True)

    def det(value, pct):
        return {"value": None if pd.isna(value) else float(value),
                "pct": None if pd.isna(pct) else float(pct)}

    detectors = {
        "selfcheck_nli": {"value": float(sent_scores.mean()), "pct": None},
        "semantic_entropy": det(row.semantic_entropy, row.semantic_entropy_pct),
        "n_clusters": {"value": None if pd.isna(row.n_clusters) else int(row.n_clusters),
                       "of": None if pd.isna(row.n_answers) else int(row.n_answers)},
        "sep_tbg": det(row.sep_tbg, row.sep_tbg_pct),
        "sep_slt": det(row.sep_slt, row.sep_slt_pct),
        "accprobe_tbg": det(row.accprobe_tbg, row.accprobe_tbg_pct),
        "logprob_neg_mean": det(row.logprob_neg_mean, row.logprob_neg_mean_pct),
        "logprob_youden_threshold": youden,
    }

    roles = tm.roles(ids, P)
    return {
        "uid": row.uid,
        "subset": row.subset,
        "category": category,
        "prompt": row.prompt,
        "context": None if pd.isna(row.context) else str(row.context),
        "reference_correct": [str(x) for x in list(row.reference_correct)],
        "reference_incorrect": [str(x) for x in list(row.reference_incorrect)],
        "label": {
            "hallucinated": None if pd.isna(row.hallucinated) else int(row.hallucinated),
            "method": row.label_method,
            "correct_score": None if pd.isna(row.correct_score) else float(row.correct_score),
        },
        "answer": row.main_answer,
        "prompt_len": P,
        "tokens": [{"text": tm.decode(t), "role": r} for t, r in zip(ids, roles)],
        "answer_tokens": answer_tokens,
        "sentences": [
            {"text": s, "selfcheck_nli": float(sent_scores[k]),
             "per_sample": [float(x) for x in nli_matrix[:, k]]}
            for k, s in enumerate(sentences)
        ],
        "detectors": detectors,
        "clusters": clusters,
        "targets": graphs,
    }


def main():
    args = parse_args()
    started = time.time()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    frame = load_frame()
    if args.uids:
        plan = [(u, "hand-picked") for u in args.uids]
    else:
        plan = select_examples(frame)
    print(f"[INFO] {len(plan)} examples: {[u for u, _ in plan]}")

    rows = frame.set_index("uid", drop=False)
    samples = load_samples([u for u, _ in plan])
    tm = TracedModel(args.llm, device)
    nli = NLIModel(args.nli_model, device)
    youden = b6_youden_threshold()

    examples = []
    for uid, category in plan:
        print(f"[INFO] {uid} ({category})")
        # The forward pass runs in inference mode, so every computation on its
        # tensors (the attribution math) must too.
        with torch.inference_mode():
            examples.append(build_example(rows.loc[uid], category, samples.get(uid, []), tm, nli, youden))
        errors = [abs(g["logit_diff"] - g["reconstructed"]) for g in examples[-1]["targets"]]
        if errors:
            print(f"       attribution reconstruction error: {max(errors):.2e} logits")
        else:
            print("       [WARNING] no content tokens to trace in this answer")

    payload = {
        "meta": {
            "model": Path(args.llm).name,
            "n_layers": tm.n_layers,
            "n_heads": tm.n_heads,
            "nli_model": Path(args.nli_model).name,
            "git_commit_sha": git_commit_sha(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "node_mass": NODE_MASS,
            "max_nodes": MAX_NODES,
        },
        "examples": examples,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"[SUCCESS] {OUTPUT_PATH} ({size_mb:.1f} MB) in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
