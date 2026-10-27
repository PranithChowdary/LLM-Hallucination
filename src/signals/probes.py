"""
B5: Semantic Entropy Probes (SEPs).

    Kossen, Han, Razzak, Schut, Malik & Gal (2024), "Semantic Entropy
    Probes: Robust and Cheap Hallucination Detection in LLMs"
        https://arxiv.org/abs/2406.15927
    Reference implementation: https://github.com/OATML/semantic-entropy-probes

Idea: semantic entropy (B4) needs M extra generations + O(M^2) NLI
calls per prompt. SEPs replace that at test time with ONE linear probe
on the hidden state of a single generation:

    1. Compute SE for every training prompt (B4 output).
    2. Binarize it: high-SE vs low-SE. The threshold gamma* minimizes
       the within-class squared error (best split of a 1-D regression
       tree):
           gamma* = argmin_g  sum_{H < g}(H - mean_low)^2
                            + sum_{H >= g}(H - mean_high)^2
    3. Train logistic regression on hidden states h_l(x) to predict the
       binarized SE, separately for each layer l and two positions:
           TBG = token before generation (last prompt token; the probe
                 can then fire BEFORE any answer is generated)
           SLT = second-last token = last answer token before EOS
    4. Use P(high SE | h) as a hallucination score. Compare with an
       "accuracy probe" trained directly on hallucination labels, which
       the paper uses as the main baseline.

This is the closest prior art to our project (single pass, linear,
hidden-state based), so the comparison with B8 matters.

Evaluation: grouped 5-fold cross-fitting (groups = norm_key, so
near-duplicate prompts never sit on both sides). Every uid gets an
out-of-fold probe score, which later blocks (C/D) can use like any
other signal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.signals.common import chat_messages


# -------------------------------------------------------------------
# 1. Hidden-state extraction (TBG and SLT, every layer)
# -------------------------------------------------------------------

class ProbeStateExtractor:
    """
    Teacher-forced pass over (A2 prompt, A2 main answer), keeping the
    hidden state at TBG and SLT for all layers (embedding output
    included, so n_layers + 1 rows).
    """

    def __init__(self, model_path: str, device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.signals.common import pick_device

        self.torch = torch
        self.device = device or pick_device()
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype
        ).to(self.device)
        self.model.eval()

    def _prompt_ids(self, prompt: str) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            chat_messages(prompt), tokenize=True, add_generation_prompt=True
        )
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        return list(encoded)

    def extract(self, prompts: list[str], answers: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (tbg, slt), each float16 of shape (n, n_layers + 1, hidden)."""
        torch = self.torch
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        tbg_all, slt_all = [], []

        for start in range(0, len(prompts), self.batch_size):
            p_chunk = prompts[start : start + self.batch_size]
            a_chunk = answers[start : start + self.batch_size]

            seqs, tbg_pos, slt_pos = [], [], []
            for prompt, answer in zip(p_chunk, a_chunk):
                p_ids = self._prompt_ids(prompt)
                a_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
                seqs.append(p_ids + a_ids)
                tbg_pos.append(len(p_ids) - 1)
                slt_pos.append(len(p_ids) + max(len(a_ids), 1) - 1)

            width = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            attention = torch.zeros((len(seqs), width), dtype=torch.long)
            for row, seq in enumerate(seqs):
                input_ids[row, : len(seq)] = torch.tensor(seq)
                attention[row, : len(seq)] = 1

            with torch.inference_mode():
                # backbone only: SEPs need hidden states, not logits
                out = self.model.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention.to(self.device),
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden = torch.stack(out.hidden_states, dim=1)  # (b, L+1, T, H)
            rows = torch.arange(len(seqs), device=hidden.device)

            tbg = hidden[rows, :, torch.tensor(tbg_pos, device=hidden.device)]
            slt = hidden[rows, :, torch.tensor(slt_pos, device=hidden.device)]
            tbg_all.append(tbg.to(torch.float16).cpu().numpy())
            slt_all.append(slt.to(torch.float16).cpu().numpy())

        return np.concatenate(tbg_all), np.concatenate(slt_all)


def load_a3_last_token_states(uids: list[str], hidden_dir: Path) -> np.ndarray:
    """
    Fallback when re-extraction is not possible: A3 already saved the
    FINAL-layer hidden state of every answer token. Its last row is the
    SLT position (A3 decoded without EOS). Returns (n, 1, hidden).
    TBG and intermediate layers are not available from A3.
    """
    import torch

    states = []
    for uid in uids:
        path = Path(hidden_dir) / (hashlib.sha256(uid.encode("utf-8")).hexdigest() + ".pt")
        record = torch.load(path, map_location="cpu")
        states.append(record["hidden_states"][-1].numpy())
    return np.stack(states)[:, None, :]


# -------------------------------------------------------------------
# 2. Binarizing semantic entropy
# -------------------------------------------------------------------

def best_split_threshold(entropy: np.ndarray) -> float:
    """
    gamma* minimizing within-class sum of squared errors. Candidate
    thresholds are midpoints between consecutive distinct values;
    labels are (entropy >= gamma*). O(n log n) via prefix sums.
    """
    h = np.sort(np.asarray(entropy, dtype=float))
    h = h[np.isfinite(h)]
    values = np.unique(h)
    if len(values) < 2:
        raise ValueError("Semantic entropy is constant; cannot binarize.")

    prefix = np.cumsum(h)
    prefix_sq = np.cumsum(h**2)
    n = len(h)
    total, total_sq = prefix[-1], prefix_sq[-1]

    def sse(s, sq, k):
        return sq - s * s / k

    best_cost, best_gamma = np.inf, None
    for low, high in zip(values[:-1], values[1:]):
        gamma = 0.5 * (low + high)
        k = int(np.searchsorted(h, gamma))  # count below gamma
        cost = sse(prefix[k - 1], prefix_sq[k - 1], k) + sse(
            total - prefix[k - 1], total_sq - prefix_sq[k - 1], n - k
        )
        if cost < best_cost:
            best_cost, best_gamma = cost, gamma
    return float(best_gamma)


# -------------------------------------------------------------------
# 3. Probe training with grouped cross-fitting
# -------------------------------------------------------------------

def grouped_folds(
    frame: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    """Fold id per row: stratified by subset, grouped by norm_key."""
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(frame), -1)
    for fold, (_, test_idx) in enumerate(
        splitter.split(frame, frame["subset"], groups=frame["norm_key"])
    ):
        folds[test_idx] = fold
    return folds


def make_probe(c: float = 1.0, seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, random_state=seed),
    )


def cross_fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    binarize_entropy: bool = False,
    c: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """
    Out-of-fold P(y = 1 | x).

    If binarize_entropy, `y` is continuous SE and gamma* is re-fitted on
    each training fold only (fitting it on all data would leak the test
    folds' SE distribution into the labels).
    Rows with NaN targets are skipped in training but still scored.
    """
    scores = np.full(len(X), np.nan)
    gammas = []
    for fold in np.unique(folds):
        train = (folds != fold) & np.isfinite(y)
        test = folds == fold

        y_train = y[train]
        if binarize_entropy:
            gamma = best_split_threshold(y_train)
            gammas.append(gamma)
            y_train = (y_train >= gamma).astype(int)
        else:
            y_train = y_train.astype(int)

        probe = make_probe(c=c, seed=seed)
        probe.fit(X[train].astype(np.float32), y_train)
        scores[test] = probe.predict_proba(X[test].astype(np.float32))[:, 1]
    return scores, gammas
