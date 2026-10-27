"""
B4: Semantic Entropy.

    Kuhn, Gal & Farquhar (ICLR 2023), "Semantic Uncertainty: Linguistic
    Invariances for Uncertainty Estimation in Natural Language Generation"
        https://arxiv.org/abs/2302.09664
    Farquhar, Kossen, Kuhn & Gal (Nature 2024), "Detecting hallucinations
    in large language models using semantic entropy"
        https://www.nature.com/articles/s41586-024-07421-0
    Reference implementation:
        https://github.com/jlko/semantic_uncertainty

Pipeline per prompt x with sampled answers s_1..s_M:
    1. Score each sample: log p(s_m | x) from a teacher-forced pass.
    2. Bidirectional entailment clustering (Kuhn et al., Algorithm 1):
       s_m joins cluster C if s_m <=> C[0] (each entails the other, judged
       by an NLI model on "question + answer" text); otherwise it starts
       a new cluster.
    3. Semantic entropy (Rao estimator, as in the reference code):
           p(C | x) = sum_{s in C} p(s | x)  /  sum_{all s} p(s | x)
           SE(x)    = - sum_C p(C | x) log p(C | x)
       using length-normalized sequence log-likelihoods.
    4. Discrete semantic entropy (Farquhar et al. 2024): same, but
       p(C | x) = |C| / M. Needs no log-likelihoods at all.

Baseline reported alongside: naive predictive entropy
    H_naive(x) = - (1/M) sum_m (1/len(s_m)) log p(s_m | x)

Departures from the papers (documented, not hidden):
    - Papers sample at T = 1.0 (M = 10); ours are T = 0.7, top_p = 0.9
      (M = 15) because that is what A2/generate_extra.py produced.
    - Log-likelihoods are recomputed by a teacher-forced pass over the
      saved text (the same approach and caveat as A3), not taken from
      the original sampling run.
    - Our answers are paragraphs (~65 words), not short phrases.
      Farquhar et al. handle paragraphs by decomposing into factoids
      first; we apply sentence-pair NLI to whole answers, which makes
      almost every answer its own cluster. Expect SE to look like
      log(M) for many prompts. This is a known limitation to discuss.
"""

from __future__ import annotations

import numpy as np

from src.signals.common import NLIModel, chat_messages


CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2


# -------------------------------------------------------------------
# 1. Sequence log-likelihoods
# -------------------------------------------------------------------

class SequenceScorer:
    """Teacher-forced log p(answer | prompt) under the generating LLM."""

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

    def prompt_ids(self, prompt: str) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            chat_messages(prompt), tokenize=True, add_generation_prompt=True
        )
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        return list(encoded)

    def score(self, prompt: str, answers: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (sum_logprob, n_tokens) per answer. Length-normalized
        log-likelihood is sum_logprob / n_tokens.
        """
        torch = self.torch
        p_ids = self.prompt_ids(prompt)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        sums, lengths = [], []
        for start in range(0, len(answers), self.batch_size):
            chunk = answers[start : start + self.batch_size]
            a_ids = [
                self.tokenizer(a, add_special_tokens=False)["input_ids"] or [pad]
                for a in chunk
            ]
            seqs = [p_ids + a for a in a_ids]
            width = max(len(s) for s in seqs)

            input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            attention = torch.zeros((len(seqs), width), dtype=torch.long)
            for row, seq in enumerate(seqs):
                input_ids[row, : len(seq)] = torch.tensor(seq)
                attention[row, : len(seq)] = 1

            input_ids = input_ids.to(self.device)
            attention = attention.to(self.device)

            with torch.inference_mode():
                # Run the backbone once, then the LM head only on answer
                # positions: full-vocab logits for every prompt token of
                # every row would not fit in memory.
                hidden = self.model.model(
                    input_ids=input_ids, attention_mask=attention, use_cache=False
                ).last_hidden_state

                for row, a in enumerate(a_ids):
                    # hidden state at position t predicts token t + 1
                    begin = len(p_ids) - 1
                    logits = self.model.lm_head(hidden[row, begin : begin + len(a)]).float()
                    targets = torch.tensor(a, device=logits.device)
                    lp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])
                    sums.append(float(lp.sum()))
                    lengths.append(len(a))

        return np.array(sums), np.array(lengths)


# -------------------------------------------------------------------
# 2. Bidirectional entailment clustering
# -------------------------------------------------------------------

def nli_pair_matrix(
    nli: NLIModel,
    question: str,
    answers: list[str],
    condition_on_question: bool = True,
) -> np.ndarray:
    """
    (M, M, 3) NLI probabilities, [i, j] = P(label | premise=a_i, hyp=a_j),
    columns [contradiction, neutral, entailment]. The diagonal is set to
    certain entailment.
    """
    return nli_pair_matrices(nli, [(question, answers)], condition_on_question)[0]


def nli_pair_matrices(
    nli: NLIModel,
    items: list[tuple[str, list[str]]],
    condition_on_question: bool = True,
) -> list[np.ndarray]:
    """Batched version of nli_pair_matrix over several prompts."""
    premises, hypotheses, index = [], [], []
    for p, (question, answers) in enumerate(items):
        texts = [f"{question} {a}" if condition_on_question else a for a in answers]
        for i in range(len(texts)):
            for j in range(len(texts)):
                if i != j:
                    premises.append(texts[i])
                    hypotheses.append(texts[j])
                    index.append((p, i, j))

    probs = nli.probs(premises, hypotheses)

    out = []
    for _, answers in items:
        m = np.zeros((len(answers), len(answers), 3), dtype=np.float32)
        m[np.arange(len(answers)), np.arange(len(answers)), ENTAILMENT] = 1.0
        out.append(m)
    for (p, i, j), row in zip(index, probs):
        out[p][i, j] = row
    return out


def are_equivalent(matrix: np.ndarray, i: int, j: int, strict: bool = True) -> bool:
    """
    strict=True  (Kuhn et al. 2023): both directions are argmax-entailment.
    strict=False (reference code option): neither direction is a
                 contradiction and not both directions are neutral.
    """
    a = int(matrix[i, j].argmax())
    b = int(matrix[j, i].argmax())
    if strict:
        return a == ENTAILMENT and b == ENTAILMENT
    return CONTRADICTION not in (a, b) and (a, b) != (NEUTRAL, NEUTRAL)


def cluster_ids(matrix: np.ndarray, strict: bool = True) -> list[int]:
    """
    Kuhn et al. Algorithm 1. Each answer is compared with the FIRST
    member of each existing cluster, in creation order, and joins the
    first match.

    Because answer m only looks at answers < m, clustering the first k
    answers gives exactly the first k ids of clustering all M answers.
    That makes the B9 budget sweep (N = 1, 2, 4, 8, 16) nested for free.
    """
    ids: list[int] = []
    representatives: list[int] = []
    for m in range(matrix.shape[0]):
        for cluster, rep in enumerate(representatives):
            if are_equivalent(matrix, rep, m, strict):
                ids.append(cluster)
                break
        else:
            ids.append(len(representatives))
            representatives.append(m)
    return ids


# -------------------------------------------------------------------
# 3. Entropies
# -------------------------------------------------------------------

def _logsumexp(x: np.ndarray) -> float:
    top = np.max(x)
    return float(top + np.log(np.exp(x - top).sum()))


def semantic_entropy(ids: list[int], log_likelihoods: np.ndarray) -> float:
    """
    Rao estimator over semantic clusters, as in the reference code
    (logsumexp_by_id with agg='sum_normalized', then predictive_entropy_rao).
    `log_likelihoods` should be length-normalized (mean token log-prob).
    """
    ids = np.asarray(ids)
    ll = np.asarray(log_likelihoods, dtype=float)
    total = _logsumexp(ll)
    log_p = np.array([_logsumexp(ll[ids == c]) - total for c in np.unique(ids)])
    return float(-(np.exp(log_p) * log_p).sum())


def discrete_semantic_entropy(ids: list[int]) -> float:
    """Entropy of the empirical cluster distribution |C| / M."""
    _, counts = np.unique(np.asarray(ids), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def naive_entropy(log_likelihoods: np.ndarray) -> float:
    """Monte-Carlo predictive entropy over raw sequences (no clustering)."""
    return float(-np.mean(log_likelihoods))
