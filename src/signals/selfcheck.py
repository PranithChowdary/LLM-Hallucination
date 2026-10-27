"""
B3: SelfCheckGPT (Manakul, Liusie & Gales, EMNLP 2023).
    https://arxiv.org/abs/2303.08896
    Reference implementation: https://github.com/potsawee/selfcheckgpt

Idea: if the model knows a fact, stochastic re-samples agree with the
main answer; if it hallucinated, re-samples diverge. Each SENTENCE r_i
of the main answer R gets an inconsistency score S(i) in [0, 1]
(higher = more likely hallucinated) by comparing it with N samples
S^1..S^N. The passage score is the mean over sentences.

Variants implemented (paper section numbers):
    5.1 BERTScore   S(i) = 1 - (1/N) sum_n max_k BERTScore(r_i, s^n_k)
    5.3 Unigram     unigram LM fit on R + all samples; per-token -log p
                    -> Avg(-log p) and Max(-log p)
    5.4 NLI         S(i) = (1/N) sum_n P(contradict | premise=S^n, hyp=r_i)
                    with P(contradict) = e^{z_c} / (e^{z_e} + e^{z_c})
    5.5 Prompt      ask an LLM "Is the sentence supported by the context
                    above? Answer Yes or No." Yes -> 0, No -> 1, else 0.5

Not implemented: 5.2 MQAG (question generation + answering). The paper
reports it below NLI and Prompt, and it needs two extra models.

Departure from the paper (documented, not hidden):
    - The paper uses N = 20 samples at temperature 1.0 and the main
      answer from greedy decoding. Our A2 main answer and the 15 extra
      samples are all sampled at T = 0.7, top_p = 0.9 (N = 15).
"""

from __future__ import annotations

import re

import numpy as np

from src.signals.common import NLIModel, split_sentences


# -------------------------------------------------------------------
# 5.3 Unigram (n-gram with n = 1)
# -------------------------------------------------------------------

_TOKEN = re.compile(r"\w+|[^\w\s]")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def selfcheck_unigram(
    main_sentences: list[str], samples: list[str], smoothing_k: float = 0.0
) -> dict:
    """
    Unigram LM trained on the main answer + all samples, exactly as in
    the reference implementation (k = 0: the main answer is part of the
    training text, so every token it contains has non-zero count).

    Returns sentence-level scores and the two passage-level scores the
    paper reports: avg_neg_logprob (Avg) and avg_max_neg_logprob (Max).
    """
    counts: dict[str, int] = {"<unk>": 0}
    total = 0
    for text in [" ".join(main_sentences), *samples]:
        for token in _tokens(text):
            counts[token] = counts.get(token, 0) + 1
            total += 1

    denom = total + smoothing_k * len(counts)

    def logprob(token: str) -> float:
        token = token if token in counts else "<unk>"
        p = (counts[token] + smoothing_k) / denom
        return float(np.log(p)) if p > 0 else float("-inf")

    sent_avg, sent_max, all_logprobs = [], [], []
    for sentence in main_sentences:
        lp = [logprob(t) for t in _tokens(sentence)] or [0.0]
        all_logprobs.extend(lp)
        sent_avg.append(-float(np.mean(lp)))
        sent_max.append(-float(np.min(lp)))

    return {
        "sent_avg_neg_logprob": sent_avg,
        "sent_max_neg_logprob": sent_max,
        "avg_neg_logprob": -float(np.mean(all_logprobs)),
        "avg_max_neg_logprob": float(np.mean(sent_max)),
    }


# -------------------------------------------------------------------
# 5.4 NLI
# -------------------------------------------------------------------

def contradiction_score(probs: np.ndarray) -> np.ndarray:
    """
    P(contradict) renormalized over {entail, contradict}, ignoring
    neutral, as in the paper. `probs` columns: [contra, neutral, entail].
    """
    contra, entail = probs[:, 0], probs[:, 2]
    return contra / np.clip(contra + entail, 1e-12, None)


def selfcheck_nli_batch(
    nli: NLIModel, items: list[tuple[list[str], list[str]]]
) -> list[np.ndarray]:
    """
    items: list of (main_sentences, evidence_passages) for several prompts.
    Returns, per prompt, a (n_samples, n_sentences) matrix of
    P(contradict | sample n, sentence i). S_NLI(i) is its column mean;
    keeping the per-sample rows lets the N-sweep (paper Fig. 7) reuse
    one run: S_NLI with N samples = mean of the first N rows.

    All (sample, sentence) pairs of several prompts go through the NLI
    model together, which is what makes 24k prompts x 15 samples feasible.
    """
    premises, hypotheses, shapes = [], [], []
    for sentences, samples in items:
        for sample in samples:
            for sentence in sentences:
                premises.append(sample)
                hypotheses.append(sentence)
        shapes.append((len(samples), len(sentences)))

    scores = contradiction_score(nli.probs(premises, hypotheses))

    out, offset = [], 0
    for n_samples, n_sentences in shapes:
        size = n_samples * n_sentences
        out.append(scores[offset : offset + size].reshape(n_samples, n_sentences))
        offset += size
    return out


# -------------------------------------------------------------------
# 5.1 BERTScore
# -------------------------------------------------------------------

class BERTScoreScorer:
    """
    Uses the `bert-score` package with the SelfCheckGPT defaults
    (roberta-large, layer 17, rescale_with_baseline=True).

    `model_path` may point at a local copy of roberta-large (offline
    SLURM jobs). bert-score looks its rescale baseline up by model name,
    so for a local path the packaged roberta-large baseline is passed
    explicitly.
    """

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 128,
        model_path: str | None = None,
    ):
        import os

        import bert_score
        from bert_score import BERTScorer

        kwargs = {}
        if model_path:
            kwargs = {
                "model_type": str(model_path),
                "num_layers": 17,
                "baseline_path": os.path.join(
                    os.path.dirname(bert_score.__file__),
                    "rescale_baseline", "en", "roberta-large.tsv",
                ),
            }

        self.scorer = BERTScorer(
            lang="en",
            rescale_with_baseline=True,
            device=device,
            batch_size=batch_size,
            **kwargs,
        )

    def score_batch(
        self, items: list[tuple[list[str], list[str]]]
    ) -> list[np.ndarray]:
        """
        Per prompt, a (n_samples, n_sentences) matrix of
        max_k BERTScore(r_i, s^n_k). S_BERT(i) = 1 - column mean.
        """
        cands, refs, index = [], [], []
        for p, (sentences, samples) in enumerate(items):
            for n, sample in enumerate(samples):
                sample_sentences = split_sentences(sample)
                for i, sentence in enumerate(sentences):
                    for sample_sentence in sample_sentences:
                        cands.append(sentence)
                        refs.append(sample_sentence)
                        index.append((p, n, i))

        _, _, f1 = self.scorer.score(cands, refs, verbose=False)
        f1 = f1.cpu().numpy()

        # max over sample sentences, then mean over samples
        best = {}
        for (p, n, i), value in zip(index, f1):
            key = (p, n, i)
            best[key] = max(best.get(key, -np.inf), float(value))

        out = []
        for p, (sentences, samples) in enumerate(items):
            matrix = np.array([
                [best[(p, n, i)] for i in range(len(sentences))]
                for n in range(len(samples))
            ])
            out.append(matrix)
        return out


# -------------------------------------------------------------------
# 5.5 Prompt
# -------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Context: {context}\n\n"
    "Sentence: {sentence}\n\n"
    "Is the sentence supported by the context above? "
    "Answer Yes or No.\n\n"
    "Answer: "
)


class PromptScorer:
    """
    LLM-prompt variant.

    The reference code generates up to 5 tokens greedily and maps text
    starting with "yes" -> 0.0, "no" -> 1.0, anything else -> 0.5. Here
    we run ONE batched forward pass and read the greedy first token,
    which gives the same decision without autoregressive generation.
    We also return p_no = P(No) / (P(Yes) + P(No)) as a continuous score.

    The paper used GPT-3.5; with Qwen2.5-0.5B this variant will be weak.
    Point `model_path` at a bigger local model for a fairer reproduction.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        batch_size: int = 32,
        use_chat_template: bool = True,
        max_length: int = 1024,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.signals.common import pick_device

        self.torch = torch
        self.device = device or pick_device()
        self.batch_size = batch_size
        self.use_chat_template = use_chat_template
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype
        ).to(self.device)
        self.model.eval()

        self.yes_ids = self._first_token_ids(["Yes", " Yes", "yes", " yes"])
        self.no_ids = self._first_token_ids(["No", " No", "no", " no"])

    def _first_token_ids(self, words: list[str]) -> list[int]:
        ids = {self.tokenizer.encode(w, add_special_tokens=False)[0] for w in words}
        return sorted(ids)

    def _render(self, context: str, sentence: str) -> str:
        text = PROMPT_TEMPLATE.format(context=context, sentence=sentence)
        if not self.use_chat_template:
            return text
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _judge(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        decisions, p_no = [], []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start : start + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with self.torch.inference_mode():
                # left padding: the last position is the next-token slot
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :].float()

            greedy = logits.argmax(dim=-1).cpu().tolist()
            probs = self.torch.softmax(logits, dim=-1)
            p_yes = probs[:, self.yes_ids].sum(dim=-1)
            p_n = probs[:, self.no_ids].sum(dim=-1)
            p_no.extend((p_n / (p_yes + p_n).clamp_min(1e-12)).cpu().tolist())

            for token_id in greedy:
                word = self.tokenizer.decode([token_id]).strip().lower()
                if word.startswith("yes"):
                    decisions.append(0.0)
                elif word.startswith("no"):
                    decisions.append(1.0)
                else:
                    decisions.append(0.5)
        return np.array(decisions), np.array(p_no)

    def score_batch(
        self, items: list[tuple[list[str], list[str]]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Per prompt, (decision, p_no) matrices of shape
        (n_samples, n_sentences). S_prompt(i) = column mean of decision.
        """
        texts, shapes = [], []
        for sentences, samples in items:
            for sample in samples:
                for sentence in sentences:
                    texts.append(self._render(sample, sentence))
            shapes.append((len(samples), len(sentences)))

        decisions, p_no = self._judge(texts)

        out, offset = [], 0
        for n_samples, n_sentences in shapes:
            size = n_samples * n_sentences
            d = decisions[offset : offset + size].reshape(n_samples, n_sentences)
            q = p_no[offset : offset + size].reshape(n_samples, n_sentences)
            out.append((d, q))
            offset += size
        return out


# -------------------------------------------------------------------
# Grey-box baselines (paper Section 6 / Table 2: "LLM's probabilities")
# -------------------------------------------------------------------

def greybox_scores(
    answer: str,
    sentences: list[str],
    token_texts: list[str],
    log_probs: np.ndarray,
    entropies: np.ndarray,
) -> dict:
    """
    The paper's grey-box baselines from the generating model's own
    token probabilities (we take them from A3's teacher-forced pass):

        sentence i:  Avg(-log p) = mean_j -log p_ij   Max(-log p) = max_j -log p_ij
                     Avg(H)      = mean_j H_ij        Max(H)      = max_j H_ij
        passage:     mean over sentences

    Tokens are assigned to sentences by character offset. If a sentence
    cannot be located in the answer text, the whole answer is treated
    as one sentence (flagged by `aligned=False`).
    """
    nll = -np.asarray(log_probs, dtype=float)
    ent = np.asarray(entropies, dtype=float)

    # character offset of each token's first non-space character, so a
    # token like " It" belongs to the sentence that starts at "It"
    offsets = np.cumsum([0] + [len(t) for t in token_texts[:-1]])
    token_starts = offsets + np.array([len(t) - len(t.lstrip()) for t in token_texts])
    sentence_starts, cursor, aligned = [], 0, True
    for sentence in sentences:
        found = answer.find(sentence, cursor)
        if found < 0:
            aligned = False
            break
        sentence_starts.append(found)
        cursor = found + len(sentence)

    if aligned and sentence_starts:
        which = np.searchsorted(np.array(sentence_starts), token_starts, side="right") - 1
        which = np.clip(which, 0, len(sentence_starts) - 1)
    else:
        which = np.zeros(len(nll), dtype=int)

    groups = [g for g in np.unique(which)]
    avg_nll = [nll[which == g].mean() for g in groups]
    max_nll = [nll[which == g].max() for g in groups]
    avg_h = [ent[which == g].mean() for g in groups]
    max_h = [ent[which == g].max() for g in groups]
    return {
        "greybox_avg_neg_logprob": float(np.mean(avg_nll)),
        "greybox_max_neg_logprob": float(np.mean(max_nll)),
        "greybox_avg_entropy": float(np.mean(avg_h)),
        "greybox_max_entropy": float(np.mean(max_h)),
        "greybox_aligned": aligned,
    }
