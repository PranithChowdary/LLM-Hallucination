"""
Shared helpers for the sampling-based detectors (B3, B4) and probes (B5).

Everything that more than one of B3/B4/B5 needs lives here:
    - paths to the A2 answers and the extra sampled answers
    - loading "main answer + N samples" per prompt
    - rendering the prompt exactly the way A2 did
    - sentence splitting
    - an NLI wrapper that works with any MNLI-style checkpoint
    - sharded, resumable result writing (same idea as A3)
    - loading the hallucination labels (metrics live in src/eval/metrics.py)

Terminology used in B3/B4/B5:
    main answer  = response_index 1 = the A2 answer in
                   data/generation_qwen_calibration.parquet
    samples      = response_index 2..16 from
                   data/generation_qwen_calibration_extra15.parquet
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
LLM_DIR = MODELS_DIR / "Qwen2.5-0.5B-Instruct"

CALIBRATION_PATH = PROJECT_ROOT / "data" / "calibration.parquet"
MAIN_ANSWERS_PATH = PROJECT_ROOT / "data" / "generation_qwen_calibration.parquet"
SAMPLES_PATH = (
    PROJECT_ROOT / "data" / "generation_qwen_calibration_extra15.parquet"
)
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "generation_labels.parquet"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / "data"  # B4/B5/B6 write to data/b4, data/b5, data/b6


def git_commit_sha() -> str:
    """Return the current commit so every output is traceable to code."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Loading main answers + samples
def load_prompts() -> pd.DataFrame:
    """Calibration prompts with references, indexed by uid."""
    prompts = pd.read_parquet(CALIBRATION_PATH)
    prompts["uid"] = prompts["uid"].astype(str)
    return prompts.set_index("uid", drop=False)


def load_responses(
    n_samples: int | None = None,
    subsets: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Return one row per prompt with:
        uid, dataset, subset, prompt, main_answer, samples (list[str])

    `samples` holds response_index 2..(n_samples + 1), in index order, so
    the first k samples are always the same k answers (needed for the
    B9 budget sweep to be nested).
    """
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLES_PATH} not found. Run "
            "src/generation/generate_extra.py first (15 extra samples "
            "per prompt), or copy its output from the cluster."
        )

    main = pd.read_parquet(MAIN_ANSWERS_PATH)
    main["uid"] = main["uid"].astype(str)

    extra = pd.read_parquet(
        SAMPLES_PATH, columns=["uid", "response_index", "generated_answer"]
    )
    extra["uid"] = extra["uid"].astype(str)

    if extra.duplicated(["uid", "response_index"]).any():
        raise ValueError("Duplicate (uid, response_index) in the samples file.")

    if n_samples is not None:
        extra = extra[extra["response_index"] <= n_samples + 1]

    extra = extra.sort_values(["uid", "response_index"], kind="stable")
    samples = extra.groupby("uid")["generated_answer"].agg(list)

    out = main[["uid", "dataset", "subset", "prompt", "generated_answer"]]
    out = out.rename(columns={"generated_answer": "main_answer"})
    out = out.merge(samples.rename("samples"), left_on="uid", right_index=True)

    missing = set(main["uid"]) - set(out["uid"])
    if missing:
        print(
            f"[WARNING] {len(missing):,} prompts have no samples yet and "
            "are skipped (generate_extra.py not finished?)."
        )

    counts = out["samples"].map(len)
    expected = counts.max() if n_samples is None else n_samples
    short = counts < expected
    if short.any():
        print(
            f"[WARNING] {int(short.sum()):,} prompts have fewer than "
            f"{expected} samples; they are kept with what they have."
        )

    if subsets:
        out = out[out["subset"].isin(subsets)]

    out = out.sort_values("uid", kind="stable").reset_index(drop=True)

    if limit is not None:
        out = take_limit(out, limit)

    return out


def take_limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """
    Quick-test subset: the first ceil(limit / n_subsets) uids of EACH
    subset (in uid order), so a test run sees every subset and both
    labels. Every script uses this, so labels, B4 and B5 pick the same uids.
    """
    if not limit:
        return frame
    per_subset = -(-limit // frame["subset"].nunique())
    out = frame.sort_values("uid", kind="stable").groupby("subset", sort=False).head(per_subset)
    return out.sort_values("uid", kind="stable").reset_index(drop=True)


def chat_messages(prompt: str) -> list[dict]:
    """
    The exact message list A2 (generate.py) and generate_extra.py used.

    Every detector that re-scores or re-reads an answer must render the
    prompt the same way, otherwise the hidden states / log-likelihoods
    belong to a different conditional distribution than the one that
    produced the answer. If A2 is ever changed (e.g. to include the
    `context` field), change it here and nowhere else.
    """
    return [{"role": "user", "content": prompt}]


# Sentence splitting
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str, min_chars: int = 3) -> list[str]:
    """
    Split a passage into sentences.

    SelfCheckGPT uses spaCy's sentence splitter. spaCy is used when it
    is installed (`pip install spacy && python -m spacy download
    en_core_web_sm`); otherwise a regex splitter is used, which is close
    enough for the short answers in this project.
    """
    text = (text or "").strip()
    if not text:
        return []

    nlp = _spacy_pipeline()
    if nlp is not None:
        sentences = [s.text.strip() for s in nlp(text).sents]
    else:
        sentences = []
        for block in re.split(r"\n+", text):
            sentences.extend(_SENTENCE_END.split(block.strip()))

    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_chars]
    return sentences or [text]


_SPACY = None


def _spacy_pipeline():
    global _SPACY
    if _SPACY is None:
        try:
            import spacy

            _SPACY = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except Exception:
            _SPACY = False
    return _SPACY or None


# NLI wrapper
class NLIModel:
    """
    Batched premise/hypothesis classifier for any MNLI checkpoint.

    Label order differs between checkpoints (deberta-v2-xlarge-mnli is
    [contradiction, neutral, entailment]; potsawee/deberta-v3-large-mnli
    is [entailment, contradiction]), so indices are read from
    `config.id2label` instead of being hard-coded. Checkpoints whose
    config only says label_0/label_1 fall back to KNOWN_LABEL_ORDER.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 64,
    ):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch = torch
        self.device = device or pick_device()
        self.max_length = max_length
        self.batch_size = batch_size

        # [transformers] The tokenizer you are loading from '/users/staff/pranith/LLM_Hallucinations/models/deberta-v2-xlarge-mnli' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fix_mistral_regex=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, dtype=dtype
        ).to(self.device)
        self.model.eval()

        id2label = {
            int(i): str(name).lower()
            for i, name in self.model.config.id2label.items()
        }
        if all(name.startswith("label_") for name in id2label.values()):
            id2label = _known_label_order(model_name_or_path, id2label)
        self.entail_idx = _find_label(id2label, "entail")
        self.contra_idx = _find_label(id2label, "contra")
        self.neutral_idx = _find_label(id2label, "neutral", required=False)

    def logits(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        """Raw logits, shape (n_pairs, n_labels), in batches."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have equal length")

        outputs = []
        for start in range(0, len(premises), self.batch_size):
            batch = self.tokenizer(
                premises[start : start + self.batch_size],
                hypotheses[start : start + self.batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with self.torch.inference_mode():
                logits = self.model(**batch).logits.float()
            outputs.append(logits.cpu().numpy())

        if not outputs:
            return np.zeros((0, self.model.config.num_labels), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def probs(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        """
        Softmax probabilities as a (n_pairs, 3) array in a FIXED order:
        [contradiction, neutral, entailment]. Neutral is 0 for 2-class
        checkpoints.
        """
        logits = self.logits(premises, hypotheses)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)

        out = np.zeros((len(p), 3), dtype=np.float32)
        out[:, 0] = p[:, self.contra_idx]
        if self.neutral_idx is not None:
            out[:, 1] = p[:, self.neutral_idx]
        out[:, 2] = p[:, self.entail_idx]
        return out


# Checkpoints whose config.json has only generic label names.
# potsawee/deberta-v3-large-mnli: SelfCheckGPT's reference code reads
# probs[:, 1] as P(contradiction), i.e. 0 = entailment, 1 = contradiction.
KNOWN_LABEL_ORDER = {
    "deberta-v3-large-mnli": ["entailment", "contradiction"],
}


def _known_label_order(model_name_or_path: str, id2label: dict[int, str]) -> dict[int, str]:
    name = str(model_name_or_path).rstrip("/").split("/")[-1]
    order = KNOWN_LABEL_ORDER.get(name)
    if order is None or len(order) != len(id2label):
        raise ValueError(
            f"NLI model {model_name_or_path} has unnamed labels {id2label}; "
            "add its label order to KNOWN_LABEL_ORDER in src/signals/common.py"
        )
    return dict(enumerate(order))


def _find_label(id2label: dict[int, str], key: str, required: bool = True):
    for index, name in id2label.items():
        if key in name:
            return index
    if required:
        raise ValueError(f"NLI model has no '{key}' label: {id2label}")
    return None


# Sharded, resumable output (same pattern as A3)
class ShardWriter:
    """
    Write results in numbered parquet shards so an interrupted SLURM job
    can resume. `done_uids()` tells the caller what to skip.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def done_uids(self) -> set[str]:
        done = set()
        for path in sorted(self.directory.glob("part_*.parquet")):
            done.update(pd.read_parquet(path, columns=["uid"])["uid"].astype(str))
        return done

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        existing = sorted(self.directory.glob("part_*.parquet"))
        number = len(existing) + 1
        path = self.directory / f"part_{number:06d}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def read_all(self) -> pd.DataFrame:
        paths = sorted(self.directory.glob("part_*.parquet"))
        if not paths:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


# Hallucination labels
def load_labels(path: Path = LABELS_PATH) -> pd.DataFrame | None:
    """
    Labels for the MAIN answer: columns uid, hallucinated (0/1).

    Returns None (and the runners skip evaluation) if no label file
    exists yet. See src/signals/labels.py.
    """
    if not Path(path).exists():
        print(
            f"[WARNING] No label file at {path}; scores are saved but "
            "AUROC/AUPRC are not computed."
        )
        return None
    labels = pd.read_parquet(path)
    labels["uid"] = labels["uid"].astype(str)
    return labels[["uid", "hallucinated"]]
