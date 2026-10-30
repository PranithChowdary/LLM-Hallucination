"""
PROVISIONAL hallucination labels for the A2 main answers (needed to
compute AUROC in B3/B4/B5). Read src/signals/labels.py before trusting
these labels; they are a stopgap until the team picks a proper judge.

Run from the repository root:
    python scripts/label_generations.py
    python scripts/label_generations.py --limit 200

Output: data/processed/generation_labels.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.common import (  # noqa: E402
    LABELS_PATH,
    MAIN_ANSWERS_PATH,
    MODELS_DIR,
    NLIModel,
    load_prompts,
    take_limit,
)
from src.signals.labels import label_answers  # noqa: E402


DEFAULT_NLI = MODELS_DIR / "deberta-v2-xlarge-mnli"
DEFAULT_NLI_HUB = "microsoft/deberta-v2-xlarge-mnli"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nli-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--keep-refusals",
        action="store_true",
        help="Label refusals instead of excluding them (NaN).",
    )
    args = parser.parse_args()

    nli_path = args.nli_model or (
        str(DEFAULT_NLI) if DEFAULT_NLI.exists() else DEFAULT_NLI_HUB
    )

    answers = pd.read_parquet(MAIN_ANSWERS_PATH)
    answers["uid"] = answers["uid"].astype(str)
    answers = answers.rename(columns={"generated_answer": "main_answer"})
    answers = answers.sort_values("uid", kind="stable")
    answers = take_limit(answers, args.limit)

    labels = label_answers(
        answers,
        load_prompts(),
        NLIModel(nli_path, args.device),
        exclude_refusals=not args.keep_refusals,
    )

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(LABELS_PATH, index=False)

    summary = labels.groupby("subset").agg(
        n=("uid", "size"),
        refusal_rate=("refusal", "mean"),
        hallucination_rate=("hallucinated", "mean"),
        labeled=("hallucinated", "count"),
    )
    print(summary.to_string())
    print(f"[SUCCESS] Labels -> {LABELS_PATH}")


if __name__ == "__main__":
    main()
