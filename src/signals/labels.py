"""
PROVISIONAL hallucination labels for the A2 main answers.

Why this exists:
    B3/B4/B5 produce scores. To report AUROC we need a 0/1 label saying
    whether each MAIN answer (A2, response_index 1) is a hallucination.
    A1 labels belong to the reference answers, not to our generations,
    and no step in Block A labels the generations yet.

What the papers did:
    - Semantic Entropy (Kuhn et al. 2023): ROUGE-L(answer, reference) > 0.3.
    - Semantic Entropy (Farquhar et al. 2024) and SEPs (Kossen et al. 2024):
      an LLM judge (GPT-3.5/4) compares the answer with the reference.
    - SelfCheckGPT: human sentence-level annotation (WikiBio).

Our answers are long (median ~65 words), so ROUGE-L against a 2-word
reference is meaningless. This module uses NLI + string containment
as a stopgap. It is good enough to debug B3-B5 end-to-end; it is NOT
good enough for paper numbers. The team should agree on a judge
(ideally a larger LLM judge, spot-checked by hand) and replace this.

Output: data/processed/generation_labels.parquet with
    uid, subset, refusal, correct_score, incorrect_score,
    unsupported_fraction, hallucinated (1 / 0 / NaN = excluded),
    label_method
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.data.loaders import normalize_text
from src.signals.common import NLIModel, split_sentences


LABEL_METHOD = "provisional-nli-v1"

# Refusals are not hallucinations and not correct answers. They are
# excluded (label NaN) unless the caller asks otherwise.
_REFUSAL = re.compile(
    r"^\s*(i'?m sorry|i am sorry|sorry,|i cannot|i can'?t|i'?m unable|"
    r"i am unable|as an ai[^.]{0,60}(cannot|can'?t|don'?t|do not))",
    flags=re.IGNORECASE,
)

ENTAIL_THRESHOLD = 0.5


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer or ""))


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    return [str(v) for v in list(value) if str(v).strip()]


def label_reference_task(
    nli: NLIModel, question: str, answer: str, correct: list[str], incorrect: list[str]
) -> dict:
    """
    QA / TruthfulQA: does the answer agree more with a correct reference
    or with an incorrect one?

        correct_score   = max over correct refs of P(answer entails "Q R")
        incorrect_score = max over incorrect refs, same thing
    A normalized reference string contained in the answer counts as 1.0.
    """
    def score(references: list[str]) -> float:
        if not references:
            return 0.0
        norm_answer = f" {normalize_text(answer)} "
        contained = [
            1.0 if f" {normalize_text(r)} " in norm_answer else 0.0
            for r in references
        ]
        probs = nli.probs(
            [answer] * len(references),
            [f"{question} {r}" for r in references],
        )[:, 2]
        return float(max(max(contained), probs.max()))

    correct_score = score(correct)
    incorrect_score = score(incorrect)

    hallucinated = int(
        correct_score < ENTAIL_THRESHOLD or incorrect_score > correct_score
    )
    return {
        "correct_score": correct_score,
        "incorrect_score": incorrect_score,
        "unsupported_fraction": np.nan,
        "hallucinated": hallucinated,
    }


def label_grounded_task(nli: NLIModel, context: str, answer: str) -> dict:
    """
    Dialogue / summarization: is each answer sentence entailed by the
    provided knowledge/document? Hallucinated if more than half of the
    sentences are not entailed.

    Caveat: A2 did NOT put `context` into the prompt, so the model never
    saw the knowledge it is being judged against.
    """
    sentences = split_sentences(answer)
    probs = nli.probs([context] * len(sentences), sentences)[:, 2]
    unsupported = float((probs < ENTAIL_THRESHOLD).mean())
    return {
        "correct_score": np.nan,
        "incorrect_score": np.nan,
        "unsupported_fraction": unsupported,
        "hallucinated": int(unsupported > 0.5),
    }


def label_answers(
    responses: pd.DataFrame,
    prompts: pd.DataFrame,
    nli: NLIModel,
    exclude_refusals: bool = True,
) -> pd.DataFrame:
    rows = []
    for row in responses.itertuples(index=False):
        ref = prompts.loc[row.uid]
        refusal = is_refusal(row.main_answer)

        if row.subset in ("qa", "truthfulqa"):
            result = label_reference_task(
                nli,
                row.prompt,
                row.main_answer,
                _as_list(ref["reference_correct"]),
                _as_list(ref["reference_incorrect"]),
            )
        else:
            result = label_grounded_task(nli, str(ref["context"]), row.main_answer)

        if refusal and exclude_refusals:
            result["hallucinated"] = np.nan

        rows.append({
            "uid": row.uid,
            "subset": row.subset,
            "refusal": refusal,
            **result,
            "label_method": LABEL_METHOD,
        })

    return pd.DataFrame(rows)
