"""
Sanity-check the NLI models before trusting labels / clustering.

transformers 5.x warns that the deberta-v2-xlarge-mnli tokenizer has an
"incorrect regex pattern". This script shows whether tokenization and
the NLI predictions are actually sane. CPU is enough (a few pairs).

Run on the server's login node from the repository root:
    python scripts/check_nli.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.common import MODELS_DIR, NLIModel  # noqa: E402


MODELS = ["deberta-v2-xlarge-mnli", "deberta-v3-large-mnli"]

# (premise, hypothesis, expected label)
PAIRS = [
    ("A man is playing a guitar on stage.", "A man is performing music.", "entailment"),
    ("A man is playing a guitar on stage.", "Nobody is playing an instrument.", "contradiction"),
    ("Paris is the capital of France.", "Berlin is the capital of France.", "contradiction"),
    ("Summarize the following document. I'm sorry, but there is no document provided for me to summarize.",
     "Summarize the following document. I cannot summarize because no document was given. Please provide it.",
     "entailment"),
    ("Who wrote Hamlet? Hamlet was written by William Shakespeare.",
     "Who wrote Hamlet? William Shakespeare is the author of Hamlet.",
     "entailment"),
]

TEXTS = [
    "Arthur's Magazine (1844-1846) was an American literary periodical.",
    "I'm sorry, but I can't summarize the document—there's none provided!",
]


def main():
    for name in MODELS:
        path = MODELS_DIR / name
        print("=" * 72)
        print(name)
        nli = NLIModel(str(path), device="cpu", batch_size=8)
        print("label indices: entail", nli.entail_idx, " contra", nli.contra_idx, " neutral", nli.neutral_idx)

        for text in TEXTS:
            ids = nli.tokenizer(text, add_special_tokens=False)["input_ids"]
            back = nli.tokenizer.decode(ids)
            tokens = nli.tokenizer.convert_ids_to_tokens(ids)[:14]
            print(f"\n  text:      {text}")
            print(f"  tokens:    {tokens}")
            print(f"  roundtrip: {'OK' if back.strip() == text.strip() else 'DIFFERS -> ' + back}")

        probs = nli.probs([p for p, _, _ in PAIRS], [h for _, h, _ in PAIRS])
        names = ["contradiction", "neutral", "entailment"]
        correct = 0
        print()
        for (p, h, expected), row in zip(PAIRS, probs):
            got = names[int(row.argmax())]
            correct += got == expected
            print(f"  [{'OK ' if got == expected else 'BAD'}] expected {expected:13s} got {got:13s} "
                  f"(c={row[0]:.2f} n={row[1]:.2f} e={row[2]:.2f})  {h[:60]}")
        print(f"\n  {correct}/{len(PAIRS)} as expected")


if __name__ == "__main__":
    main()
