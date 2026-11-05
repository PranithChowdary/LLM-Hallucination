"""
Regenerate every summarization answer with the document in the prompt.

Why: A1 stores the summarization prompt as the fixed instruction
"Summarize the following document." with the document in `context`, and
A2 sent only the instruction. The model never saw a document, so all
7,998 main answers and 119,970 extra samples are non-answers.

What it does:
    1. Samples response_index 1..16 for each summarization prompt, using
       model_prompt() and the same settings as generate_extra_batched.py.
       Finished answers go to data/summarization_regen/, so a rerun resumes.
    2. When every answer exists, copies the two generation files to
       data/pre_regen/ (once) and replaces their summarization rows.
       QA, dialogue and TruthfulQA rows are not touched.

Difference from A2 (visible in the `seed` column): summarization main
answers are sampled in batches with one seed per batch, like the extras.

Run from the repository root:
    python src/generation/regenerate_summarization.py
    python src/generation/regenerate_summarization.py --limit 48   # test, no merge
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.common import (  # noqa: E402
    CALIBRATION_PATH,
    LLM_DIR,
    MAIN_ANSWERS_PATH,
    SAMPLES_PATH,
    model_prompt,
    question_text,
)
from src.signals.labels import is_refusal  # noqa: E402


SHARD_DIR = PROJECT_ROOT / "data" / "summarization_regen"
BACKUP_DIR = PROJECT_ROOT / "data" / "pre_regen"

FIRST_RESPONSE_INDEX = 1  # the main answer
LAST_RESPONSE_INDEX = 16
BATCHES_PER_SHARD = 10

COLUMNS = [
    "uid", "response_index", "dataset", "subset", "prompt", "generated_answer",
    "model", "seed", "temperature", "top_p", "max_new_tokens", "batched",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default=str(LLM_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Prompts to run; skips the merge.")
    return parser.parse_args()


def load_prompts() -> pd.DataFrame:
    prompts = pd.read_parquet(CALIBRATION_PATH)
    prompts = prompts[prompts["subset"] == "summarization"].copy()
    prompts["uid"] = prompts["uid"].astype(str)
    prompts["prompt"] = [
        model_prompt(p, c, s)
        for p, c, s in zip(prompts["prompt"], prompts["context"], prompts["subset"])
    ]
    return prompts.sort_values("uid", kind="stable")


def load_regenerated(shard_dir: Path = SHARD_DIR) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(shard_dir.glob("part_*.parquet"))]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["uid", "response_index"], keep="first"
    )


def write_shard(rows: list[dict], shard_dir: Path = SHARD_DIR):
    if not rows:
        return
    shard_dir.mkdir(parents=True, exist_ok=True)
    number = len(list(shard_dir.glob("part_*.parquet"))) + 1
    tmp = shard_dir / f"tmp_{number:06d}.parquet"
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(tmp, index=False)
    tmp.replace(shard_dir / f"part_{number:06d}.parquet")


def generate(prompts: pd.DataFrame, done: set, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.generation.generate_extra_batched import (
        MAX_NEW_TOKENS,
        TEMPERATURE,
        TOP_P,
        batch_seed,
        set_seed,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[INFO] Device: {device}  batch size: {args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    rows = {row.uid: row for row in prompts.itertuples(index=False)}
    rendered = {
        uid: tokenizer.apply_chat_template(
            [{"role": "user", "content": row.prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for uid, row in rows.items()
    }

    tasks = [
        (uid, index)
        for uid in rows
        for index in range(FIRST_RESPONSE_INDEX, LAST_RESPONSE_INDEX + 1)
        if (uid, index) not in done
    ]
    # Longest first: if a batch does not fit on the GPU, it fails right away.
    tasks.sort(key=lambda t: (-len(rendered[t[0]]), t[0], t[1]))
    print(f"[INFO] Remaining answers: {len(tasks):,}")

    started = time.time()
    pending, finished = [], 0
    n_batches = (len(tasks) + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        batch = tasks[b * args.batch_size : (b + 1) * args.batch_size]
        seed = batch_seed(*batch[0])
        set_seed(seed)

        enc = tokenizer(
            [rendered[uid] for uid, _ in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        answers = tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for (uid, index), answer in zip(batch, answers):
            row = rows[uid]
            pending.append({
                "uid": uid,
                "response_index": index,
                "dataset": row.dataset,
                "subset": row.subset,
                "prompt": row.prompt,
                "generated_answer": answer,
                "model": str(args.model),
                "seed": seed,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
                "batched": True,
            })
        finished += len(batch)

        if (b + 1) % BATCHES_PER_SHARD == 0 or b == n_batches - 1:
            write_shard(pending)
            pending = []
            elapsed = time.time() - started
            rate = finished / elapsed
            print(
                f"[CHECKPOINT] {finished:,}/{len(tasks):,}  {rate:.1f} answers/s  "
                f"elapsed {elapsed / 60:.1f} min  "
                f"ETA {(len(tasks) - finished) / rate / 60:.1f} min"
            )


def replace_rows(old: pd.DataFrame, new: pd.DataFrame, sort_by: list[str]) -> pd.DataFrame:
    """`old` with its rows for the uids in `new` swapped for `new`, same columns."""
    kept = old[~old["uid"].astype(str).isin(set(new["uid"]))]
    merged = pd.concat([kept, new[list(old.columns)]], ignore_index=True)
    return merged.sort_values(sort_by, kind="stable").reset_index(drop=True)


def write_parquet(frame: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def merge(
    regenerated: pd.DataFrame,
    main_path: Path = MAIN_ANSWERS_PATH,
    extra_path: Path = SAMPLES_PATH,
    backup_dir: Path = BACKUP_DIR,
):
    """Back up both generation files once, then swap in the new rows."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in (main_path, extra_path):
        if not (backup_dir / path.name).exists():
            shutil.copy2(path, backup_dir / path.name)

    old_main = pd.read_parquet(main_path)
    old_extra = pd.read_parquet(extra_path)

    is_main = regenerated["response_index"] == FIRST_RESPONSE_INDEX
    main = replace_rows(old_main, regenerated[is_main], ["uid"])

    # Detectors read the full prompt from the main file. Keeping 119,970
    # copies of the documents here would only bloat this Git LFS file.
    samples = regenerated[~is_main].copy()
    samples["prompt"] = [question_text(p, s) for p, s in zip(samples["prompt"], samples["subset"])]
    extra = replace_rows(old_extra, samples, ["uid", "response_index"])

    if len(main) != len(old_main) or len(extra) != len(old_extra):
        raise RuntimeError(
            f"Row counts changed: main {len(old_main):,} -> {len(main):,}, "
            f"extra {len(old_extra):,} -> {len(extra):,}"
        )

    write_parquet(main, main_path)
    write_parquet(extra, extra_path)
    print(f"[SUCCESS] Replaced {int(is_main.sum()):,} main answers and "
          f"{int((~is_main).sum()):,} samples. Originals are in {backup_dir}")


def main():
    args = parse_args()
    prompts = load_prompts()
    if args.limit is not None:
        prompts = prompts.head(args.limit)

    regenerated = load_regenerated()
    done = set(zip(regenerated["uid"].astype(str), regenerated["response_index"].astype(int)))
    expected = len(prompts) * (LAST_RESPONSE_INDEX - FIRST_RESPONSE_INDEX + 1)
    print(f"[INFO] Summarization prompts: {len(prompts):,}  answers wanted: {expected:,}")

    if not {(u, i) for u in prompts["uid"]
            for i in range(FIRST_RESPONSE_INDEX, LAST_RESPONSE_INDEX + 1)} <= done:
        generate(prompts, done, args)
        regenerated = load_regenerated()

    regenerated = regenerated[regenerated["uid"].isin(set(prompts["uid"]))]
    main_answers = regenerated[regenerated["response_index"] == FIRST_RESPONSE_INDEX]
    refusals = main_answers["generated_answer"].map(is_refusal).mean()
    print(f"[INFO] Refusal rate among main answers: {refusals:.1%} (was 69.3% without the document)")
    for answer in main_answers["generated_answer"].head(3):
        print(f"[SAMPLE] {answer[:200]!r}")

    if args.limit is not None:
        print("[INFO] --limit given: generation files were not changed.")
        return
    if len(regenerated) != expected:
        raise RuntimeError(f"Expected {expected:,} answers, have {len(regenerated):,}.")
    merge(regenerated)


if __name__ == "__main__":
    main()
