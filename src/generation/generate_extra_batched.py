"""
Batched, resumable version of generate_extra.py (responses 2..16 per prompt).

Why: generate_extra.py makes one model.generate() call per answer and
rewrites the whole output parquet every 100 answers, so 369,780 answers
take most of a day on a GPU that is almost idle. This script generates
BATCH_SIZE answers per call and writes small checkpoint shards instead.

Same sampling setup as A2 / generate_extra.py:
    same chat template (one user message + generation prompt),
    temperature 0.7, top_p 0.9, max_new_tokens 128, do_sample=True,
    and the model's own generation_config for everything else.

Difference (recorded in the output): answers are sampled in batches with
one seed per batch (column `batched` = True), instead of one seed per
(uid, response_index). They are draws from the same distribution but
not bit-identical to what generate_extra.py would have produced.

Resume: rows already in the final parquet (e.g. from an interrupted
generate_extra.py run) and in the shard folder are skipped.

Run from the repository root:
    python src/generation/generate_extra_batched.py
    python src/generation/generate_extra_batched.py --batch-size 512
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"

INPUT_FILE = PROJECT_ROOT / "data" / "calibration.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "generation_qwen_calibration_extra15.parquet"
SHARD_DIR = PROJECT_ROOT / "data" / "extra15_shards"

BASE_SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 128

FIRST_RESPONSE_INDEX = 2
LAST_RESPONSE_INDEX = 16

BATCHES_PER_SHARD = 20

COLUMNS = [
    "uid", "response_index", "prompt", "generated_answer", "model",
    "seed", "temperature", "top_p", "max_new_tokens", "batched",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default=str(MODEL_PATH))
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_seed(first_uid: str, first_index: int) -> int:
    text = f"{BASE_SEED}:batch:{first_uid}:{first_index}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


def load_existing() -> pd.DataFrame:
    frames = []
    if OUTPUT_FILE.exists():
        try:
            old = pd.read_parquet(OUTPUT_FILE)
        except Exception as exc:
            raise RuntimeError(
                f"{OUTPUT_FILE} cannot be read ({exc}). It was probably cut off "
                "mid-write when the old job was cancelled. Move it aside and rerun."
            ) from exc
        if "batched" not in old.columns:
            old["batched"] = False
        frames.append(old[COLUMNS])
    for path in sorted(SHARD_DIR.glob("part_*.parquet")):
        frames.append(pd.read_parquet(path)[COLUMNS])
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    existing = pd.concat(frames, ignore_index=True)
    return existing.drop_duplicates(["uid", "response_index"], keep="first")


def write_shard(rows: list[dict]):
    if not rows:
        return
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    number = len(list(SHARD_DIR.glob("part_*.parquet"))) + 1
    path = SHARD_DIR / f"part_{number:06d}.parquet"
    tmp = SHARD_DIR / f"tmp_{number:06d}.parquet"
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def merge_final(n_prompts: int):
    """Combine the old file and all shards into OUTPUT_FILE (atomic)."""
    merged = load_existing().sort_values(["uid", "response_index"], kind="stable")
    tmp = OUTPUT_FILE.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, OUTPUT_FILE)

    expected = n_prompts * (LAST_RESPONSE_INDEX - FIRST_RESPONSE_INDEX + 1)
    print(f"[INFO] Final file: {len(merged):,} / {expected:,} rows -> {OUTPUT_FILE}")
    if len(merged) == expected:
        for path in SHARD_DIR.glob("*.parquet"):
            path.unlink()
        SHARD_DIR.rmdir()
        print("[SUCCESS] All responses present; shards merged and removed.")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[INFO] Device: {device}  batch size: {args.batch_size}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    prompts = pd.read_parquet(INPUT_FILE)
    prompts["uid"] = prompts["uid"].astype(str)
    if prompts["uid"].duplicated().any():
        raise ValueError("Duplicate UIDs found in calibration.parquet")

    existing = load_existing()
    completed = set(zip(existing["uid"].astype(str), existing["response_index"].astype(int)))
    print(f"[INFO] Calibration prompts: {len(prompts):,}")
    print(f"[INFO] Already generated:   {len(completed):,}")

    tasks = [
        (row.uid, idx, row.prompt)
        for row in prompts.sort_values("uid", kind="stable").itertuples(index=False)
        for idx in range(FIRST_RESPONSE_INDEX, LAST_RESPONSE_INDEX + 1)
        if (row.uid, idx) not in completed
    ]
    print(f"[INFO] Remaining:           {len(tasks):,}")
    if not tasks:
        merge_final(len(prompts))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    # Render each prompt once with the exact A2 template.
    rendered = {
        uid: tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for uid, prompt in prompts[["uid", "prompt"]].itertuples(index=False)
    }

    # Similar lengths in one batch = less padding. Shuffle-free and
    # deterministic: sort by rendered length, then uid, then index.
    tasks.sort(key=lambda t: (len(rendered[t[0]]), t[0], t[1]))

    started = time.time()
    pending, done = [], 0
    n_batches = (len(tasks) + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        batch = tasks[b * args.batch_size : (b + 1) * args.batch_size]
        seed = batch_seed(batch[0][0], batch[0][1])
        set_seed(seed)

        enc = tokenizer(
            [rendered[uid] for uid, _, _ in batch],
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

        new_tokens = out[:, enc["input_ids"].shape[1] :]
        answers = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for (uid, idx, prompt), answer in zip(batch, answers):
            pending.append({
                "uid": uid,
                "response_index": idx,
                "prompt": prompt,
                "generated_answer": answer,
                "model": str(args.model),
                "seed": seed,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
                "batched": True,
            })
        done += len(batch)

        if (b + 1) % BATCHES_PER_SHARD == 0 or b == n_batches - 1:
            write_shard(pending)
            pending = []
            elapsed = time.time() - started
            rate = done / elapsed
            eta = (len(tasks) - done) / rate if rate else float("nan")
            print(
                f"[CHECKPOINT] {done:,}/{len(tasks):,}  "
                f"{rate:.1f} answers/s  elapsed {elapsed / 60:.1f} min  "
                f"ETA {eta / 60:.1f} min"
            )

    merge_final(len(prompts))


if __name__ == "__main__":
    main()
