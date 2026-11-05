import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# config...
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.common import model_prompt  # noqa: E402

MODEL_NAME = str(PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct")

SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 128

CALIBRATION_FILE = PROJECT_ROOT / "data" / "calibration.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "generation_qwen_calibration.parquet"

SAVE_EVERY = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# set seed...
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# main function...
def main():
    set_seed(SEED)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Data type: {DTYPE}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load calibration data
    print("[INFO] Loading calibration data...")
    df = pd.read_parquet(CALIBRATION_FILE)
    df["prompt"] = [
        model_prompt(p, c, s)
        for p, c, s in zip(df["prompt"], df["context"], df["subset"])
    ]
    print(f"[INFO] Total calibration prompts: {len(df)}")

    # Resume support
    if OUTPUT_FILE.exists():
        existing_df = pd.read_parquet(OUTPUT_FILE)
        completed_uids = set(existing_df["uid"])
        print(f"[INFO] Existing output found: {len(existing_df)} generations")

        df = df[~df["uid"].isin(completed_uids)].copy()
        print(f"[INFO] Remaining prompts to generate: {len(df)}")
        results = existing_df.to_dict("records")
    else:
        print("[INFO] No previous output found.")
        results = []

    if len(df) == 0:
        print("[SUCCESS] All prompts have already been generated.")
        return

    # Load tokenizer & model
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,  # 'dtype' avoids the deprecation warning
        local_files_only=True
    )
    model = model.to(DEVICE)
    model.eval()
    print("[INFO] Model loaded successfully.")

    # Generation Loop
    rows = df.to_dict("records")
    pbar = tqdm(rows, desc="Generating", total=len(rows), ncols=100)

    for count, row in enumerate(pbar, start=1):
        prompt = row["prompt"]
        messages = [{"role": "user", "content": prompt}]

        # Returns a BatchEncoding dict with input_ids and attention_mask
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(DEVICE)

        input_token_len = model_inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = model.generate(
                **model_inputs,  # Unpack dict directly into generate
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Slice off input tokens to keep only generated response
        generated_tokens = output[0][input_token_len:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        results.append({
            "uid": row["uid"],
            "dataset": row["dataset"],
            "subset": row["subset"],
            "prompt": prompt,
            "generated_answer": answer,
            "model": str(MODEL_NAME),
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS
        })

        # Atomic checkpoint save
        if count % SAVE_EVERY == 0:
            checkpoint_df = pd.DataFrame(results)
            temp_file = OUTPUT_FILE.with_suffix(".parquet.tmp")
            checkpoint_df.to_parquet(temp_file, index=False, engine="pyarrow")
            os.replace(temp_file, OUTPUT_FILE)
            pbar.set_postfix({"saved": len(results)})

    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")

    print("\n[SUCCESS] Full generation completed.")
    print(f"[INFO] Total generations: {len(result_df)}")
    print(f"[INFO] Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
