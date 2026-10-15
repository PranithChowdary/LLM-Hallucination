from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 128

# A2 is response 1. Generating responses 2 through 16.
FIRST_RESPONSE_INDEX = 2
LAST_RESPONSE_INDEX = 16

SAVE_EVERY = 100

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "calibration.parquet"
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "generation_qwen_calibration_extra15.parquet"
)


def set_seed(seed):
    """Set random seeds for reproducible sampling."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for(uid, response_index):
    """Derive a stable, distinct seed for each prompt and response."""
    text = f"{BASE_SEED}:{uid}:{response_index}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data type: {dtype}")

    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] No GPU detected. Full generation on CPU may be very slow.")

    # Load the frozen calibration prompts.
    prompts = pd.read_parquet(INPUT_FILE)

    if prompts["uid"].duplicated().any():
        raise ValueError("Duplicate UIDs found in calibration.parquet")

    print(f"[INFO] Calibration prompts: {len(prompts)}")

    expected_columns = {"uid", "prompt"}
    if not expected_columns.issubset(prompts.columns):
        raise ValueError(f"Input must contain: {expected_columns}")

    # Load previous checkpoint, if available.
    if OUTPUT_FILE.exists():
        previous = pd.read_parquet(OUTPUT_FILE)

        required = {
            "uid", "response_index", "generated_answer",
            "model", "seed", "temperature", "top_p",
            "max_new_tokens"
        }
        if not required.issubset(previous.columns):
            raise ValueError("Existing output has an unexpected schema.")

        if previous.duplicated(["uid", "response_index"]).any():
            raise ValueError("Duplicate prompt/response pairs in checkpoint.")

        results = previous.to_dict("records")
        completed = set(zip(previous["uid"], previous["response_index"]))
        print(f"[INFO] Existing extra responses: {len(results)}")
    else:
        results = []
        completed = set()

    # Fail early if the checkpoint belongs to different prompts.
    valid_uids = set(prompts["uid"])
    if any(uid not in valid_uids for uid, _ in completed):
        raise ValueError("Checkpoint contains UIDs absent from calibration data.")

    # Load the model only if work remains.
    tasks = [
        (row.uid, response_index, row.prompt)
        for row in prompts.itertuples(index=False)
        for response_index in range(
            FIRST_RESPONSE_INDEX, LAST_RESPONSE_INDEX + 1
        )
        if (row.uid, response_index) not in completed
    ]

    print(f"[INFO] Additional responses remaining: {len(tasks)}")

    if not tasks:
        print("[SUCCESS] All 15 additional responses are already present.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=dtype
    ).to(device)
    model.eval()

    for number, (uid, response_index, prompt) in enumerate(tasks, start=1):
        current_seed = seed_for(uid, response_index)
        set_seed(current_seed)

        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )["input_ids"].to(device)

        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True
            )

        new_tokens = output[0][inputs.shape[-1]:]
        answer = tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )

        results.append({
            "uid": uid,
            "response_index": response_index,
            "prompt": prompt,
            "generated_answer": answer,
            "model": MODEL_NAME,
            "seed": current_seed,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS
        })

        if number % SAVE_EVERY == 0:
            pd.DataFrame(results).to_parquet(OUTPUT_FILE, index=False)
            print(
                f"[CHECKPOINT] {number}/{len(tasks)} new responses processed; "
                f"{len(results)} saved in total."
            )

    pd.DataFrame(results).to_parquet(OUTPUT_FILE, index=False)

    print("[SUCCESS] Additional generation completed.")
    print(f"[INFO] Extra responses saved: {len(results)}")
    print(f"[INFO] Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
