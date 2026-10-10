"""
A3: Full-dataset token feature extraction.

Run from the repository root:
    python src/extract_features/extract_token_features.py

This script does NOT generate new answers.
It evaluates the saved A2 answer text using teacher-forced forward passes.

Important:
- A2 saved decoded answer text, not original generation-time token IDs/logits.
- Token IDs are reconstructed from that saved text.
- Scores are recomputed; they are not original A2 sampling-time scores.
- A3 extracts numerical features; it does not label hallucinations.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------------------------------------------------
# 1. Paths and settings
# -------------------------------------------------------------------

MODEL_DIR = Path("models/Qwen2.5-0.5B-Instruct")
GENERATION_PATH = Path("data/generation_qwen_calibration.parquet")
CALIBRATION_PATH = Path("data/calibration.parquet")

OUTPUT_DIR = Path("data/a3")
FEATURE_DIR = OUTPUT_DIR / "token_features"
HIDDEN_DIR = OUTPUT_DIR / "hidden_states"
MANIFEST_PATH = OUTPUT_DIR / "a3_manifest.json"

# Save a feature shard after every 100 completed examples.
# If interrupted, the next run resumes from completed shards.
EXAMPLES_PER_CHECKPOINT = 100

# Hidden-state vectors are stored in float16 to reduce disk usage.
HIDDEN_DTYPE = torch.float16

REQUIRED_GENERATION_COLUMNS = {
    "uid", "prompt", "generated_answer"
}
REQUIRED_CALIBRATION_COLUMNS = {"uid", "prompt"}


# -------------------------------------------------------------------
# 2. Small helper functions
# -------------------------------------------------------------------

def uid_filename(uid: str) -> str:
    """Return a stable, filesystem-safe filename for a UID."""
    return hashlib.sha256(uid.encode("utf-8")).hexdigest() + ".pt"


def save_manifest(status: str, **extra) -> None:
    """Write progress information safely, including after interruption."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task": "A3 token-level feature extraction",
        "status": status,
        "model_path": str(MODEL_DIR),
        "generation_input": str(GENERATION_PATH),
        "calibration_input": str(CALIBRATION_PATH),
        "feature_output": str(FEATURE_DIR),
        "hidden_state_output": str(HIDDEN_DIR),
        "feature_columns": [
            "uid",
            "token_position",
            "token_id",
            "token_text",
            "logit",
            "probability",
            "log_probability",
            "entropy",
        ],
        "hidden_state_storage_dtype": "float16",
        "method": "teacher-forced forward pass",
        "generation_performed": False,
        "token_ids_reconstructed_from_saved_text": True,
        "scores_are_original_a2_sampling_scores": False,
        "updated_unix_time": time.time(),
    }
    manifest.update(extra)

    temporary_path = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, MANIFEST_PATH)


# -------------------------------------------------------------------
# 3. Validate the input datasets before processing
# -------------------------------------------------------------------

def load_and_validate_inputs():
    """Check required columns, unique UIDs, exact coverage, and prompts."""

    for path in (MODEL_DIR, GENERATION_PATH, CALIBRATION_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    generations = pd.read_parquet(GENERATION_PATH)
    calibration = pd.read_parquet(CALIBRATION_PATH)

    missing_generation = REQUIRED_GENERATION_COLUMNS - set(generations.columns)
    missing_calibration = REQUIRED_CALIBRATION_COLUMNS - set(calibration.columns)

    if missing_generation:
        raise ValueError(
            f"Generation file is missing columns: {sorted(missing_generation)}"
        )
    if missing_calibration:
        raise ValueError(
            f"Calibration file is missing columns: {sorted(missing_calibration)}"
        )

    # Convert UIDs to strings to make comparisons consistent.
    generations["uid"] = generations["uid"].astype("string")
    calibration["uid"] = calibration["uid"].astype("string")

    if generations["uid"].isna().any() or calibration["uid"].isna().any():
        raise ValueError("Null UIDs were found in an input dataset.")

    if generations["uid"].duplicated().any():
        raise ValueError("Duplicate UIDs found in the generation dataset.")
    if calibration["uid"].duplicated().any():
        raise ValueError("Duplicate UIDs found in the calibration dataset.")

    generation_uids = set(generations["uid"].tolist())
    calibration_uids = set(calibration["uid"].tolist())

    if generation_uids != calibration_uids:
        missing = sorted(calibration_uids - generation_uids)[:10]
        extra = sorted(generation_uids - calibration_uids)[:10]
        raise ValueError(
            "Generation/calibration UID mismatch. "
            f"Missing from generation (first 10): {missing}; "
            f"extra in generation (first 10): {extra}"
        )

    # Make sure each saved generation row still corresponds to its original prompt.
    calibration_prompts = calibration.set_index("uid")["prompt"].to_dict()

    for row in generations[["uid", "prompt"]].itertuples(index=False):
        original_prompt = calibration_prompts[row.uid]

        if pd.isna(row.prompt) or pd.isna(original_prompt):
            raise ValueError(f"Null prompt found for UID {row.uid}")

        if str(row.prompt) != str(original_prompt):
            raise ValueError(f"Prompt mismatch for UID {row.uid}")

    if generations["generated_answer"].isna().any():
        raise ValueError("Null generated_answer values were found.")

    generations["generated_answer"] = generations["generated_answer"].astype(str)

    empty_answers = generations["generated_answer"].str.strip().eq("")
    if empty_answers.any():
        bad_uids = generations.loc[empty_answers, "uid"].head(10).tolist()
        raise ValueError(f"Empty answers found. Example UIDs: {bad_uids}")

    # Stable order helps reproducibility between resumed runs.
    generations = generations.sort_values("uid", kind="stable").reset_index(drop=True)

    print(f"Generation examples: {len(generations):,}")
    print(f"Calibration examples: {len(calibration):,}")
    print("UID coverage and prompt alignment: PASSED")

    return generations


# -------------------------------------------------------------------
# 4. Prepare prompt and reconstructed answer token IDs
# -------------------------------------------------------------------

def encode_prompt(tokenizer, prompt: str, device: torch.device):
    """
    Render the same basic user-message chat template used by A2:
    one user message followed by the assistant generation prefix.

    If A2 used any additional system messages or a different template,
    this function must be adjusted to match that exact A2 setup.
    """
    messages = [{"role": "user", "content": prompt}]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Extracting the input_ids based on the type...
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    elif isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded  # It is already a tensor

    return input_ids.to(device)


def encode_saved_answer(tokenizer, answer: str, device: torch.device):
    """
    Reconstruct answer token IDs from the saved decoded text.

    This cannot guarantee the exact original sampled token IDs, because
    A2 stored decoded text and removed special tokens during decoding.
    """
    encoded = tokenizer(
        answer,
        add_special_tokens=False,
        return_tensors="pt",
    )

    answer_ids = encoded["input_ids"].to(device)

    if answer_ids.shape[1] == 0:
        raise ValueError("Saved answer produced zero tokens after tokenization.")

    return answer_ids


# -------------------------------------------------------------------
# 5. Extract features for one example
# -------------------------------------------------------------------

def extract_one_example(model, tokenizer, uid, prompt, answer, device):
    """
    Return:
      1. One feature row per reconstructed answer token.
      2. A hidden-state dictionary saved to disk.

    Alignment:
      - Logits at position prompt_length + i - 1 predict answer token i.
      - Hidden state at position prompt_length + i represents answer token i.
    """

    prompt_ids = encode_prompt(tokenizer, prompt, device)
    answer_ids = encode_saved_answer(tokenizer, answer, device)

    # prompt_ids and answer_ids are BatchEncoding object .shape won't work so, modifying the encode_prompt()...
    prompt_length = prompt_ids.shape[1]
    answer_length = answer_ids.shape[1]

    # Sequence: rendered prompt + reconstructed answer tokens.
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    # No call to model.generate(): this is a forward pass over saved text.
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        # Select the positions whose next-token predictions correspond
        # to the saved answer tokens.
        logits = result.logits[
            0,
            prompt_length - 1 : prompt_length + answer_length - 1,
            :
        ].float()

        # Use the final transformer layer's hidden state at each answer token.
        hidden = result.hidden_states[-1][
            0,
            prompt_length : prompt_length + answer_length,
            :
        ]

        if logits.shape[0] != answer_length:
            raise RuntimeError(
                f"Logit alignment failure for UID {uid}: "
                f"{logits.shape[0]} rows, expected {answer_length}"
            )

        if hidden.shape[0] != answer_length:
            raise RuntimeError(
                f"Hidden-state alignment failure for UID {uid}: "
                f"{hidden.shape[0]} rows, expected {answer_length}"
            )

        # The full vocabulary distribution is needed for entropy, but we
        # save only compact scores for the actual answer token.
        log_probs = torch.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probs)

        actual_token_ids = answer_ids[0]

        actual_logits = logits.gather(
            1, actual_token_ids.unsqueeze(1)
        ).squeeze(1)

        actual_log_probs = log_probs.gather(
            1, actual_token_ids.unsqueeze(1)
        ).squeeze(1)

        actual_probs = probabilities.gather(
            1, actual_token_ids.unsqueeze(1)
        ).squeeze(1)

        entropy = -(probabilities * log_probs).sum(dim=-1)

        token_ids_cpu = actual_token_ids.cpu().tolist()
        token_texts = [
            tokenizer.decode(
                [int(token_id)],
                clean_up_tokenization_spaces=False,
            )
            for token_id in token_ids_cpu
        ]

        logits_cpu = actual_logits.cpu().tolist()
        probs_cpu = actual_probs.cpu().tolist()
        log_probs_cpu = actual_log_probs.cpu().tolist()
        entropy_cpu = entropy.cpu().tolist()

        feature_rows = []

        for position in range(answer_length):
            feature_rows.append({
                "uid": str(uid),
                "token_position": int(position),
                "token_id": int(token_ids_cpu[position]),
                "token_text": token_texts[position],
                "logit": float(logits_cpu[position]),
                "probability": float(probs_cpu[position]),
                "log_probability": float(log_probs_cpu[position]),
                "entropy": float(entropy_cpu[position]),
            })

        # Store final-layer hidden vectors in float16 to reduce disk usage.
        hidden_cpu = hidden.detach().to(
            device="cpu",
            dtype=HIDDEN_DTYPE,
        ).contiguous()

        hidden_record = {
            "uid": str(uid),
            "token_positions": torch.arange(answer_length, dtype=torch.int32),
            "hidden_states": hidden_cpu,
            "hidden_size": int(hidden_cpu.shape[1]),
            "storage_dtype": "float16",
            "token_ids_reconstructed_from_text": True,
        }

    # Use a hash-based filename so UIDs with slashes or special characters
    # cannot accidentally create directories or invalid filenames.
    hidden_path = HIDDEN_DIR / uid_filename(str(uid))
    torch.save(hidden_record, hidden_path)

    # Return a small summary for the run log.
    summary = {
        "uid": str(uid),
        "answer_token_count": int(answer_length),
        "prompt_token_count": int(prompt_length),
        "hidden_size": int(hidden_cpu.shape[1]),
        "hidden_state_file": str(hidden_path),
    }

    # Remove references to large tensors before processing the next example.
    del result, logits, log_probs, probabilities, hidden

    return feature_rows, summary


# -------------------------------------------------------------------
# 6. Resume support and checkpointing
# -------------------------------------------------------------------

def completed_uids_from_shards():
    """Read UIDs already committed to feature Parquet shards."""
    completed = set()

    for path in sorted(FEATURE_DIR.glob("part_*.parquet")):
        frame = pd.read_parquet(path, columns=["uid"])
        completed.update(frame["uid"].astype(str).unique().tolist())

    return completed


def next_shard_number():
    """Return the next unused Parquet shard number."""
    numbers = []

    for path in FEATURE_DIR.glob("part_*.parquet"):
        try:
            numbers.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue

    return max(numbers, default=0) + 1


def write_feature_shard(rows, shard_number):
    """Atomically write a Parquet shard so incomplete files aren't mistaken for checkpoints."""
    path = FEATURE_DIR / f"part_{shard_number:06d}.parquet"
    temporary_path = path.with_suffix(".parquet.tmp")

    pd.DataFrame(rows).to_parquet(temporary_path, index=False)
    os.replace(temporary_path, path)

    return path


# -------------------------------------------------------------------
# 7. Main full-dataset run
# -------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    HIDDEN_DIR.mkdir(parents=True, exist_ok=True)

    generations = load_and_validate_inputs()

    expected_uids = set(generations["uid"].astype(str))
    completed_uids = completed_uids_from_shards()

    unexpected_uids = completed_uids - expected_uids
    if unexpected_uids:
        raise ValueError(
            "Existing feature shards contain UIDs absent from current inputs. "
            f"Example UIDs: {sorted(unexpected_uids)[:10]}. "
            "Move old A3 outputs aside before starting a different run."
        )

    remaining = generations[
        ~generations["uid"].astype(str).isin(completed_uids)
    ].reset_index(drop=True)

    print(f"Already checkpointed: {len(completed_uids):,}")
    print(f"Remaining examples: {len(remaining):,}")

    # Prefer GPU. CPU fallback is allowed for testing, but may be slow.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_dtype = torch.float16
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
    else:
        device = torch.device("cpu")
        model_dtype = torch.float32
        device_name = "CPU"
        print("WARNING: CUDA unavailable. Full-dataset CPU processing may be slow.")

    print(f"Loading tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    print(f"Loading model from {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        dtype=model_dtype, # torch_dtype is deprecated; used dtype instead...
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    save_manifest(
        "running",
        expected_examples=int(len(generations)),
        checkpointed_examples=int(len(completed_uids)),
        device=str(device),
        device_name=device_name,
        model_dtype=str(model_dtype),
        examples_per_checkpoint=EXAMPLES_PER_CHECKPOINT,
    )

    shard_number = next_shard_number()
    feature_buffer = []
    batch_uids = []
    started = time.time()

    try:
        for index, row in enumerate(remaining.itertuples(index=False), start=1):
            uid = str(row.uid)

            feature_rows, summary = extract_one_example(
                model=model,
                tokenizer=tokenizer,
                uid=uid,
                prompt=str(row.prompt),
                answer=str(row.generated_answer),
                device=device,
            )

            feature_buffer.extend(feature_rows)
            batch_uids.append(uid)

            if index % 10 == 0:
                print(
                    f"Progress: {len(completed_uids) + index:,}/"
                    f"{len(generations):,} examples; "
                    f"{len(feature_buffer):,} token rows in current checkpoint."
                )

            if len(batch_uids) >= EXAMPLES_PER_CHECKPOINT:
                shard_path = write_feature_shard(feature_buffer, shard_number)

                # Only UIDs in a successfully written shard count as complete.
                completed_uids.update(batch_uids)

                print(
                    f"Saved {shard_path}; "
                    f"checkpointed {len(completed_uids):,}/{len(generations):,}"
                )

                shard_number += 1
                feature_buffer = []
                batch_uids = []

                save_manifest(
                    "running",
                    expected_examples=int(len(generations)),
                    checkpointed_examples=int(len(completed_uids)),
                    device=str(device),
                    device_name=device_name,
                    model_dtype=str(model_dtype),
                    examples_per_checkpoint=EXAMPLES_PER_CHECKPOINT,
                    last_completed_shard=shard_path.name,
                    elapsed_seconds=round(time.time() - started, 2),
                )

        # Save the final partial checkpoint, if any examples remain.
        if feature_buffer:
            shard_path = write_feature_shard(feature_buffer, shard_number)
            completed_uids.update(batch_uids)
            print(f"Saved final partial shard: {shard_path}")

        save_manifest(
            "extraction_written",
            expected_examples=int(len(generations)),
            checkpointed_examples=int(len(completed_uids)),
            device=str(device),
            device_name=device_name,
            model_dtype=str(model_dtype),
            elapsed_seconds=round(time.time() - started, 2),
        )

        print("Extraction finished writing outputs.")
        print("Next step: python scripts/finalize_a3.py")

    except KeyboardInterrupt:
        # Uncommitted examples will be processed again on the next run.
        save_manifest(
            "interrupted",
            expected_examples=int(len(generations)),
            checkpointed_examples=int(len(completed_uids)),
            device=str(device),
            device_name=device_name,
            note="Rerun the same command to resume from completed feature shards.",
        )
        print("Run interrupted. Completed shards are preserved; rerun to resume.")
        raise

    except Exception as exc:
        save_manifest(
            "error",
            expected_examples=int(len(generations)),
            checkpointed_examples=int(len(completed_uids)),
            device=str(device),
            device_name=device_name,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
