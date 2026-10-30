
"""
Validate the A3 single-example pilot.

Checks:
1. Saved token features are structurally valid.
2. Saved probabilities, log probabilities, entropy, and target-token
   logits match a fresh teacher-forced forward pass at six positions.
3. Saved float16 hidden states agree with fresh float32 hidden states
   within a tolerance suitable for float16 storage.

This script does NOT generate text or change the saved pilot files.
Run it from the project root:
    python validate_a3_pilot.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------
# Paths and validation settings
# ------------------------------------------------------------

MODEL_PATH = Path("models/Qwen2.5-0.5B-Instruct")
GENERATION_PATH = Path("data/generation_qwen_calibration.parquet")
FEATURES_PATH = Path("data/a3_test/token_features.parquet")
HIDDEN_PATH = Path("data/a3_test/hidden_states.pt")
SUMMARY_PATH = Path("data/a3_test/example_summary.json")

CHECK_POSITIONS = [0, 1, 4, 32, 64, 127]

# Logits and probabilities are recomputed using the same model and
# reconstructed token sequence. Small floating-point differences can occur.
LOGIT_ATOL = 1e-4
LOGIT_RTOL = 1e-4
PROB_ATOL = 1e-5
PROB_RTOL = 1e-4
LOGPROB_ATOL = 1e-4
LOGPROB_RTOL = 1e-4
ENTROPY_ATOL = 1e-4
ENTROPY_RTOL = 1e-4

# Hidden states were saved as float16, so compare with a looser tolerance.
HIDDEN_ATOL = 1e-2
HIDDEN_RTOL = 1e-2


def require(condition, message):
    """Stop with a clear error if a required validation fails."""
    if not condition:
        raise AssertionError(message)


def main():
    print("Loading pilot artifacts...")

    for path in [
        MODEL_PATH,
        GENERATION_PATH,
        FEATURES_PATH,
        HIDDEN_PATH,
        SUMMARY_PATH,
    ]:
        require(path.exists(), f"Required file not found: {path}")

    generations = pd.read_parquet(GENERATION_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
        summary = json.load(file)

    # --------------------------------------------------------
    # 1. Basic artifact and feature checks
    # --------------------------------------------------------

    required_columns = {
        "uid",
        "token_position",
        "token_id",
        "token_text",
        "logit",
        "probability",
        "log_probability",
        "entropy",
    }
    require(
        required_columns.issubset(features.columns),
        f"Missing feature columns: {required_columns - set(features.columns)}",
    )

    require(len(features) > 0, "Token feature table is empty.")
    require(features["uid"].nunique() == 1, "Expected one pilot UID.")
    require(features["uid"].iloc[0] == summary["uid"],
            "Feature UID does not match summary UID.")

    features = features.sort_values("token_position").reset_index(drop=True)

    positions = features["token_position"].to_numpy(dtype=np.int64)
    expected_positions = np.arange(len(features))
    require(
        np.array_equal(positions, expected_positions),
        "Token positions are not consecutive from zero.",
    )

    for column in ["logit", "probability", "log_probability", "entropy"]:
        values = features[column].to_numpy(dtype=np.float64)
        require(np.isfinite(values).all(), f"Non-finite values in {column}.")

    require(
        ((features["probability"] >= 0) &
         (features["probability"] <= 1)).all(),
        "Probabilities must be between zero and one.",
    )
    require(
        (features["entropy"] >= -1e-6).all(),
        "Entropy contains a negative value.",
    )

    uid = features["uid"].iloc[0]
    matches = generations.loc[generations["uid"] == uid]
    require(len(matches) == 1, f"Expected one generation row for UID {uid}.")

    row = matches.iloc[0]
    require(
        isinstance(row["generated_answer"], str)
        and row["generated_answer"].strip(),
        "Saved generated answer is empty.",
    )

    print(f"Basic artifact checks passed for UID: {uid}")
    print(f"Saved answer token rows: {len(features)}")

    # --------------------------------------------------------
    # 2. Load model and reconstruct prompt + answer token IDs
    # --------------------------------------------------------

    print("Loading local model on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.to("cpu")

    # Reconstruct the same user-message prompt format used in the pilot.
    messages = [{"role": "user", "content": row["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        row["generated_answer"],
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    prompt_length = prompt_ids.shape[1]
    answer_token_count = answer_ids.shape[1]

    require(
        answer_token_count == len(features),
        "Reconstructed answer token count does not match feature row count. "
        f"Reconstructed={answer_token_count}, saved={len(features)}. "
        "Check that the prompt/template and tokenization match the pilot.",
    )

    # Concatenate prompt and reconstructed saved answer.
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    # --------------------------------------------------------
    # 3. Fresh teacher-forced reference forward pass
    # --------------------------------------------------------

    print("Running reference forward pass (no text generation)...")
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    # For answer token i, logits at the preceding sequence position
    # predict that token. The first answer token is predicted by the
    # last prompt-token position.
    answer_logits = output.logits[
        0,
        prompt_length - 1 : prompt_length + answer_token_count - 1,
        :,
    ].float()

    log_probs = torch.log_softmax(answer_logits, dim=-1)
    probs = torch.exp(log_probs)

    target_ids = answer_ids[0].long()
    reference_target_logits = answer_logits.gather(
        1, target_ids.unsqueeze(1)
    ).squeeze(1)
    reference_logprobs = log_probs.gather(
        1, target_ids.unsqueeze(1)
    ).squeeze(1)
    reference_probs = probs.gather(
        1, target_ids.unsqueeze(1)
    ).squeeze(1)
    reference_entropy = -(probs * log_probs).sum(dim=-1)

    # --------------------------------------------------------
    # 4. Compare six sampled positions
    # --------------------------------------------------------

    require(
        max(CHECK_POSITIONS) < answer_token_count,
        f"Pilot has only {answer_token_count} answer tokens; "
        "cannot check all requested positions.",
    )

    print("\nSix-position numerical check:")
    print(
        "Position | Metric           | Saved value    | Reference      "
        "| Absolute diff | Result"
    )
    print("-" * 91)

    checks = [
        ("logit", "logit", reference_target_logits, LOGIT_ATOL, LOGIT_RTOL),
        (
            "probability",
            "probability",
            reference_probs,
            PROB_ATOL,
            PROB_RTOL,
        ),
        (
            "log_probability",
            "log_probability",
            reference_logprobs,
            LOGPROB_ATOL,
            LOGPROB_RTOL,
        ),
        ("entropy", "entropy", reference_entropy, ENTROPY_ATOL, ENTROPY_RTOL),
    ]

    all_numeric_checks_passed = True

    for position in CHECK_POSITIONS:
        saved_row = features.loc[position]

        for saved_column, label, reference_values, atol, rtol in checks:
            saved_value = float(saved_row[saved_column])
            reference_value = float(reference_values[position].item())
            difference = abs(saved_value - reference_value)

            passed = np.isclose(
                saved_value,
                reference_value,
                atol=atol,
                rtol=rtol,
            )

            print(
                f"{position:8d} | {label:16s} | "
                f"{saved_value:14.7f} | {reference_value:14.7f} | "
                f"{difference:13.7g} | {'PASS' if passed else 'FAIL'}"
            )

            if not passed:
                all_numeric_checks_passed = False

    # --------------------------------------------------------
    # 5. Compare saved float16 hidden states with reference
    # --------------------------------------------------------

    print("\nChecking hidden states...")

    # The last element is the final transformer layer's hidden state.
    reference_hidden = output.hidden_states[-1][
        0,
        prompt_length : prompt_length + answer_token_count,
        :,
    ].cpu().numpy()

    # PyTorch may default to weights_only=True in newer versions.
    # This file is the locally generated pilot artifact.
    try:
        saved_hidden_data = torch.load(
            HIDDEN_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        saved_hidden_data = torch.load(
            HIDDEN_PATH,
            map_location="cpu",
        )

    saved_hidden = saved_hidden_data["hidden_states"].cpu().numpy()

    require(
        saved_hidden.shape == reference_hidden.shape,
        f"Hidden-state shape mismatch: saved={saved_hidden.shape}, "
        f"reference={reference_hidden.shape}",
    )

    hidden_passed = np.allclose(
        saved_hidden.astype(np.float32),
        reference_hidden.astype(np.float32),
        atol=HIDDEN_ATOL,
        rtol=HIDDEN_RTOL,
    )

    max_abs_difference = float(
        np.max(
            np.abs(
                saved_hidden.astype(np.float32)
                - reference_hidden.astype(np.float32)
            )
        )
    )
    mean_abs_difference = float(
        np.mean(
            np.abs(
                saved_hidden.astype(np.float32)
                - reference_hidden.astype(np.float32)
            )
        )
    )

    print(f"Saved hidden-state shape: {saved_hidden.shape}")
    print(f"Max absolute difference:  {max_abs_difference:.8f}")
    print(f"Mean absolute difference: {mean_abs_difference:.8f}")
    print(
        f"Float16-tolerance comparison: "
        f"{'PASS' if hidden_passed else 'FAIL'}"
    )

    # --------------------------------------------------------
    # 6. Final status
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    if all_numeric_checks_passed and hidden_passed:
        print("RESULT: ALL PILOT VALIDATION CHECKS PASSED")
    else:
        print("RESULT: VALIDATION NEEDS INVESTIGATION")
        if not all_numeric_checks_passed:
            print("- One or more sampled numeric feature checks failed.")
        if not hidden_passed:
            print("- Hidden states did not match within the configured tolerance.")
    print("=" * 55)

    if not all_numeric_checks_passed or not hidden_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()