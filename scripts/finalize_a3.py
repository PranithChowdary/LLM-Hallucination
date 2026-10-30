"""
A3 final validation and documentation.

Run from the repository root after extraction:
    python scripts/finalize_a3.py

This script does not load the language model or generate answers.
It validates saved features and hidden states, then writes the final report.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch


# ---------------------------- Paths --------------------------------

GENERATION_PATH = Path("data/generation_qwen_calibration.parquet")
CALIBRATION_PATH = Path("data/calibration.parquet")

A3_DIR = Path("data/a3")
FEATURE_DIR = A3_DIR / "token_features"
HIDDEN_DIR = A3_DIR / "hidden_states"

MANIFEST_PATH = A3_DIR / "a3_manifest.json"
REPORT_PATH = Path("a3_extraction_report.md")

REQUIRED_COLUMNS = {
    "uid",
    "token_position",
    "token_id",
    "token_text",
    "logit",
    "probability",
    "log_probability",
    "entropy",
}


def hidden_filename(uid):
    """Must match the hash-based filename convention in the extractor."""
    return hashlib.sha256(str(uid).encode("utf-8")).hexdigest() + ".pt"


def fail_if(condition, message):
    """Raise a clear error when a validation condition fails."""
    if condition:
        raise ValueError(message)


def main():
    # ----------------------- Load expected inputs -------------------

    for path in (GENERATION_PATH, CALIBRATION_PATH, FEATURE_DIR, HIDDEN_DIR):
        fail_if(not path.exists(), f"Required input/output path is missing: {path}")

    generations = pd.read_parquet(GENERATION_PATH)
    calibration = pd.read_parquet(CALIBRATION_PATH)

    expected_uids = set(generations["uid"].astype(str))
    calibration_uids = set(calibration["uid"].astype(str))

    fail_if(
        expected_uids != calibration_uids,
        "Generation UIDs do not exactly match calibration UIDs.",
    )

    # ----------------------- Load all feature shards ----------------

    shard_paths = sorted(FEATURE_DIR.glob("part_*.parquet"))
    fail_if(not shard_paths, "No token-feature Parquet shards were found.")

    frames = []

    for path in shard_paths:
        frame = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)

        fail_if(
            bool(missing),
            f"{path} is missing required columns: {sorted(missing)}",
        )

        frames.append(frame)

    features = pd.concat(frames, ignore_index=True)
    features["uid"] = features["uid"].astype(str)

    fail_if(features.empty, "The combined token-feature table is empty.")
    fail_if(
        features[list(REQUIRED_COLUMNS)].isna().any().any(),
        "Null values found in required token-feature columns.",
    )

    # ----------------------- Validate UID coverage ------------------

    extracted_uids = set(features["uid"])
    missing_uids = expected_uids - extracted_uids
    extra_uids = extracted_uids - expected_uids

    fail_if(
        bool(missing_uids or extra_uids),
        "UID coverage mismatch. "
        f"Missing count={len(missing_uids)}; extra count={len(extra_uids)}. "
        f"Missing examples={sorted(missing_uids)[:10]}; "
        f"extra examples={sorted(extra_uids)[:10]}",
    )

    fail_if(
        features.duplicated(["uid", "token_position"]).any(),
        "Duplicate (uid, token_position) rows were found.",
    )

    # ----------------------- Validate numeric values -----------------

    numeric_columns = [
        "logit",
        "probability",
        "log_probability",
        "entropy",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(features[column], errors="coerce")
        fail_if(
            values.isna().any(),
            f"Non-numeric or missing values found in {column}.",
        )
        fail_if(
            not values.map(math.isfinite).all(),
            f"Non-finite values found in {column}.",
        )

    fail_if(
        not features["probability"].between(0.0, 1.0).all(),
        "Probability values fall outside [0, 1].",
    )

    # Log-probabilities should be <= 0, allowing a tiny numerical tolerance.
    fail_if(
        not features["log_probability"].le(1e-6).all(),
        "Unexpected positive log-probability values found.",
    )

    # Entropy should be non-negative, allowing tiny floating-point error.
    fail_if(
        not features["entropy"].ge(-1e-6).all(),
        "Negative entropy values found.",
    )

    # ----------------------- Validate token positions ----------------

    token_counts = features.groupby("uid").size()

    for uid, group in features.groupby("uid", sort=False):
        positions = sorted(group["token_position"].astype(int).tolist())

        fail_if(
            positions != list(range(len(positions))),
            f"Token positions are not consecutive from zero for UID {uid}.",
        )

    # ----------------------- Validate hidden-state files --------------

    hidden_sizes = set()
    missing_hidden_files = []
    invalid_hidden_files = []

    for uid in sorted(expected_uids):
        path = HIDDEN_DIR / hidden_filename(uid)

        if not path.is_file():
            missing_hidden_files.append(uid)
            continue

        try:
            # The extractor creates these files itself using torch.save.
            record = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # Compatibility fallback for PyTorch versions without weights_only.
            record = torch.load(path, map_location="cpu")
        except Exception as exc:
            invalid_hidden_files.append((uid, f"load failed: {exc}"))
            continue

        if record.get("uid") != uid:
            invalid_hidden_files.append((uid, "UID mismatch inside hidden-state file"))
            continue

        hidden = record.get("hidden_states")

        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
            invalid_hidden_files.append((uid, "hidden_states is not a 2D tensor"))
            continue

        expected_token_count = int(token_counts.loc[uid])

        if hidden.shape[0] != expected_token_count:
            invalid_hidden_files.append(
                (
                    uid,
                    f"hidden rows={hidden.shape[0]}, token rows={expected_token_count}",
                )
            )
            continue

        if not torch.isfinite(hidden).all():
            invalid_hidden_files.append((uid, "hidden states contain non-finite values"))
            continue

        hidden_sizes.add(int(hidden.shape[1]))

    fail_if(
        bool(missing_hidden_files),
        f"Missing hidden-state files: {len(missing_hidden_files)}. "
        f"Example UIDs: {missing_hidden_files[:10]}",
    )

    fail_if(
        bool(invalid_hidden_files),
        f"Invalid hidden-state files: {len(invalid_hidden_files)}. "
        f"Examples: {invalid_hidden_files[:10]}",
    )

    fail_if(
        len(hidden_sizes) != 1,
        f"Expected one consistent hidden size; found {sorted(hidden_sizes)}.",
    )

    # ----------------------- Build final manifest --------------------

    previous_manifest = {}

    if MANIFEST_PATH.is_file():
        previous_manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )

    total_examples = len(expected_uids)
    total_token_rows = len(features)
    hidden_size = next(iter(hidden_sizes))

    manifest = {
        **previous_manifest,
        "status": "validated",
        "expected_examples": total_examples,
        "extracted_examples": features["uid"].nunique(),
        "token_feature_rows": total_token_rows,
        "feature_shard_count": len(shard_paths),
        "feature_shards": [str(path) for path in shard_paths],
        "hidden_state_file_count": total_examples,
        "hidden_size": hidden_size,
        "hidden_state_storage_dtype": "float16",
        "validation": {
            "generation_uid_coverage_matches_calibration": True,
            "feature_uid_coverage_exact": True,
            "no_duplicate_uid_token_positions": True,
            "numeric_features_finite": True,
            "probabilities_in_range": True,
            "log_probabilities_non_positive": True,
            "entropy_non_negative": True,
            "token_positions_consecutive": True,
            "all_hidden_state_files_present": True,
            "hidden_state_rows_match_token_rows": True,
            "hidden_states_finite": True,
            "hidden_size_consistent": True,
        },
        "methodology": {
            "teacher_forced_forward_pass": True,
            "new_answers_generated": False,
            "token_ids_reconstructed_from_saved_text": True,
            "scores_are_original_a2_sampling_scores": False,
            "hidden_states_from_final_transformer_layer": True,
            "full_vocabulary_logits_saved": False,
            "hallucination_labels_created": False,
        },
    }

    temporary_manifest = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, MANIFEST_PATH)

    # ----------------------- Generate final report -------------------

    report = f"""# A3 Token-Level Feature Extraction Report

## Final status: VALIDATION PASSED

- Expected examples: {total_examples:,}
- Extracted examples: {features["uid"].nunique():,}
- Token-feature rows: {total_token_rows:,}
- Feature Parquet shards: {len(shard_paths)}
- Hidden-state files: {total_examples:,}
- Hidden size: {hidden_size}
- Hidden-state storage: float16

## Output files

- Token features: `{FEATURE_DIR}/part_*.parquet`
- Hidden states: `{HIDDEN_DIR}/*.pt`
- Machine-readable manifest: `{MANIFEST_PATH}`

## Token-feature columns

`uid`, `token_position`, `token_id`, `token_text`, `logit`,
`probability`, `log_probability`, `entropy`

Each hidden-state file contains the UID, token positions, final-layer
hidden-state matrix, hidden size, and storage metadata.

## Method

For each saved A2 answer, the extraction script reconstructs token IDs from
the saved decoded answer text, concatenates those tokens after the prompt
rendered using the model's user-message chat template, and performs a
teacher-forced forward pass.

For each answer token, it records the score of that token under the
recomputed next-token distribution and the final-layer hidden state at that
token's position. Full-vocabulary logits are used in memory to compute the
observed token's probability and the distribution entropy, but are not saved.

The script does not call `model.generate()` and does not create new answers.

## Methodological limitation

A2 saved decoded answer text rather than the original generation-time token
IDs, logits, or hidden states. Therefore, A3 token IDs are reconstructed from
text, and the numerical scores and hidden states are recomputed. These are
**not the original A2 sampling-time scores**. Reconstructed tokenization may
differ from the original sampled token sequence.

## Validation checks

- Generation UIDs exactly match calibration UIDs.
- Feature UIDs exactly cover the generation set.
- No duplicate `(uid, token_position)` rows exist.
- Token positions are consecutive from zero within each answer.
- Numeric features are finite.
- Probabilities are in `[0, 1]`.
- Log-probabilities are non-positive within numerical tolerance.
- Entropy is non-negative within numerical tolerance.
- Every UID has a valid hidden-state file.
- Hidden-state row counts match token-feature row counts.
- Hidden states are finite and share a consistent hidden size.

## Scope

A3 extracts numerical token-level features. It does not determine whether an
answer is a hallucination, assign hallucination labels, or prove factual
correctness from a probability or entropy score alone.

## Reproducibility

The model path and run metadata are recorded in the manifest. For a stronger
reproducibility record, also document the exact model checkpoint/revision,
Python/PyTorch/Transformers versions, and GPU model used for the run.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("A3 final validation PASSED")
    print(f"Examples: {total_examples:,}")
    print(f"Token-feature rows: {total_token_rows:,}")
    print(f"Feature shards: {len(shard_paths)}")
    print(f"Hidden-state files: {total_examples:,}")
    print(f"Manifest written to: {MANIFEST_PATH}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()