
"""
Create documentation for the validated A3 single-example pilot.

Run from the project root:
    python create_a3_pilot_docs.py

This script documents the validation results already observed.
It does not rerun extraction or independently validate the artifacts.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

OUTPUT_DIR = Path("data/a3_test")
MANIFEST_PATH = OUTPUT_DIR / "a3_manifest.json"
REPORT_PATH = OUTPUT_DIR / "a3_extraction_report.md"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Manifest: machine-readable pilot record
# ---------------------------------------------------------

manifest = {
    "task": "A3 token-level feature extraction pilot",
    "status": "single_example_pilot_validated",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": {
        "pilot_uid": "halueval-qa-00000",
        "answer_token_count": 128,
        "hidden_state_shape": [128, 896],
        "execution_device": "CPU",
        "full_dataset_validated": False
    },
    "model": {
        "local_path": "models/Qwen2.5-0.5B-Instruct",
        "checkpoint_revision_or_hash": None,
        "note": (
            "The validation console output did not record a checkpoint "
            "revision or hash. Confirm checkpoint identity before "
            "production extraction."
        )
    },
    "inputs": {
        "generation_table": "data/generation_qwen_calibration.parquet",
        "prompt_source": "saved A2 prompt",
        "answer_source": "saved A2 generated_answer text"
    },
    "outputs": {
        "token_features": "data/a3_test/token_features.parquet",
        "hidden_states": "data/a3_test/hidden_states.pt",
        "example_summary": "data/a3_test/example_summary.json",
        "validation_script": "validate_a3_pilot.py"
    },
    "method": {
        "forward_pass": "teacher-forced",
        "text_generation_performed": False,
        "answer_token_ids": (
            "Reconstructed by tokenizing the saved decoded answer text"
        ),
        "token_features": [
            "target-token logit",
            "target-token probability",
            "target-token log probability",
            "entropy of the full next-token distribution"
        ],
        "hidden_states": (
            "Final transformer layer at answer-token positions"
        ),
        "hidden_state_storage_dtype": "float16",
        "score_interpretation": (
            "Scores were recomputed from a forward pass. They are not "
            "the original generation-time sampling scores from A2."
        )
    },
    "validation": {
        "basic_artifact_checks": "PASS",
        "feature_rows": 128,
        "consecutive_token_positions": "PASS",
        "finite_numeric_values": "PASS",
        "probability_range": "PASS",
        "sampled_zero_based_positions": [0, 1, 4, 32, 64, 127],
        "target_token_logits": "PASS at all six sampled positions",
        "probabilities": "PASS at all six sampled positions",
        "log_probabilities": "PASS at all six sampled positions",
        "entropy": "PASS at all six sampled positions",
        "sampled_numeric_absolute_differences": 0,
        "hidden_state_shape": {
            "saved": [128, 896],
            "reference": [128, 896],
            "result": "PASS"
        },
        "hidden_state_comparison": {
            "result": "PASS",
            "atol": 0.01,
            "rtol": 0.01,
            "max_absolute_difference": 0.06146240,
            "mean_absolute_difference": 0.00089843
        },
        "overall_result": "ALL PILOT VALIDATION CHECKS PASSED"
    },
    "limitations": [
        "Only one example was validated; full-dataset validation remains.",
        "Six positions were spot-checked for numeric features, not all 128.",
        "Reconstructed token IDs may not exactly match the original internal "
        "A2 token sequence because A2 saved decoded text.",
        "A3 scores are recomputed, not recovered A2 generation-time scores.",
        "The exact model checkpoint identity still needs confirmation.",
        "This feature-extraction stage does not assign hallucination labels."
    ],
    "provenance": (
        "Validation results recorded from the console output supplied "
        "for validate_a3_pilot.py; this script does not rerun validation."
    )
}

# ---------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------

report = r"""# A3 Single-Example Pilot Extraction Report

## Status

**Result: ALL PILOT VALIDATION CHECKS PASSED**

This report records the observed results from the single-example CPU
pilot. It does not claim that the full calibration dataset has been
processed or validated.

## 1. Scope

- **Pilot UID:** `halueval-qa-00000`
- **Answer feature rows:** 128
- **Final-layer hidden-state shape:** `(128, 896)`
- **Execution device:** CPU
- **Model path:** `models/Qwen2.5-0.5B-Instruct`

### Input and output artifacts

| Purpose | Path |
|---|---|
| A2 generation input | `data/generation_qwen_calibration.parquet` |
| Token features | `data/a3_test/token_features.parquet` |
| Hidden states | `data/a3_test/hidden_states.pt` |
| Example summary | `data/a3_test/example_summary.json` |
| Validator | `validate_a3_pilot.py` |

## 2. Method

A3 uses a teacher-forced forward pass over the prompt followed by the
saved A2 answer. It does not generate new text.

Answer token IDs are reconstructed by tokenizing the saved decoded
answer text. The forward pass is used to calculate the target token's
logit, probability, log probability, and entropy of the full next-token
distribution. Final-layer hidden states are collected at answer-token
positions and saved in float16.

**Important:** A2 saved decoded answer text, not the original
generation-time token IDs, logits, or hidden states. A3 therefore
reconstructs token IDs and recomputes scores. These must not be described
as recovered original A2 sampling scores.

## 3. Validation results

### Artifact and structure checks

- UID matched the summary: `halueval-qa-00000`.
- The feature table contained 128 rows.
- Token positions were consecutive from zero.
- Numeric feature values were finite.
- Probabilities were within the valid range.
- Reconstructed answer token count matched the feature-row count.

**Result: PASS**

### Numerical spot checks

The validator checked zero-based answer-token positions
**0, 1, 4, 32, 64, and 127**.

| Feature | Result |
|---|---|
| Target-token logit | PASS at all six positions |
| Target-token probability | PASS at all six positions |
| Target-token log probability | PASS at all six positions |
| Full-distribution entropy | PASS at all six positions |

The supplied console output showed an absolute difference of zero for
each sampled numerical feature comparison.

This is a targeted spot check, not an exhaustive check of all 128 tokens.

### Hidden-state comparison

| Measurement | Result |
|---|---:|
| Saved shape | `(128, 896)` |
| Reference shape | `(128, 896)` |
| Maximum absolute difference | `0.06146240` |
| Mean absolute difference | `0.00089843` |
| Absolute tolerance | `0.01` |
| Relative tolerance | `0.01` |
| `np.allclose` result | PASS |

The comparison passed using both absolute and relative tolerance.
The maximum absolute difference alone does not determine the result,
because relative tolerance also contributes to the comparison.

## 4. Limitations and production follow-ups

1. This is a single-example pilot, not full-dataset validation.
2. Only six positions were checked for logits, probabilities,
   log probabilities, and entropy.
3. Token IDs were reconstructed from decoded A2 answer text and may not
   exactly reproduce the original internal generation token sequence.
4. Scores were recomputed by teacher forcing; they are not original
   generation-time sampling scores.
5. The validation output did not record a checkpoint revision or hash.
   Confirm the local checkpoint matches A2 before production extraction.
6. This feature-extraction stage does not determine whether an answer
   is hallucinated.

## 5. Conclusion

The pilot passed the implemented structural checks, all six sampled
numerical comparisons, and the float16 hidden-state comparison.
Production extraction can proceed to implementation, with the
limitations above retained and full UID coverage and output integrity
validated during the production run.

---

*This report records the validation console output supplied for the
pilot. Creating this document does not rerun or independently verify
the validation.*
"""

# ---------------------------------------------------------
# Write files
# ---------------------------------------------------------

MANIFEST_PATH.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
)

REPORT_PATH.write_text(report, encoding="utf-8")

print("A3 pilot documentation created:")
print(f"  Manifest: {MANIFEST_PATH}")
print(f"  Report:   {REPORT_PATH}")