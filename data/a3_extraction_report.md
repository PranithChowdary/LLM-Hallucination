# A3 Token-Level Feature Extraction Report

## Final status: VALIDATION PASSED

- Expected examples: 24,652
- Extracted examples: 24,652
- Token-feature rows: 2,016,270
- Feature Parquet shards: 247
- Hidden-state files: 24,652
- Hidden size: 896
- Hidden-state storage: float16

## Output files

- Token features: `data/a3/token_features/part_*.parquet`
- Hidden states: `data/a3/hidden_states/*.pt`
- Machine-readable manifest: `data/a3/a3_manifest.json`

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
