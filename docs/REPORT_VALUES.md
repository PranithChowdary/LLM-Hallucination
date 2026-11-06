# Report values: data and Chapter 3 text

Values computed from the repository data on 2026-11-05, at commit `261a32f`. They describe the run before the summarization rerun (`slurm/submit_regen.sh`), so sections 1, 3, 4, 5 and 7 will need recomputing after it.

Scope, unless stated otherwise:
- **Answers:** the 24,652 A2 main answers.
- **Labelled answers:** the 16,907 answers that are not refusals.
- **Labels:** `provisional-nli-v1`, which are not verified ground truth.

## Read this first: summarization answers are not summaries

A2 sent the prompt `Summarize the following document.` for all 7,998 summarization rows, without the document ([src/data/loaders.py:505](../src/data/loaders.py:505), [src/signals/labels.py:109](../src/signals/labels.py:109)). The 119,970 extra summarization samples have the same problem.

- 5,542 answers (69.3%) were caught as refusals, for example "there is no document provided".
- The other 2,456 were labelled, and they are non-answers too. Every one starts with "I apologize,", a wording the refusal filter in [src/signals/labels.py](../src/signals/labels.py) doesn't catch.
- 86.4% of those 2,456 are labelled hallucinated, because the labeller compares them with a document the model never saw.

So the summarization results measure "the model says it has no document", not summarization. They also make up 14.5% of the labelled pool behind every overall metric. Any table that includes them needs this caveat, and QA, dialogue and TruthfulQA should be treated as the valid subsets.

The same gap reaches the other subsets. Across all labelled answers, 3,842 of 16,907 (22.7%) start with "I apologize,": 2,456 summarization, 1,036 QA, 342 dialogue and 8 TruthfulQA. In QA, 96.8% of these are labelled hallucinated, against 78.7% of the other QA answers. Many are probably refusals, so the true hallucination rates are likely lower than the tables below show.

The dialogue and QA prompts also omit `context`, so those answers are closed-book. Reference comparison still works for QA, but dialogue answers are judged against knowledge the model didn't see.

## 1. Answers per subset

| Subset | Full dataset | Calibration (answers generated) | Test (no answers yet) | Refusals | Labelled | Hallucinated | % hallucinated |
|---|--:|--:|--:|--:|--:|--:|--:|
| HaluEval dialogue | 10,000 | 8,000 | 2,000 | 1,055 | 6,945 | 6,797 | 97.9 |
| HaluEval QA | 10,000 | 8,000 | 2,000 | 1,009 | 6,991 | 5,688 | 81.4 |
| HaluEval summarization | 9,998 | 7,998 | 2,000 | 5,542 | 2,456 | 2,122 | 86.4 |
| TruthfulQA | 817 | 654 | 163 | 139 | 515 | 406 | 78.8 |
| **Total** | **30,815** | **24,652** | **6,163** | **7,745** | **16,907** | **15,013** | **88.8** |

There are also 369,780 extra samples, 15 per calibration prompt, used by B3 and B4.

Two refusal counts exist, and they use different definitions. `generation_audit.md` reports 5,904 "refusal-like" answers (23.95%). The labeller excludes 7,745 (31.4%). Use 7,745 wherever labels are involved.

## 2. How the 24,652 answers were selected

Nothing was randomly sampled down. The 24,652 answers are the whole calibration split, with one answer per prompt ([src/data/freeze_dataset.py](../src/data/freeze_dataset.py), `data/split_manifest.json`).

1. Merge all of HaluEval (QA, dialogue, summarization) and TruthfulQA: 30,817 rows.
2. Remove rows with an exact duplicate normalized prompt/source key (`norm_key`): 2 rows, both summarization, leaving 30,815.
3. Flag 2 rows whose answer appears in both the correct and incorrect reference lists. They are kept, and both fall in calibration.
4. Split with `StratifiedGroupKFold`: 5 folds, seed 42, stratified by dataset + subset, grouped by `norm_key`. The first fold is test and the other four are calibration.
5. Result: 24,652 calibration rows (80.0%) and 6,163 test rows (20.0%), with 0 shared `uid` and 0 shared `norm_key` values.
6. Generate one answer per calibration prompt with Qwen2.5-0.5B-Instruct: temperature 0.7, top-p 0.9, `max_new_tokens` 128, seed 42, float16.
7. For labelled analyses, drop the 7,745 refusals, leaving 16,907.

The test split has no generated answers yet.

## 3. Feature statistics

All 24,652 answers. No feature has missing values.

| Feature | Mean | SD | Min | P5 | P25 | Median | P75 | P95 | Max |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Mean token log-prob | -0.833 | 0.281 | -1.829 | -1.257 | -1.045 | -0.859 | -0.618 | -0.352 | -0.0657 |
| Min token log-prob | -3.715 | 1.006 | -21.346 | -4.811 | -4.354 | -3.928 | -3.144 | -1.837 | -0.320 |
| Token log-prob variance | 0.946 | 0.395 | 0.0097 | 0.269 | 0.647 | 0.991 | 1.229 | 1.521 | 5.174 |
| Mean token entropy (nats) | 1.244 | 0.455 | 0.193 | 0.568 | 0.841 | 1.270 | 1.596 | 1.957 | 9.135 |
| Low-confidence token fraction (p < 0.5) | 0.400 | 0.125 | 0.000 | 0.172 | 0.312 | 0.415 | 0.493 | 0.581 | 0.775 |
| Output length (tokens) | 81.8 | 40.9 | 5 | 27 | 40 | 82 | 128 | 128 | 130 |
| Word count | 64.4 | 30.6 | 1 | 23 | 34 | 65 | 94 | 107 | 119 |
| Hedge count | 0.256 | 0.557 | 0 | 0 | 0 | 0 | 0 | 1 | 4 |
| Hedge rate | 0.0038 | 0.0084 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0238 | 0.100 |
| Entity mentions (approx.) | 7.751 | 7.395 | 0 | 0 | 1 | 6 | 13 | 22 | 48 |
| Numeral count | 0.936 | 1.711 | 0 | 0 | 0 | 0 | 1 | 4 | 18 |
| Entity + numeral density | 0.118 | 0.101 | 0.000 | 0.000 | 0.0385 | 0.0909 | 0.178 | 0.311 | 0.780 |
| Assertion sentences (approx.) | 1.821 | 1.463 | 0 | 0 | 1 | 2 | 3 | 4 | 8 |
| Assertion density | 0.436 | 0.313 | 0.000 | 0.000 | 0.167 | 0.500 | 0.667 | 1.000 | 1.000 |
| Repeated word count | 13.436 | 10.187 | 0 | 1 | 4 | 12 | 22 | 31 | 59 |
| Repetition rate | 0.178 | 0.0879 | 0.000 | 0.0345 | 0.114 | 0.179 | 0.239 | 0.323 | 0.562 |

Answers with a value of exactly zero: hedge count 19,529 (79.2%), numeral count 15,993, assertion sentences 5,603, entity + numeral density 1,830, repeated words 899.

**Mean (SD) by subset**

| Feature | Dialogue | QA | Summarization | TruthfulQA |
|---|--:|--:|--:|--:|
| Mean token log-prob | -0.998 (0.235) | -0.887 (0.262) | -0.606 (0.172) | -0.936 (0.277) |
| Min token log-prob | -4.087 (0.816) | -4.033 (1.032) | -3.007 (0.759) | -3.946 (0.923) |
| Token log-prob variance | 1.146 (0.327) | 1.067 (0.373) | 0.618 (0.252) | 1.018 (0.326) |
| Mean token entropy (nats) | 1.570 (0.367) | 1.338 (0.384) | 0.812 (0.201) | 1.376 (0.367) |
| Low-confidence token fraction (p < 0.5) | 0.472 (0.102) | 0.418 (0.119) | 0.307 (0.0859) | 0.448 (0.133) |
| Output length (tokens) | 98.8 (35.7) | 102.3 (34.6) | 42.1 (15.170) | 108.6 (35.4) |
| Word count | 76.9 (28.1) | 77.6 (26.7) | 36.8 (13.633) | 88.2 (29.3) |
| Hedge count | 0.187 (0.498) | 0.301 (0.629) | 0.260 (0.499) | 0.524 (0.792) |
| Hedge rate | 0.0022 (0.0061) | 0.0035 (0.0074) | 0.0056 (0.0108) | 0.0055 (0.0082) |
| Entity mentions (approx.) | 10.116 (6.511) | 11.932 (7.612) | 1.314 (0.957) | 6.407 (5.657) |
| Numeral count | 1.082 (1.790) | 1.672 (2.005) | 0.0031 (0.108) | 1.564 (1.987) |
| Entity + numeral density | 0.147 (0.0928) | 0.175 (0.105) | 0.0358 (0.0215) | 0.0940 (0.0723) |
| Assertion sentences (approx.) | 2.097 (1.230) | 2.771 (1.322) | 0.529 (0.675) | 2.615 (1.485) |
| Assertion density | 0.460 (0.264) | 0.622 (0.265) | 0.223 (0.276) | 0.487 (0.274) |
| Repeated word count | 15.195 (8.795) | 19.410 (10.387) | 5.154 (3.930) | 20.1 (9.942) |
| Repetition rate | 0.176 (0.0775) | 0.228 (0.0956) | 0.129 (0.0551) | 0.206 (0.0888) |

**Labelled answers by class** (16,907). AUROC below 0.5 means higher values go with "not hallucinated".

| Feature | Mean, not hallucinated (n=1,894) | Mean, hallucinated (n=15,013) | r with label | AUROC |
|---|--:|--:|--:|--:|
| Mean token log-prob | -0.869 | -0.928 | -0.072 | 0.419 |
| Min token log-prob | -3.995 | -3.999 | -0.001 | 0.495 |
| Token log-prob variance | 1.049 | 1.079 | 0.026 | 0.530 |
| Mean token entropy (nats) | 1.282 | 1.410 | 0.093 | 0.601 |
| Low-confidence token fraction (p < 0.5) | 0.416 | 0.442 | 0.073 | 0.587 |
| Output length (tokens) | 100.8 | 95.2 | -0.047 | 0.442 |
| Word count | 75.5 | 73.9 | -0.018 | 0.488 |
| Hedge count | 0.197 | 0.226 | 0.017 | 0.510 |
| Hedge rate | 0.0024 | 0.0027 | 0.012 | 0.509 |
| Entity mentions (approx.) | 11.614 | 9.826 | -0.075 | 0.439 |
| Numeral count | 1.680 | 1.260 | -0.069 | 0.437 |
| Entity + numeral density | 0.171 | 0.143 | -0.081 | 0.431 |
| Assertion sentences (approx.) | 2.795 | 2.183 | -0.138 | 0.386 |
| Assertion density | 0.583 | 0.496 | -0.095 | 0.410 |
| Repeated word count | 19.473 | 16.340 | -0.100 | 0.412 |
| Repetition rate | 0.240 | 0.201 | -0.147 | 0.371 |

## 4. Baseline results

**B6, naive log-probability** (mean negative token log-prob, labelled answers). Random AUPRC equals the hallucinated fraction.

| Subset | n | % hallucinated | AUROC | 95% CI | AUPRC | AURAC |
|---|--:|--:|--:|---|--:|--:|
| All | 16,907 | 88.8 | 0.581 | 0.568 to 0.593 | 0.918 | 0.131 |
| Dialogue | 6,945 | 97.9 | 0.596 | 0.547 to 0.642 | 0.984 | 0.028 |
| QA | 6,991 | 81.4 | 0.528 | 0.513 to 0.544 | 0.838 | 0.180 |
| Summarization | 2,456 | 86.4 | 0.522 | 0.490 to 0.555 | 0.874 | 0.141 |
| TruthfulQA | 515 | 78.8 | 0.506 | 0.444 to 0.570 | 0.780 | 0.218 |

**B6 thresholds** (5-fold mean)

| Rule | Threshold | Recall | Precision | F1 | FPR | Flag rate | Accuracy of accepted answers |
|---|--:|--:|--:|--:|--:|--:|--:|
| Recall 0.90 | 0.525 | 0.900 | 0.888 | 0.894 | 0.901 | 0.900 | 0.112 |
| Recall 0.95 | 0.429 | 0.950 | 0.888 | 0.918 | 0.954 | 0.951 | 0.107 |
| Youden | 1.019 | 0.408 | 0.924 | 0.566 | 0.265 | 0.392 | 0.135 |

**Other baselines, all subsets**

| Baseline | AUROC | 95% CI | Source |
|---|--:|---|---|
| Random | 0.500 | 0.486 to 0.513 | B3 notebook |
| Min token log-prob | 0.505 | 0.491 to 0.519 | B6 |
| Sum of token log-probs | 0.512 | 0.498 to 0.525 | B6 |
| Grey-box Max(-logP) | 0.554 | 0.540 to 0.568 | B3 notebook |
| Grey-box Max(H) | 0.579 | 0.566 to 0.591 | B3 notebook |
| Grey-box Avg(-logP) | 0.582 | 0.570 to 0.595 | B3 notebook |
| Naive entropy | 0.583 | 0.570 to 0.595 | B4 |
| Grey-box Avg(H) | 0.608 | 0.596 to 0.620 | B3 notebook |

On B8's single validation fold (n = 3,382), the same log-prob baseline scores 0.587, against 0.697 for logistic regression and 0.709 for gradient boosting.

## 5. Answers that hit the 128-token limit

**8,035 of 24,652 main answers (32.6%).**

| Subset | Answers | At limit | % |
|---|--:|--:|--:|
| Dialogue | 8,000 | 3,586 | 44.8 |
| QA | 8,000 | 4,004 | 50.1 |
| Summarization | 7,998 | 6 | 0.1 |
| TruthfulQA | 654 | 439 | 67.1 |
| **Total** | **24,652** | **8,035** | **32.6** |

- **This is an estimate.** A2 saved neither the stop reason nor the token IDs, so the count is answers with at least 128 tokens after re-tokenizing the saved text in A3.
- **Evidence:** 7,390 of the 8,035 (92.0%) don't end in sentence-final punctuation, against 166 of the other 16,617 (1.0%).
- **Uncertainty:** re-tokenizing can shift lengths slightly. 48 answers came out at 129 or 130 tokens, and 128 of the 205 answers at 127 tokens also lack final punctuation, so the true count may be a little higher.
- **Labelled answers:** 7,407 of 16,907 (43.8%) hit the limit. Of those, 85.5% are labelled hallucinated, against 91.4% of the rest.
- **Knock-on effect:** the limit caps output length, word count and every count-based feature. Truncated answers can also lose the part of the answer that would be judged.
- **Not measured:** the 369,780 extra samples, because the tokenizer isn't in this repository.

## 6. Software and hardware

| Item | Value | Evidence |
|---|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, about 96 GB, 1 GPU per job | A3 manifest, job logs |
| Cluster | Kalpana (IIT Bombay), SLURM, compute node `mangala`, 8 CPU cores per job | Job scripts |
| Precision | float16 for the LLM | A3 manifest |
| Python | 3.14.6 | B3 notebook metadata |
| CUDA | 13.0 | `+cu130` torch build |
| PyTorch | 2.13.0+cu130 | Preflight output, B3 to B6 runs |
| Transformers | 5.18.0 | Preflight output |
| scikit-learn | 1.9.0 | Preflight output |
| NumPy / pandas / PyArrow | 2.5.1 / 2.3.3 / 25.0.1 | Preflight output |
| bert-score / sentencepiece | 0.3.12 / 0.2.2 | Preflight output |
| matplotlib / tqdm | 3.11.1 / 4.68.2 | Preflight output |
| huggingface_hub / nbconvert / ipykernel | 1.33.0 / 7.17.1 / 7.3.0 | Preflight output |
| LLM | Qwen2.5-0.5B-Instruct (local copy) | Generation metadata |
| B4 to B6 code | Commit `6debefb` | Run configs |

Gaps to be careful about:
- These versions were recorded in the B3 to B6 preflight, not when A1, A2 and A3 ran, so don't state them as the exact A2 environment.
- A later preflight, on 2026-10-31, showed torch 2.14.1+cu130. That was after these experiments.
- `generation_audit.md` was produced on a separate Windows machine, and the B8 models were saved with scikit-learn 1.8.0, so B8 ran in a different environment. That environment's other versions aren't recorded.
- The repository doesn't record the driver version, CPU model, operating system or Hugging Face model revision. Run the commands below on the server to get them.

```bash
srun --account=cminds_anandi --partition=cn4_mangala --qos=mangala --gres=gpu:1 --ntasks=1 bash -c 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; lscpu | grep "Model name"; head -2 /etc/os-release'
```
```bash
head -1 ~/LLM_Hallucinations/models/Qwen2.5-0.5B-Instruct/.cache/huggingface/download/config.json.metadata
```

## 7. Hedging and repetition

**Decision:** keep the Chapter 4 plots and correct Chapter 3. Both features are already implemented in [src/extract_features/extract_linguistic_features.py](../src/extract_features/extract_linguistic_features.py), saved in `data/a3/linguistic_features.parquet`, and used by B7 and B8. "Future work" is out of date. Confirm that the plots use `hedge_rate` and `repetition_rate` from that file.

### Text for Chapter 3

> **Linguistic features.** We computed two linguistic features from the text of each generated answer. Answers were lowercased and split into word tokens with the regular expression `\b[\w'-]+\b`. Let $N$ be the number of word tokens in an answer.
>
> *Hedging rate.* We matched the answer against a fixed lexicon of 42 hedging phrases (for example, "as far as I know", "it seems", "may be") and 19 hedging words (for example, "perhaps", "likely", "roughly"). Phrases were matched first, longest first, and overlapping matches were counted once. With $h$ the number of matches, the hedging rate is
>
> $$\text{HedgeRate} = \frac{h}{N}$$
>
> *Repetition rate.* With $U$ the number of distinct word tokens in the answer, the repetition rate is the share of tokens that repeat an earlier token:
>
> $$\text{RepetitionRate} = \frac{N - U}{N} = 1 - \frac{U}{N}$$
>
> Both rates are 0 when $N = 0$. Both are heuristics: the hedging lexicon was hand-written and was not validated against human annotation, and the repetition rate counts function words and does not measure repeated phrases or ideas.

### Text for Chapter 4

> Hedging was rare: 79.2% of answers contained no hedge, and the mean hedging rate was 0.0038 (SD 0.0084). It carried almost no signal for the provisional hallucination label (r = 0.012, AUROC = 0.509). The mean repetition rate was 0.178 (SD 0.088). Repetition had the strongest label correlation of the 16 single-pass features (r = -0.147), with lower repetition in answers labelled hallucinated (0.201 against 0.240). This pooled result should be read with care. Repetition rate is correlated with answer length (r = 0.72), and it was close to chance within dialogue (AUROC 0.517) and summarization (0.513), while QA (0.404) and TruthfulQA (0.421) showed a modest effect. Part of the pooled effect therefore reflects differences between subsets.

| Feature | r with label | AUROC: all | Dialogue | QA | Summarization | TruthfulQA | B8 GB importance | B8 LR coefficient |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Hedge rate | 0.012 | 0.509 | 0.534 | 0.536 | 0.505 | 0.455 | 0.010 | -0.106 |
| Repetition rate | -0.147 | 0.371 | 0.517 | 0.404 | 0.513 | 0.421 | 0.196 | -0.438 |

Repetition rate is B8 gradient boosting's second most important feature, but that importance comes from the pooled data, so don't present it as evidence that repetition causes or reveals hallucination.

### What stays as future work

- Validating the hedging lexicon against human annotation
- Measuring phrase-level or semantic repetition, and excluding stopwords
- Controlling for answer length and subset
