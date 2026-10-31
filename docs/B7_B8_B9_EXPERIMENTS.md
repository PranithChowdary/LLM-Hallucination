# B7, B8 and B9: Hallucination Detection Experiments



## Research Context



These experiments investigate whether inexpensive linguistic and

probability-based signals can help identify hallucinated LLM responses.



The experiments use calibration data and provisional NLI-derived labels.

Therefore, all reported metrics are exploratory and must not be interpreted

as verified ground-truth or held-out test performance.



## B7 - Feature Correlation Analysis



**Script:** `src/eval/b7\_feature\_correlation.py`



### Objective

Analyze relationships among numerical hallucination-related features

and identify features associated with the provisional hallucination label.



### Method

- Construct a numerical feature table.

- Calculate Pearson correlation between numerical features.

- Identify highly correlated feature pairs using |r| >= 0.80.

- Analyze correlations between individual features and the provisional label.

- Keep calibration and test data separated.



### Outputs

- `data/b7/feature\_table.parquet`

- `data/b7/correlation\_matrix.csv`

- `data/b7/high\_correlation\_pairs.csv`

- `data/b7/correlation\_heatmap.png`

- `data/b7/report.txt`

- `data/b7/target\_correlations.csv`



### Exploratory observations

The strongest feature-to-label correlations were relatively weak:

- Repetition rate: -0.1465

- Assertion sentence count: -0.1384

- Repeated word count: -0.0998

- Assertion density: -0.0954

- Average token entropy: +0.0935



These correlations do not establish causation or reliable standalone

hallucination detection.



## B8 - Single-Pass Classifier



**Script:** `src/eval/b8\_single\_pass\_classifier.py`



### Objective

Evaluate whether inexpensive numerical features can support

single-pass hallucination classification.



### Models

1. Logistic Regression

   - Median imputation

   - StandardScaler

   - Class-weight balancing



2. Gradient Boosting Classifier

   - Median imputation

   - Balanced sample weights



### Evaluation

- 16 numerical features from B7

- StratifiedGroupKFold with 5 folds

- Random seed: 42

- First fold used for the reported validation evaluation

- Grouping by normalized prompt key (`norm\_key`), with UID fallback

- No overlap between training and validation groups



### Outputs

- `data/b8/logistic\_regression.joblib`

- `data/b8/gradient\_boosting.joblib`

- `data/b8/logistic\_coefficients.csv`

- `data/b8/gradient\_boosting\_importance.csv`

- `data/b8/metrics.json`

- `data/b8/split\_info.json`

- `data/b8/validation\_predictions.parquet`



### Validation results



| Metric | Logistic Regression | Gradient Boosting |

|---|---:|---:|

| AUROC | 0.6973 | 0.7093 |

| AUPRC | 0.9442 | 0.9469 |

| Accuracy | 0.6685 | 0.6641 |

| Precision | 0.9250 | 0.9300 |

| Recall | 0.6820 | 0.6723 |

| F1-score | 0.7851 | 0.7804 |



Validation set:

- 3,382 labelled rows

- Positive-label fraction: 88.79%

- Training/validation group overlap: 0



The high AUPRC must be interpreted in the context of the high positive-label

prevalence. These results are exploratory and are not held-out test results.



## B9 - Incremental Semantic Entropy Sampling-Budget Sweep



**Planned script:** `scripts/b9\_sampling\_budget.py`



### Objective

Study how semantic-entropy-based signals change as the number of sampled

answers increases, while reusing the expensive computations already

performed in B4.



### Planned approach

- Reuse B4's saved 15-answer NLI argmax matrices.

- Reuse B4's saved likelihood sums and token counts for the 15 generated answers.

- Add the original A2 main answer as answer index 0.

- Compute 30 new directed NLI comparisons between the main answer and

  the 15 generated samples.

- Teacher-forced-score only the main answer.

- Evaluate budgets N = 1, 2, 4, 8 and 16.



Answer order:



`\[main\_answer, sample\_1, ..., sample\_15]`



**Important:** B9's N=1 represents the main answer. B4's N=1 represented

the first generated sample.



B4 stores NLI predicted class labels, not the original probability

distributions. B9 reconstructs one-hot class probabilities, preserving

the predicted classes used by the clustering procedure.



### Execution status

B9 implementation is under preparation and has not yet been executed. Execution is planned in the original GPU environment after code review and submission.



### Interpretation caveats

- The labels are provisional NLI-derived labels.

- Results are exploratory calibration results.

- They must not be described as verified ground truth or held-out test performance.

- B4's original NLI probability distributions cannot be recovered from

  its saved argmax labels.



## Data and Evaluation Limitations



The provisional labels are derived from NLI-based comparisons and are not

independently verified ground truth.



The labelled calibration data contains:

- Total rows: 24,652

- Labelled rows: 16,907

- Provisional positive labels: 15,013

- Provisional negative labels: 1,894

- Unlabelled/refusal rows: 7,745



All reported B7 and B8 metrics are exploratory. No held-out test performance

is claimed by these experiments.
