from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# Paths and configuration
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "b8"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = ROOT / "data" / "b7" / "feature_table.parquet"
LABELS_PATH = ROOT / "data" / "processed" / "generation_labels.parquet"
CALIBRATION_PATH = ROOT / "data" / "calibration.parquet"
SCORES_PATH = ROOT / "data" / "b6" / "scores.parquet"

SEED = 42


# Data validation helpers
def require_unique_uid(df, name):
    if "uid" not in df.columns:
        raise ValueError(f"{name} is missing the uid column")

    if df["uid"].isna().any() or df["uid"].duplicated().any():
        raise ValueError(f"{name} has missing or duplicate UIDs")


def calculate_classifier_metrics(y_true, scores, predictions):
    return {
        "n": int(len(y_true)),
        "positive_fraction": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "accuracy_at_0_5": float(accuracy_score(y_true, predictions)),
        "precision_at_0_5": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_at_0_5": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_at_0_5": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def calculate_baseline_metrics(y_true, scores):
    # B6 scores are raw negative-log-probability values.
    # Report ranking metrics only; do not apply a probability threshold.
    return {
        "n": int(len(y_true)),
        "positive_fraction": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "threshold_metrics": (
            "Not calculated: B6 scores are not calibrated probabilities."
        ),
    }


# Load data
def main():
    for path in [
        FEATURES_PATH,
        LABELS_PATH,
        CALIBRATION_PATH,
        SCORES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)[
        ["uid", "hallucinated"]
    ]
    calibration = pd.read_parquet(CALIBRATION_PATH)[
        ["uid", "norm_key", "dataset", "subset"]
    ]
    baseline = pd.read_parquet(SCORES_PATH)[
        ["uid", "logprob_neg_mean"]
    ]

    for df, name in [
        (features, "B7 features"),
        (labels, "generation labels"),
        (calibration, "calibration data"),
        (baseline, "B6 scores"),
    ]:
        require_unique_uid(df, name)

    # Select numerical features only. UID is an identifier, not a feature.
    feature_columns = [
        column
        for column in features.columns
        if column != "uid"
        and pd.api.types.is_numeric_dtype(features[column])
    ]

    if not feature_columns:
        raise ValueError("No numerical feature columns were found.")

    # Join data safely by UID
    data = features.merge(
        labels,
        on="uid",
        how="inner",
        validate="one_to_one",
    )

    data = data.merge(
        calibration,
        on="uid",
        how="inner",
        validate="one_to_one",
    )

    data = data.merge(
        baseline,
        on="uid",
        how="left",
        validate="one_to_one",
    )

    # Exclude provisional refusals/unlabelled rows.
    data["hallucinated"] = pd.to_numeric(
        data["hallucinated"], errors="coerce"
    )

    data = data[
        data["hallucinated"].isin([0, 1])
    ].copy()

    data["hallucinated"] = data["hallucinated"].astype(int)

    if data.empty:
        raise ValueError("No labelled rows remain after filtering.")

    # Create grouping keys for leakage-safe splitting.
    data["group_key"] = data["norm_key"].astype("string")

    missing_group = (
        data["group_key"].isna()
        | (data["group_key"].str.strip() == "")
    )

    data.loc[missing_group, "group_key"] = (
        data.loc[missing_group, "uid"].astype(str)
    )

    X = data[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    y = data["hallucinated"].to_numpy()
    groups = data["group_key"].astype(str).to_numpy()

    # Group-aware train/validation split
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    train_idx, val_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    overlap = train_groups.intersection(val_groups)

    if overlap:
        raise ValueError(
            f"Group leakage detected: {len(overlap)} overlapping groups."
        )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    if len(np.unique(y_train)) != 2:
        raise ValueError("Training split must contain both labels.")

    if len(np.unique(y_val)) != 2:
        raise ValueError("Validation split must contain both labels.")

    # Logistic Regression pipeline
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ])

    # Gradient Boosting pipeline
    boosted = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            GradientBoostingClassifier(
                random_state=SEED,
            ),
        ),
    ])

    # Compute class weights using training labels only.
    train_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    # Fit models using training data only.
    logistic.fit(X_train, y_train)

    boosted.fit(
        X_train,
        y_train,
        model__sample_weight=train_weights,
    )

    # Generate validation predictions
    probabilities = {
        "logistic_regression": logistic.predict_proba(X_val)[:, 1],
        "gradient_boosting": boosted.predict_proba(X_val)[:, 1],
    }

    validation_data = pd.DataFrame({
        "uid": data.iloc[val_idx]["uid"].to_numpy(),
        "y_true": y_val,
    })

    metrics = {}

    for model_name, probs in probabilities.items():
        predictions = (probs >= 0.5).astype(int)

        metrics[model_name] = calculate_classifier_metrics(
            y_true=y_val,
            scores=probs,
            predictions=predictions,
        )

        validation_data[f"{model_name}_probability"] = probs
        validation_data[
            f"{model_name}_prediction_at_0_5"
        ] = predictions

    # Compare with the B6 log-probability baseline
    baseline_scores = pd.to_numeric(
        data.iloc[val_idx]["logprob_neg_mean"],
        errors="coerce",
    ).to_numpy()

    valid_baseline = np.isfinite(baseline_scores)

    if (
        valid_baseline.sum() > 0
        and len(np.unique(y_val[valid_baseline])) == 2
    ):
        metrics["b6_logprob_neg_mean"] = calculate_baseline_metrics(
            y_true=y_val[valid_baseline],
            scores=baseline_scores[valid_baseline],
        )

        validation_data["b6_logprob_neg_mean"] = baseline_scores
    else:
        metrics["b6_logprob_neg_mean"] = {
            "note": (
                "Insufficient valid baseline scores or only one "
                "class present in the baseline evaluation rows."
            )
        }

    # Save metrics and validation predictions
    with open(
        OUT / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2, allow_nan=False)

    validation_data.to_parquet(
        OUT / "validation_predictions.parquet",
        index=False,
    )

    # Save fitted models
    joblib.dump(
        logistic,
        OUT / "logistic_regression.joblib",
    )

    joblib.dump(
        boosted,
        OUT / "gradient_boosting.joblib",
    )

    # Save feature diagnostics
    logistic_coefficients = (
        logistic.named_steps["model"].coef_[0]
    )

    pd.DataFrame({
        "feature": feature_columns,
        "coefficient": logistic_coefficients,
        "absolute_coefficient": np.abs(logistic_coefficients),
    }).sort_values(
        "absolute_coefficient",
        ascending=False,
    ).to_csv(
        OUT / "logistic_coefficients.csv",
        index=False,
    )

    pd.DataFrame({
        "feature": feature_columns,
        "importance": (
            boosted.named_steps["model"].feature_importances_
        ),
    }).sort_values(
        "importance",
        ascending=False,
    ).to_csv(
        OUT / "gradient_boosting_importance.csv",
        index=False,
    )

    # Record split details and limitations.
    split_info = {
        "seed": SEED,
        "splitter": "StratifiedGroupKFold",
        "validation_fold": 0,
        "training_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "training_groups": int(len(train_groups)),
        "validation_groups": int(len(val_groups)),
        "train_validation_group_overlap": int(len(overlap)),
        "feature_count": int(len(feature_columns)),
        "features": feature_columns,
        "label_method": "provisional-nli-v1",
        "labels_are_verified_ground_truth": False,
        "held_out_test_used": False,
        "baseline_threshold_metrics_reported": False,
    }

    with open(
        OUT / "split_info.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(split_info, file, indent=2, allow_nan=False)

    # Print summary
    print("B8 completed successfully.")
    print(f"Usable labelled rows: {len(data)}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Validation rows: {len(val_idx)}")
    print(f"Numerical features: {len(feature_columns)}")
    print(f"Train/validation group overlap: {len(overlap)}")

    print("\nValidation metrics:")

    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")

        for metric_name, value in model_metrics.items():
            print(f"  {metric_name}: {value}")

    print(f"\nArtifacts saved in: {OUT}")
    print("\nNote: labels are provisional, not verified ground truth.")
    print("Held-out test data was not used.")


if __name__ == "__main__":
    main()
