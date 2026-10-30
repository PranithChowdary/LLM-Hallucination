
from pathlib import Path

import numpy as np
import pandas as pd


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
TOKEN_DIR = PROJECT_DIR / "data" / "a3" / "token_features"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "b1_answer_features.parquet"

# Threshold for identifying low-confidence tokens.
# A token is classified as low-confidence when its probability is below 0.5.
LOW_CONFIDENCE_THRESHOLD = 0.5


def main():
    # Find all extracted Parquet files
    parquet_files = sorted(TOKEN_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {TOKEN_DIR}"
        )

    print(f"Found {len(parquet_files)} Parquet files.")

    # Read token-level data from all shards
    frames = []

    for file_path in parquet_files:
        print(f"Reading {file_path.name}...")
        frames.append(pd.read_parquet(file_path))

    tokens = pd.concat(frames, ignore_index=True)

    # Check required columns
    required_columns = {
        "uid",
        "token_position",
        "probability",
        "log_probability",
        "entropy",
    }

    missing_columns = required_columns - set(tokens.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Ensure numerical columns contain numeric values
    numeric_columns = [
        "token_position",
        "probability",
        "log_probability",
        "entropy",
    ]

    for column in numeric_columns:
        tokens[column] = pd.to_numeric(
            tokens[column], errors="coerce"
        )

    # Remove rows that cannot contribute to the required features
    tokens = tokens.dropna(
        subset=[
            "uid",
            "token_position",
            "probability",
            "log_probability",
            "entropy",
        ]
    )

    # Aggregate token-level information into one row per answer
    features = (
        tokens.groupby("uid")
        .agg(
            mean_token_log_probability=("log_probability", "mean"),
            min_token_log_probability=("log_probability", "min"),
            variance_token_log_probability=("log_probability", "var"),
            average_token_entropy=("entropy", "mean"),
            low_confidence_token_fraction=(
                "probability",
                lambda values: (
                    values < LOW_CONFIDENCE_THRESHOLD
                ).mean(),
            ),
            output_token_length=("token_position", "size"),
        )
        .reset_index()
    )

    # A one-token answer has undefined sample variance.
    # Set its variance to zero because there is no variation to measure.
    features["variance_token_log_probability"] = (
        features["variance_token_log_probability"].fillna(0.0)
    )

    # Create output directory if necessary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the aggregated features
    features.to_parquet(OUTPUT_FILE, index=False)

    print("\nB1 feature extraction completed!")
    print(f"Unique answers processed: {len(features)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nFirst five feature rows:")
    print(features.head().to_string(index=False))


if __name__ == "__main__":
    main()
