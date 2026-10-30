from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("data")
OUT = ROOT / "b7"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load the B1 and linguistic feature tables
b1 = pd.read_parquet(ROOT / "processed/b1_answer_features.parquet")
ling = pd.read_parquet(ROOT / "a3/linguistic_features.parquet")
labels = pd.read_parquet(ROOT / "processed/generation_labels.parquet")

# 2. Keep numerical features only; exclude IDs and metadata
b1_features = b1.drop(columns=["uid"])
ling_features = ling.drop(
    columns=["uid", "dataset", "subset"],
    errors="ignore"
)

# Prefix names so the source of every feature is clear
b1_features = b1_features.rename(
    columns=lambda c: f"b1_{c}"
)
ling_features = ling_features.rename(
    columns=lambda c: f"ling_{c}"
)

# 3. Merge by UID and verify one-to-one alignment
features = b1[["uid"]].merge(
    b1_features.assign(uid=b1["uid"]),
    on="uid",
    validate="one_to_one"
)

features = features.merge(
    ling_features.assign(uid=ling["uid"]),
    on="uid",
    validate="one_to_one"
)

assert features["uid"].is_unique
assert len(features) == len(b1) == len(ling)

# Save the merged feature table for reproducibility
features.to_parquet(OUT / "feature_table.parquet", index=False)

numeric = features.drop(columns=["uid"]).select_dtypes(
    include="number"
)

# 4. Correlation matrix between features
corr = numeric.corr(method="pearson")
corr.to_csv(OUT / "correlation_matrix.csv")

# 5. Identify highly correlated feature pairs
pairs = []
cols = corr.columns.tolist()

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        value = corr.iloc[i, j]
        if pd.notna(value) and abs(value) >= 0.80:
            pairs.append({
                "feature_1": cols[i],
                "feature_2": cols[j],
                "correlation": value,
                "absolute_correlation": abs(value),
            })

pairs_df = pd.DataFrame(
    pairs,
    columns=[
        "feature_1", "feature_2",
        "correlation", "absolute_correlation"
    ]
)

if not pairs_df.empty:
    pairs_df = pairs_df.sort_values(
        "absolute_correlation", ascending=False
    )

pairs_df.to_csv(OUT / "high_correlation_pairs.csv", index=False)

# 6. Plot a correlation heatmap using matplotlib
fig_size = max(10, len(cols) * 0.75)
fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

image = ax.imshow(
    corr.to_numpy(),
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=8)
ax.set_yticks(range(len(cols)))
ax.set_yticklabels(cols, fontsize=8)
ax.set_title("B7: Feature Correlation Matrix")
fig.colorbar(image, ax=ax, label="Pearson correlation")
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmap.png", dpi=200)
plt.close(fig)

# 7. Explore correlations with the provisional label separately
target = labels[["uid", "hallucinated"]]
analysis = features.merge(
    target, on="uid", validate="one_to_one"
)
analysis = analysis.dropna(subset=["hallucinated"])

target_corr = analysis.drop(columns=["uid"]).corr(
    method="pearson"
)["hallucinated"].drop("hallucinated")

target_corr = target_corr.sort_values(
    key=lambda s: s.abs(), ascending=False
)

target_corr.rename("correlation_with_provisional_label").to_csv(
    OUT / "target_correlations.csv"
)

# 8. Write a summary report
with open(OUT / "report.txt", "w", encoding="utf-8") as f:
    f.write("B7 FEATURE CORRELATION ANALYSIS\n")
    f.write("=" * 40 + "\n")
    f.write(f"Feature rows: {len(features)}\n")
    f.write(f"Numeric features: {len(cols)}\n")
    f.write("Correlation method: Pearson\n")
    f.write("High-correlation threshold: absolute r >= 0.80\n")
    f.write("Labels: provisional-nli-v1; not verified ground truth\n\n")

    f.write("HIGHLY CORRELATED FEATURE PAIRS\n")
    f.write("-" * 40 + "\n")
    if pairs_df.empty:
        f.write("No pairs met the threshold.\n")
    else:
        f.write(pairs_df.to_string(index=False) + "\n")

    f.write("\nFEATURE CORRELATION WITH PROVISIONAL LABEL\n")
    f.write("-" * 40 + "\n")
    f.write(target_corr.to_string() + "\n")

print("B7 completed.")
print("Rows:", len(features))
print("Numeric features:", len(cols))
print("Highly correlated pairs:", len(pairs_df))
print("\nTop correlations with provisional label:")
print(target_corr.head(10).to_string())
print("\nFiles saved in:", OUT)
