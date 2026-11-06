"""
Collect the results of Blocks A and B into one small folder for git.

The experiments leave gigabytes behind (hidden states, token features,
shards). Their conclusions are small. This script reads what the
experiments saved and writes results/:

    README.md                       summary, checks and file map
    tables/*.csv                    every table, plus B3's notebook tables
    figures/*.png                   every figure, plus B3's notebook figures
    scores/per_answer_scores.parquet   one row per answer: label and detector
                                       scores, the input to Block C

It runs nothing heavy and does not need models, hidden states or token
features, so it works on the server or on a laptop.

Run from the repository root:
    python scripts/collect_results.py
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.common import (  # noqa: E402
    CALIBRATION_PATH,
    DATA_DIR,
    LABELS_PATH,
    MAIN_ANSWERS_PATH,
    PROJECT_ROOT,
    git_commit_sha,
)


OUTPUT_DIR = PROJECT_ROOT / "results"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "b3_selfcheckgpt.ipynb"
TEST_PATH = DATA_DIR / "test.parquet"

MAX_NEW_TOKENS = 128
SUBSETS = ["all", "dialogue", "qa", "summarization", "truthfulqa"]

# (start of the notebook cell's source, one name per table it displays)
B3_TABLES = [
    ("metrics = evaluate_scores", ["b3_metrics"]),
    ('display(Markdown("**AUROC per subset', ["b3_auroc_by_subset", "b3_auprc_by_subset"]),
    ("max_n = int(", ["b3_sample_sweep"]),
    ("if nli is not None:", ["b3_reference_evidence_auroc", "b3_reference_evidence_auprc"]),
    ("n_pairs = int(", ["b3_compute_cost"]),
]

# Files that are already results: (source under data/, name in tables/)
COPIED_TABLES = [
    ("b4/budget_sweep.csv", "b4_budget_sweep.csv"),
    ("b4/metrics.csv", "b4_metrics.csv"),
    ("b5/comparison_metrics.csv", "b5_comparison_metrics.csv"),
    ("b5/per_layer_metrics.csv", "b5_per_layer_metrics.csv"),
    ("b6/metrics.csv", "b6_metrics.csv"),
    ("b6/threshold_summary.csv", "b6_threshold_summary.csv"),
    ("b7/correlation_matrix.csv", "b7_correlation_matrix.csv"),
    ("b7/high_correlation_pairs.csv", "b7_high_correlation_pairs.csv"),
    ("b7/target_correlations.csv", "b7_target_correlations.csv"),
    ("b8/gradient_boosting_importance.csv", "b8_gradient_boosting_importance.csv"),
    ("b8/logistic_coefficients.csv", "b8_logistic_coefficients.csv"),
    ("b8/metrics.json", "b8_metrics.json"),
    ("b8/split_info.json", "b8_split_info.json"),
    ("b9/budget_sweep.csv", "b9_budget_sweep.csv"),
    ("b9/metrics.csv", "b9_metrics.csv"),
    ("split_manifest.json", "split_manifest.json"),
]

# (score name in the block's metrics, name in the detector table)
B4_DETECTORS = [
    ("semantic_entropy", "Semantic entropy"),
    ("discrete_semantic_entropy", "Discrete semantic entropy"),
    ("semantic_entropy_nonstrict", "Semantic entropy (non-strict)"),
    ("naive_entropy", "Naive entropy"),
]
B5_DETECTORS = ["SEP (TBG)", "SEP (SLT)", "Accuracy probe (TBG)", "Accuracy probe (SLT)"]
B6_DETECTORS = [
    ("logprob_neg_mean", "Log-prob, mean"),
    ("logprob_neg_min", "Log-prob, min"),
    ("logprob_neg_sum", "Log-prob, sum"),
]
B8_DETECTORS = [
    ("logistic_regression", "B8 logistic regression"),
    ("gradient_boosting", "B8 gradient boosting"),
]

SCORE_COLUMNS = {
    "b4": ["semantic_entropy", "discrete_semantic_entropy", "semantic_entropy_nonstrict",
           "naive_entropy", "n_clusters"],
    "b5": ["sep_tbg", "sep_slt", "accprobe_tbg", "accprobe_slt"],
    "b6": ["logprob_neg_mean", "logprob_neg_min", "logprob_neg_sum"],
}


# Small helpers
def html_table(markup: str) -> pd.DataFrame:
    """A pandas-rendered HTML table, with its index-name row folded in."""
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", markup, re.S):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)
        rows.append([html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells])
    header, *body = rows
    if body and not any(body[0][1:]):
        header[0] = body[0][0]
        body = body[1:]
    table = pd.DataFrame(body, columns=header)
    return table.drop(columns="") if "" in table.columns else table


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    def text(value) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.{digits}f}"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in frame.columns) + " |",
        "|" + "---|" * len(frame.columns),
    ]
    lines += ["| " + " | ".join(text(v) for v in row) + " |"
              for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def auroc(values, labels) -> float:
    ranks = pd.Series(values).rank().to_numpy()
    labels = np.asarray(labels)
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def last_commit_time(path: Path) -> int:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", str(path)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).stdout.strip()
    return int(out) if out else 0


# Tables
def dataset_counts(labels: pd.DataFrame) -> pd.DataFrame:
    calibration = pd.read_parquet(CALIBRATION_PATH, columns=["subset"])["subset"].value_counts()
    test = pd.read_parquet(TEST_PATH, columns=["subset"])["subset"].value_counts()

    table = labels.groupby("subset").agg(
        refusals=("refusal", "sum"),
        labelled=("hallucinated", "count"),
        hallucinated=("hallucinated", "sum"),
    )
    table.insert(0, "test_prompts", test)
    table.insert(0, "calibration_answers", calibration)
    table.loc["all"] = table.sum()
    table = table.astype(int)
    table["frac_hallucinated"] = table["hallucinated"] / table["labelled"]
    return table.reset_index()


def token_limit(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Reached = at least MAX_NEW_TOKENS after re-tokenizing the saved text."""
    frame = features[["uid", "b1_output_token_length"]].merge(labels, on="uid")
    frame["at_limit"] = frame["b1_output_token_length"] >= MAX_NEW_TOKENS

    def row(part: pd.DataFrame, name: str) -> dict:
        labelled = part[part["hallucinated"].notna()]
        return {
            "subset": name,
            "answers": len(part),
            "at_limit": int(part["at_limit"].sum()),
            "frac_at_limit": part["at_limit"].mean(),
            "frac_hallucinated_at_limit": labelled.loc[labelled["at_limit"], "hallucinated"].mean(),
            "frac_hallucinated_below_limit": labelled.loc[~labelled["at_limit"], "hallucinated"].mean(),
        }

    rows = [row(part, name) for name, part in frame.groupby("subset")]
    return pd.DataFrame(rows + [row(frame, "all")])


def feature_tables(features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    names = [c for c in features.columns if c != "uid"]
    frame = features.merge(labels, on="uid")

    stats = frame[names].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats["n_zero"] = (frame[names] == 0).sum()
    stats = stats.drop(columns="count").rename_axis("feature").reset_index()

    by_subset = frame.groupby("subset")[names].agg(["mean", "std"]).T
    by_subset = by_subset.rename_axis(["feature", "statistic"]).reset_index()

    labelled = frame[frame["hallucinated"].notna()].copy()
    labelled["y"] = labelled["hallucinated"].astype(int)
    rows = []
    for name in names:
        row = {
            "feature": name,
            "mean_not_hallucinated": labelled.loc[labelled["y"] == 0, name].mean(),
            "mean_hallucinated": labelled.loc[labelled["y"] == 1, name].mean(),
            "r_with_label": np.corrcoef(labelled[name], labelled["y"])[0, 1],
            "auroc_all": auroc(labelled[name], labelled["y"]),
        }
        for subset, part in labelled.groupby("subset"):
            row[f"auroc_{subset}"] = auroc(part[name], part["y"])
        rows.append(row)

    return {
        "feature_statistics": stats,
        "feature_statistics_by_subset": by_subset,
        "feature_label_association": pd.DataFrame(rows),
    }


def notebook_tables(notebook: dict) -> dict[str, pd.DataFrame]:
    tables = {}
    for start, names in B3_TABLES:
        cell = next(
            c for c in notebook["cells"]
            if c["cell_type"] == "code" and "".join(c["source"]).startswith(start)
        )
        found = [
            "".join(o["data"]["text/html"]) for o in cell.get("outputs", [])
            if "text/html" in o.get("data", {})
        ]
        if len(found) != len(names):
            raise ValueError(f"Expected {len(names)} tables after {start!r}, found {len(found)}.")
        tables.update({name: html_table(markup) for name, markup in zip(names, found)})
    return tables


def detector_comparison(b3: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every detector on one long table, one row per (detector, subset)."""
    columns = ["n", "frac_hallucinated", "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "aurac"]
    rows = []

    def add(block: str, kind: str, detector: str, metrics: pd.DataFrame):
        for row in metrics.itertuples(index=False):
            rows.append({
                "block": block, "kind": kind, "detector": detector, "subset": row.subset,
                **{c: getattr(row, c, np.nan) for c in columns},
            })

    b4 = pd.read_csv(DATA_DIR / "b4" / "metrics.csv")
    for score, name in B4_DETECTORS:
        kind = "sampling" if score != "naive_entropy" else "sampling (no clustering)"
        add("B4", kind, name, b4[b4["score"] == score])

    b5 = pd.read_csv(DATA_DIR / "b5" / "comparison_metrics.csv")
    for name in B5_DETECTORS:
        add("B5", "single-pass", name, b5[b5["score"] == name])

    b6 = pd.read_csv(DATA_DIR / "b6" / "metrics.csv")
    for score, name in B6_DETECTORS:
        add("B6", "single-pass", name, b6[b6["score"] == score])

    b8 = json.loads((DATA_DIR / "b8" / "metrics.json").read_text())
    for key, name in B8_DETECTORS:
        rows.append({
            "block": "B8", "kind": "single-pass", "detector": name, "subset": "all",
            "n": b8[key]["n"], "frac_hallucinated": b8[key]["positive_fraction"],
            "auroc": b8[key]["auroc"], "auprc": b8[key]["auprc"],
        })

    # B3 prints these as text tables: the CI and AURAC only for "all".
    overall = b3["b3_metrics"].set_index("method")
    by_subset = b3["b3_auroc_by_subset"].set_index("score")
    auprc = b3["b3_auprc_by_subset"].set_index("score")
    for method in by_subset.index:
        kind = overall.loc[method, "type"]
        for subset in by_subset.columns:
            row = {
                "block": "B3", "kind": kind, "detector": method, "subset": subset,
                "auroc": float(by_subset.loc[method, subset]),
                "auprc": float(auprc.loc[method, subset]),
            }
            if subset == "all":
                low, high = re.findall(r"[\d.]+", overall.loc[method, "95% CI"])
                row.update(
                    auroc_ci_low=float(low), auroc_ci_high=float(high),
                    aurac=float(overall.loc[method, "AURAC"]),
                )
            rows.append(row)

    return pd.DataFrame(rows)


def per_answer_scores(labels: pd.DataFrame) -> pd.DataFrame:
    frame = labels[["uid", "subset", "refusal", "hallucinated"]].copy()
    for block, columns in SCORE_COLUMNS.items():
        scores = pd.read_parquet(DATA_DIR / block / "scores.parquet", columns=["uid"] + columns)
        frame = frame.merge(scores, on="uid", how="left")

    # B8 predicts only its validation fold, so other rows are empty.
    b8 = pd.read_parquet(
        DATA_DIR / "b8" / "validation_predictions.parquet",
        columns=["uid", "logistic_regression_probability", "gradient_boosting_probability"],
    )
    return frame.merge(b8, on="uid", how="left").sort_values("uid", kind="stable")


# Figures
def copy_figures(out: Path) -> int:
    count = 0
    for block in ("b4", "b5", "b6"):
        for path in sorted((DATA_DIR / block / "figures").glob("*.png")):
            shutil.copy2(path, out / f"{block}_{path.name}")
            count += 1
    shutil.copy2(DATA_DIR / "b7" / "correlation_heatmap.png", out / "b7_correlation_heatmap.png")
    return count + 1


def notebook_figures(notebook: dict, out: Path) -> int:
    """Each figure is named after the notebook section it sits in."""
    section, count = "start", 0
    used: dict[str, int] = {}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        heading = re.match(r"##\s+(.+)", source)
        if cell["cell_type"] == "markdown" and heading:
            section = slug(heading.group(1))
        for output in cell.get("outputs", []):
            image = output.get("data", {}).get("image/png")
            if image is None:
                continue
            image = "".join(image) if isinstance(image, list) else image
            used[section] = used.get(section, 0) + 1
            suffix = "" if used[section] == 1 else f"_{used[section]}"
            (out / f"b3_{section}{suffix}.png").write_bytes(base64.b64decode(image))
            count += 1
    return count


# Checks
def run_checks(labels: pd.DataFrame, detectors: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(PASS / WARN / FAIL, check, detail). All results must be one run."""
    checks = []

    def check(ok: bool, name: str, detail: str, failure: str = "FAIL"):
        checks.append(("PASS" if ok else failure, name, detail))

    n_labelled = int(labels["hallucinated"].notna().sum())

    methods = sorted(labels["label_method"].unique())
    check(len(methods) == 1, "One label method", ", ".join(methods))

    overall = detectors[(detectors["subset"] == "all") & detectors["n"].notna()]
    wrong = overall[(overall["block"] != "B8") & (overall["n"] != n_labelled)]
    check(wrong.empty, "B4, B5, B6 use every labelled answer",
          f"{n_labelled:,} labelled" + ("" if wrong.empty else f"; differs: {sorted(set(wrong['detector']))}"))

    b9 = pd.read_csv(DATA_DIR / "b9" / "budget_sweep.csv")
    check(set(b9["n_labeled"]) == {n_labelled}, "B9 uses every labelled answer",
          f"n = {sorted(int(v) for v in set(b9['n_labeled']))}")

    split = json.loads((DATA_DIR / "b8" / "split_info.json").read_text())
    n_b8 = split["training_rows"] + split["validation_rows"]
    check(n_b8 == n_labelled, "B8 uses every labelled answer", f"{n_b8:,} rows")
    check(split.get("label_method") in methods, "B8 used the current labels", str(split.get("label_method")))

    shas = {
        block: json.loads((DATA_DIR / block / "config.json").read_text()).get("git_commit_sha")
        for block in ("b4", "b5", "b6")
    }
    check(len(set(shas.values())) == 1, "B4, B5, B6 ran on one commit",
          ", ".join(sorted({str(s)[:7] for s in shas.values()})))

    main = pd.read_parquet(MAIN_ANSWERS_PATH, columns=["subset", "prompt"])
    summaries = main.loc[main["subset"] == "summarization", "prompt"]
    has_document = summaries.str.contains("\n\n", regex=False).mean()
    check(has_document == 1, "Summarization prompts include the document", f"{has_document:.1%}")

    stale = [
        name for name in ("hidden_states.zip", "token_features.zip")
        if last_commit_time(DATA_DIR / "a3" / name) < last_commit_time(MAIN_ANSWERS_PATH)
    ]
    check(not stale, "A3 zips match the current answers",
          "older than the answers: " + ", ".join(stale) if stale else "up to date", "WARN")
    return checks


# README
def auroc_summary(detectors: pd.DataFrame) -> pd.DataFrame:
    table = detectors.pivot_table(index=["block", "detector"], columns="subset", values="auroc", sort=False)
    table = table.reindex(columns=SUBSETS).sort_values("all", ascending=False)
    return table.reset_index()


def write_readme(out: Path, tables: dict, detectors: pd.DataFrame, checks: list, counts: dict):
    b6 = tables["b6_threshold_summary"]
    b6 = b6[b6["score"] == "logprob_neg_mean"]
    b6 = b6[["rule", "threshold_mean", "recall_mean", "precision_mean", "flag_rate_mean",
             "accepted_accuracy_mean"]]

    sweep = tables["b9_budget_sweep"]
    sweep = sweep[sweep["subset"] == "all"]
    sweep = sweep[sweep["method"].isin(["semantic_entropy", "discrete_semantic_entropy", "naive_entropy"])]
    sweep = sweep.pivot(index="n_samples", columns="method", values="auroc").reset_index()
    sweep["n_samples"] = sweep["n_samples"].astype(int)

    features = tables["feature_label_association"]
    features = features.reindex(features["r_with_label"].abs().sort_values(ascending=False).index)
    features = features[["feature", "r_with_label", "auroc_all"]].head(6)

    importance = tables["b8_gradient_boosting_importance"].head(6)

    sections = [
        "# Results summary",
        "Generated by `scripts/collect_results.py`. Do not edit by hand: rerun the script.",
        f"Code commit: `{git_commit_sha()[:7]}`. Scope: the calibration split only. "
        "The test split has no generated answers yet.",
        "## Read this before quoting a number",
        "- Labels are provisional NLI labels, not verified ground truth. Every detector number below "
        "inherits their errors.\n"
        "- These are exploratory results on the calibration split, not held-out test performance.\n"
        "- AUROC below 0.5 means the score points the wrong way for that label.\n"
        "- Accuracy-probe layers were picked using these same labels, so those numbers are optimistic.\n"
        "- B8 results come from one validation fold, so they aren't directly comparable with "
        "the cross-fitted or full-data results.\n"
        "- Single-pass and sampling detectors differ in cost, which is not shown here.",
        "## Checks",
        markdown_table(pd.DataFrame(checks, columns=["result", "check", "detail"])),
        "## Answers and labels",
        markdown_table(tables["dataset_counts"]),
        "## Detectors: AUROC by subset",
        "Sorted by the `all` column. Full metrics are in `tables/detector_comparison.csv`.",
        markdown_table(auroc_summary(detectors)),
        "## Naive log-probability thresholds (B6)",
        "Mean over held-out folds, using the mean token log-probability.",
        markdown_table(b6),
        "## Sampling budget (B9): AUROC, all subsets",
        markdown_table(sweep),
        f"## Answers that reached the {MAX_NEW_TOKENS}-token limit",
        "Estimated by re-tokenizing the saved text, because the stop reason was not saved.",
        markdown_table(tables["token_limit"]),
        "## Strongest single features (B7)",
        markdown_table(features),
        "## Gradient boosting importance (B8)",
        markdown_table(importance),
        "## Files",
        f"- `tables/`: {counts['tables']} result files\n"
        f"- `figures/`: {counts['figures']} figures\n"
        "- `scores/per_answer_scores.parquet`: label and detector scores per answer, for Block C\n"
        "- B3 per-answer scores are not included, because the notebook did not save them",
    ]
    (out / "README.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    scores_dir = args.output_dir / "scores"
    for directory in (tables_dir, figures_dir, scores_dir):
        directory.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(LABELS_PATH)
    labels["uid"] = labels["uid"].astype(str)
    features = pd.read_parquet(DATA_DIR / "b7" / "feature_table.parquet")
    features["uid"] = features["uid"].astype(str)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    b3 = notebook_tables(notebook)
    detectors = detector_comparison(b3)

    tables = {
        "dataset_counts": dataset_counts(labels),
        "token_limit": token_limit(features, labels[["uid", "subset", "hallucinated"]]),
        "detector_comparison": detectors,
        **feature_tables(features, labels[["uid", "subset", "hallucinated"]]),
        **b3,
    }
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    for source, name in COPIED_TABLES:
        shutil.copy2(DATA_DIR / source, tables_dir / name)
    tables["b6_threshold_summary"] = pd.read_csv(DATA_DIR / "b6" / "threshold_summary.csv")
    tables["b9_budget_sweep"] = pd.read_csv(DATA_DIR / "b9" / "budget_sweep.csv")
    tables["b8_gradient_boosting_importance"] = pd.read_csv(
        DATA_DIR / "b8" / "gradient_boosting_importance.csv"
    )

    n_figures = copy_figures(figures_dir) + notebook_figures(notebook, figures_dir)

    per_answer_scores(labels).to_parquet(scores_dir / "per_answer_scores.parquet", index=False)

    checks = run_checks(labels, detectors)
    counts = {"tables": len(list(tables_dir.iterdir())), "figures": n_figures}
    write_readme(args.output_dir, tables, detectors, checks, counts)

    for result, name, detail in checks:
        print(f"[{result}] {name}: {detail}")
    size = sum(p.stat().st_size for p in args.output_dir.rglob("*") if p.is_file())
    print(f"[SUCCESS] {counts['tables']} tables, {n_figures} figures, "
          f"{size / 1e6:.1f} MB -> {args.output_dir}")
    if any(result == "FAIL" for result, _, _ in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
