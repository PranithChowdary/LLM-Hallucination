"""
Matplotlib figures shared by B3-B6 (static PNGs for the paper/notebook).

Style rules (kept consistent across every figure):
    - categorical colors in a FIXED order; a method keeps its color in
      every figure because colors are assigned by name, not by rank
    - one y-axis per panel, recessive grid, 2px lines, legend always
      present when a panel has more than one series
    - more than 8 series never share a panel: split into small multiples
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Validated categorical palette (light surface), fixed order.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9890"  # chance lines / reference marks


def setup(headless: bool = False):
    """Apply the shared style; returns pyplot. Scripts pass headless=True."""
    import matplotlib

    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 9.5,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    return plt


class ColorMap:
    """Assigns palette slots to names in first-seen order, then keeps them."""

    def __init__(self, names: list[str] | None = None):
        self._colors: dict[str, str] = {}
        for name in names or []:
            self[name]

    def __getitem__(self, name: str) -> str:
        if name not in self._colors:
            if len(self._colors) >= len(PALETTE):
                raise ValueError(
                    "More than 8 series in one color map; split into small multiples."
                )
            self._colors[name] = PALETTE[len(self._colors)]
        return self._colors[name]


def roc_pr_panels(
    axes,
    curves: dict[str, np.ndarray],
    labels: np.ndarray,
    colors: ColorMap,
    title_suffix: str = "",
):
    """ROC on axes[0], precision-recall on axes[1]."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

    from src.data.report import auroc

    labels = np.asarray(labels).astype(int)
    ax_roc, ax_pr = axes

    for name, scores in curves.items():
        s = np.asarray(scores, dtype=float)
        keep = np.isfinite(s)
        fpr, tpr, _ = roc_curve(labels[keep], s[keep])
        prec, rec, _ = precision_recall_curve(labels[keep], s[keep])
        a = auroc(s[keep], labels[keep])
        p = average_precision_score(labels[keep], s[keep])
        ax_roc.plot(fpr, tpr, color=colors[name], label=f"{name} ({a:.3f})")
        ax_pr.plot(rec, prec, color=colors[name], label=f"{name} ({p:.3f})")

    ax_roc.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--", label="chance (0.500)")
    base = labels.mean()
    ax_pr.axhline(base, color=NEUTRAL, lw=1, ls="--", label=f"chance ({base:.3f})")

    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"ROC{title_suffix}", xlim=(0, 1), ylim=(0, 1.01))
    ax_pr.set(xlabel="Recall (hallucinated)", ylabel="Precision",
              title=f"Precision-recall{title_suffix}", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.legend(title="AUROC", loc="best", title_fontsize=8.5)
    ax_pr.legend(title="AUPRC", loc="best", title_fontsize=8.5)


def rejection_panel(ax, curves: dict[str, np.ndarray], labels: np.ndarray, colors: ColorMap):
    """Accuracy of retained answers vs fraction retained (Farquhar et al.)."""
    from src.eval.metrics import aurac, rejection_accuracy_curve

    labels = np.asarray(labels).astype(int)
    for name, scores in curves.items():
        s = np.asarray(scores, dtype=float)
        keep = np.isfinite(s)
        retained, acc = rejection_accuracy_curve(s[keep], labels[keep])
        ax.plot(retained, acc, color=colors[name],
                label=f"{name} ({aurac(s[keep], labels[keep]):.3f})")
    ax.axhline(1 - labels.mean(), color=NEUTRAL, lw=1, ls="--", label="no rejection")
    ax.set(xlabel="Fraction of answers kept (most confident first)",
           ylabel="Accuracy of kept answers", title="Rejection-accuracy",
           xlim=(1, 0))
    ax.legend(title="AURAC", loc="best", title_fontsize=8.5)


def metric_bars(ax, table, metric: str = "auroc", colors: ColorMap | None = None,
                ci: tuple[str, str] | None = ("auroc_ci_low", "auroc_ci_high"),
                chance: float | None = 0.5):
    """
    Horizontal bars, one per row of `table` (needs columns 'score' and
    `metric`), sorted best-first, value printed at the bar end.
    """
    table = table.sort_values(metric, ascending=True)
    y = np.arange(len(table))
    color = [colors[n] if colors else PALETTE[0] for n in table["score"]]
    ax.barh(y, table[metric], color=color, height=0.62, edgecolor=SURFACE, linewidth=2)
    if ci and ci[0] in table:
        ax.errorbar(table[metric], y,
                    xerr=[table[metric] - table[ci[0]], table[ci[1]] - table[metric]],
                    fmt="none", ecolor=TEXT_MUTED, elinewidth=1, capsize=2)
    for yi, v in zip(y, table[metric]):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    if chance is not None:
        ax.axvline(chance, color=NEUTRAL, lw=1, ls="--")
    ax.set_yticks(y, table["score"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(metric.upper())


def save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"[FIGURE] {path}")
