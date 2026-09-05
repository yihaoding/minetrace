"""Generate A1 (ROC grid) + A4 (score distribution) figures from eval_scores.json.

Inputs:
    reports/eval_scores.json         # raw per-sample (score, label) per metal × strategy

Outputs:
    reports/figures/fig_roc_grid.pdf       (also .png)
    reports/figures/fig_score_dist.pdf     (also .png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path("/group/pmc050/yding/gad_reasoning")
SCORES = ROOT / "reports" / "eval_scores.json"
OUT    = ROOT / "reports" / "figures"
OUT.mkdir(exist_ok=True, parents=True)

STRAT_LABELS = {
    "random":      "Random",
    "far_random":  "Far Random (>50 km)",
    "true_nonmine":"True Non-Mine",
    "spatial":     "Spatial (S→N)",
}
STRAT_COLORS = {
    "random":      "#3498db",
    "far_random":  "#27ae60",
    "true_nonmine":"#e67e22",
    "spatial":     "#8e44ad",
}

# Paper-style defaults — sans-serif, vector-friendly
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "DejaVu Sans"],
    "font.size":        8,
    "axes.labelsize":   8,
    "axes.titlesize":   9,
    "legend.fontsize":  7,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "pdf.fonttype":     42,   # embed as Type-1/TrueType for journal submission
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
    "savefig.dpi":      300,
})


def _save(fig, stem: str) -> None:
    for ext in (".pdf", ".png"):
        fig.savefig(OUT / f"{stem}{ext}")
    plt.close(fig)
    print(f"  → {OUT / (stem + '.pdf')}")


def plot_roc_grid(data: dict) -> None:
    metals = list(data.keys())
    n = len(metals)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.6 * rows), squeeze=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.22, top=0.92, bottom=0.10)

    handles_for_legend, labels_for_legend = [], []
    for i, metal in enumerate(metals):
        ax = axes[i // cols][i % cols]
        ax.plot([0, 1], [0, 1], color="#bbb", lw=0.7, ls="--", zorder=0)
        for strat, r in data[metal].items():
            if r is None:
                continue
            y_pred = np.array(r["y_pred"]); y_true = np.array(r["y_true"])
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            (line,) = ax.plot(fpr, tpr,
                              color=STRAT_COLORS[strat], lw=1.3,
                              label=STRAT_LABELS[strat])
            if i == 0:
                handles_for_legend.append(line)
                labels_for_legend.append(STRAT_LABELS[strat])
        # Per-panel AUC text in lower-right
        auc_lines = []
        for strat in ["random","far_random","true_nonmine","spatial"]:
            r = data[metal].get(strat)
            if r:
                auc_lines.append(f"{STRAT_LABELS[strat][:3]:>3s} {r['auc']:.3f}")
        ax.text(0.97, 0.03, "\n".join(auc_lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5,
                family="monospace", color="#222",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

        ax.set_title(metal, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if i % cols == 0:    ax.set_ylabel("TPR")
        if i // cols == rows - 1:  ax.set_xlabel("FPR")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.15, lw=0.4)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.legend(handles_for_legend, labels_for_legend,
                loc="lower center", ncol=4, frameon=False,
                fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("ROC curves — 9 metals × 4 negative-sample strategies",
                 fontsize=10, fontweight="bold")
    _save(fig, "fig_roc_grid")


def plot_score_dist(data: dict, strat: str = "true_nonmine") -> None:
    """Score histograms (positives vs negatives) under the strictest strategy."""
    metals = list(data.keys())
    n = len(metals)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.6 * rows), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.30, top=0.92, bottom=0.10)
    bins = np.linspace(0, 1, 26)

    pos_handle = neg_handle = None
    for i, metal in enumerate(metals):
        ax = axes[i // cols][i % cols]
        r = data[metal].get(strat)
        if r is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="#888")
            ax.set_title(metal)
            continue
        y_pred = np.array(r["y_pred"]); y_true = np.array(r["y_true"])
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]
        h_neg = ax.hist(neg, bins=bins, color="#bdc3c7", alpha=0.85,
                          edgecolor="white", lw=0.3)
        h_pos = ax.hist(pos, bins=bins, color="#c0392b", alpha=0.75,
                          edgecolor="white", lw=0.3)
        if i == 0:
            neg_handle = h_neg[2][0]; pos_handle = h_pos[2][0]
        ax.set_title(f"{metal}  AUC={r['auc']:.3f}", fontweight="bold")
        ax.set_xlim(0, 1)
        if i % cols == 0:   ax.set_ylabel("count")
        if i // cols == rows - 1: ax.set_xlabel("score")
        ax.grid(alpha=0.15, lw=0.4)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    if pos_handle is not None:
        fig.legend([neg_handle, pos_handle],
                    ["non-mine (n=200)", "mine/deposit (n=30)"],
                    loc="lower center", ncol=2, frameon=False, fontsize=8,
                    bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Score distributions — strategy: {STRAT_LABELS[strat]}",
                 fontsize=10, fontweight="bold")
    _save(fig, "fig_score_dist")


def main() -> None:
    if not SCORES.exists():
        raise FileNotFoundError(f"{SCORES} not found — run eval_comprehensive.py first")
    with SCORES.open() as fh:
        data = json.load(fh)
    plot_roc_grid(data)
    plot_score_dist(data, strat="true_nonmine")
    # Also dump score dist for Random strategy (for supplementary)
    plot_score_dist_random_too = False  # toggle if needed
    print("Done.")


if __name__ == "__main__":
    main()
