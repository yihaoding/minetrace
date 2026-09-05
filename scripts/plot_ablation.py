"""A2: aggregate ablation JSONs into a heatmap + markdown table.

Reads:
    reports/eval_comprehensive_summary.json                 (full = baseline)
    reports/eval_comprehensive_summary_<tag>.json           (each ablation)

Writes:
    reports/figures/fig_ablation_heatmap.pdf  (+.png)
    reports/ablation_table.md                 (per-strategy markdown tables)
    reports/ablation_summary.json             (consolidated)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/group/pmc050/yding/gad_reasoning")
REPORTS = ROOT / "reports"
OUT = REPORTS / "figures"
OUT.mkdir(exist_ok=True, parents=True)

# Display order — most informative first
CONFIGS = [
    ("full",                "All 3 layers"),
    ("geochem_geophys",     "Geochem + Geophys"),
    ("geochem_structure",   "Geochem + Structure"),
    ("geochem",             "Geochem only"),
    ("geophys",             "Geophys only"),
    ("structure",           "Structure only"),
]

STRATS = ["random", "far_random", "true_nonmine", "spatial"]
STRAT_LABEL = {"random":"Random", "far_random":"Far(>50km)",
               "true_nonmine":"TrueNonMine", "spatial":"Spatial"}

plt.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
    "font.size":8, "pdf.fonttype":42, "ps.fonttype":42,
    "savefig.bbox":"tight", "savefig.dpi":300,
})


def _load(tag: str) -> dict | None:
    name = "eval_comprehensive_summary.json" if tag == "full" \
        else f"eval_comprehensive_summary_{tag}.json"
    p = REPORTS / name
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def main() -> None:
    loaded = {tag: _load(tag) for tag, _ in CONFIGS}
    missing = [t for t, d in loaded.items() if d is None]
    if missing:
        print(f"WARNING: missing JSONs for {missing}; skipping those rows")

    # Use whichever metals appear in the baseline (assumed present)
    base = loaded.get("full")
    if base is None:
        raise RuntimeError("Baseline (full) JSON not found — cannot proceed")
    metals = base["metals"]

    # Build heatmap: rows = configs, cols = (metal × strategy) — too dense.
    # Better: one heatmap per strategy, rows=configs cols=metals.
    fig, axes = plt.subplots(1, len(STRATS), figsize=(2.0 * len(STRATS) + 0.5, 2.6),
                              squeeze=False)
    axes = axes[0]
    for ai, strat in enumerate(STRATS):
        ax = axes[ai]
        mat = np.full((len(CONFIGS), len(metals)), np.nan)
        for i, (tag, _) in enumerate(CONFIGS):
            d = loaded[tag]
            if d is None: continue
            for j, m in enumerate(metals):
                r = d["results"].get(m, {}).get(strat)
                if r: mat[i, j] = r["auc"]
        im = ax.imshow(mat, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(metals))); ax.set_xticklabels(metals, rotation=0, fontsize=7)
        ax.set_yticks(range(len(CONFIGS)))
        if ai == 0:
            ax.set_yticklabels([label for _, label in CONFIGS], fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_title(STRAT_LABEL[strat], fontsize=8.5, fontweight="bold")
        for i in range(len(CONFIGS)):
            for j in range(len(metals)):
                v = mat[i, j]
                if np.isnan(v): continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                         color="white" if v < 0.55 else "black")
        ax.set_xticks(np.arange(-.5, len(metals)), minor=True)
        ax.set_yticks(np.arange(-.5, len(CONFIGS)), minor=True)
        ax.grid(which="minor", color="white", lw=0.6)
        ax.tick_params(which="minor", length=0)

    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="AUC")
    fig.suptitle("Expert-layer ablation — AUC by metal × negative strategy",
                 fontsize=10, fontweight="bold", y=1.02)
    for ext in (".pdf", ".png"):
        fig.savefig(OUT / f"fig_ablation_heatmap{ext}")
    plt.close(fig)
    print(f"  → {OUT/'fig_ablation_heatmap.pdf'}")

    # ── Markdown table ───────────────────────────────────────────────────
    md_lines = ["# Expert-layer ablation\n",
                f"Baseline = all 3 layers (geochem + geophys + structure). "
                f"AUC values per metal × strategy. n_test=30 per metal.\n"]
    for strat in STRATS:
        md_lines.append(f"\n## {STRAT_LABEL[strat]}\n")
        md_lines.append("| Config | " + " | ".join(metals) + " |")
        md_lines.append("|---|" + "|".join(["---:"] * len(metals)) + "|")
        full_row = [None] * len(metals)
        for i, (tag, label) in enumerate(CONFIGS):
            d = loaded[tag]
            if d is None: continue
            row = []
            for j, m in enumerate(metals):
                r = d["results"].get(m, {}).get(strat)
                v = r["auc"] if r else None
                if tag == "full" and v is not None:
                    full_row[j] = v
                if v is None:
                    cell = "—"
                elif tag == "full":
                    cell = f"**{v:.3f}**"
                else:
                    base_v = full_row[j]
                    if base_v is not None:
                        delta = v - base_v
                        cell = f"{v:.3f} ({delta:+.3f})"
                    else:
                        cell = f"{v:.3f}"
                row.append(cell)
            md_lines.append(f"| {label} | " + " | ".join(row) + " |")
    md_path = REPORTS / "ablation_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"  → {md_path}")

    # ── Consolidated JSON ────────────────────────────────────────────────
    out_json = REPORTS / "ablation_summary.json"
    with out_json.open("w") as fh:
        json.dump({
            "metals":  metals,
            "strats":  STRATS,
            "configs": [{"tag": t, "label": l} for t, l in CONFIGS],
            "auc": {
                t: ({m: {s: (d["results"][m][s]["auc"] if d["results"][m][s] else None)
                         for s in STRATS} for m in metals}
                    if d else None)
                for t, _ in CONFIGS for d in [loaded[t]]
            },
        }, fh, indent=2)
    print(f"  → {out_json}")


if __name__ == "__main__":
    main()
