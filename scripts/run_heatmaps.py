"""A5: WA prospectivity heatmap generator (5 km grid).

For each target metal, fit a fresh agent (no held-out test), then score every
grid cell across WA bbox. Overlay known Mine/Deposit points in red.

Usage:
    conda run -n geochem python scripts/run_heatmaps.py [METAL ...]

If no metals are given, runs Cu, Ni, Au, REE in sequence.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX  # noqa: E402
from domains.geochem.samples import GeochemSample  # noqa: E402
from scripts.eval_comprehensive import (
    METAL_CONFIGS, build_catalog, load_mine_deposit_samples,
)

REPORTS = ROOT / "reports"
OUT_FIG = REPORTS / "figures"; OUT_FIG.mkdir(exist_ok=True, parents=True)

GRID_KM = 5.0  # cell size
# WA bbox: (x_min, x_max, y_min, y_max) in degrees
X_MIN, X_MAX, Y_MIN, Y_MAX = WA_BBOX
DEG_PER_KM_LAT = 1.0 / 110.0
DEG_PER_KM_LON = 1.0 / 95.0   # rough, mid-latitude WA


def make_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dy = GRID_KM * DEG_PER_KM_LAT
    dx = GRID_KM * DEG_PER_KM_LON
    xs = np.arange(X_MIN, X_MAX + dx, dx)
    ys = np.arange(Y_MIN, Y_MAX + dy, dy)
    XX, YY = np.meshgrid(xs, ys)
    return XX, YY, xs, ys


def score_grid(metal: str, catalog, batch_size: int = 2000) -> dict:
    cfg = METAL_CONFIGS[metal]
    all_pos = load_mine_deposit_samples(cfg["sites_csv"], cfg["site_col"], cfg["confidence_map"])
    print(f"  {metal}: total positives = {len(all_pos)}")

    tc = GeochemTargetConfig(
        target=metal,
        pathfinders=cfg["pathfinders"],
        ratio_features=cfg["ratio_features"],
        confidence_map=cfg["confidence_map"],
        sites_csv=cfg["sites_csv"],
    )
    agent = GeochemAgent(catalog=catalog, target_config=tc,
                          n_bg=400, bbox=WA_BBOX, seed=42)
    agent.plan(check_bbox=WA_BBOX)
    agent.setup(exclude_ids=None)
    print(f"  {metal}: experts={len(agent.active_experts())}, scoring grid …")

    XX, YY, xs, ys = make_grid()
    H, W = XX.shape
    flat_x = XX.ravel(); flat_y = YY.ravel()
    scores = np.full(flat_x.shape, np.nan)

    for start in range(0, len(flat_x), batch_size):
        end = min(start + batch_size, len(flat_x))
        batch = [GeochemSample(site_code=f"G{i:07d}", x=float(flat_x[i]), y=float(flat_y[i]))
                 for i in range(start, end)]
        ns = agent.score_batch(batch)
        for j, n in enumerate(ns):
            if n is not None:
                scores[start + j] = n.score
        print(f"    {end}/{len(flat_x)}  ({100*end/len(flat_x):.0f}%)")

    grid = scores.reshape(H, W)

    # also load positives for overlay
    pos_x = [s.x for s in all_pos]; pos_y = [s.y for s in all_pos]

    return {
        "metal": metal,
        "grid":  grid,
        "xs":    xs,
        "ys":    ys,
        "pos_x": pos_x,
        "pos_y": pos_y,
    }


def plot_heatmap(data: dict) -> None:
    metal = data["metal"]
    grid  = data["grid"]
    xs    = data["xs"]; ys = data["ys"]
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8, "pdf.fonttype":42, "ps.fonttype":42,
        "savefig.bbox":"tight","savefig.dpi":300,
    })
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    extent = (xs.min(), xs.max(), ys.min(), ys.max())
    im = ax.imshow(grid, origin="lower", extent=extent, cmap="inferno",
                    vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.scatter(data["pos_x"], data["pos_y"], s=4, c="#00ffff", edgecolors="black",
                linewidths=0.2, label=f"{metal} Mine/Deposit (n={len(data['pos_x'])})")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"{metal} prospectivity — 5 km grid", fontweight="bold", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("score")
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.85)
    for ext in (".pdf", ".png"):
        fig.savefig(OUT_FIG / f"fig_heatmap_{metal}{ext}")
    plt.close(fig)
    print(f"  → {OUT_FIG/f'fig_heatmap_{metal}.pdf'}")


def main() -> None:
    metals = sys.argv[1:] if len(sys.argv) > 1 else ["Cu", "Ni", "Au", "REE"]
    catalog = build_catalog()
    for metal in metals:
        print(f"\n── {metal} ──")
        data = score_grid(metal, catalog)
        # Save raw grid (npz) for re-plotting / further analysis
        np.savez_compressed(REPORTS / f"heatmap_{metal}.npz",
                             grid=data["grid"], xs=data["xs"], ys=data["ys"],
                             pos_x=data["pos_x"], pos_y=data["pos_y"])
        plot_heatmap(data)


if __name__ == "__main__":
    main()
