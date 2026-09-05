"""A3: training-sample learning curve.

For each of 4 representative elements (Cu/Au/Co/REE), sweep n_train ∈ {10, 30, 100,
300, all} × 5 seeds, evaluate against the True Non-Mine negative set.

Output:
    reports/learning_curve.json
    reports/figures/fig_learning_curve.pdf (+.png)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

from core.catalog import DataCatalog, SourceSpec  # noqa: E402
from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX  # noqa: E402
from domains.geochem.samples import GeochemSample  # noqa: E402
from scripts.eval_comprehensive import (
    METAL_CONFIGS, build_catalog, load_mine_deposit_samples, load_no_target_sites,
    _forbid_xy_km, BUFFER_KM, KM_PER_DEG,
)

REPORTS = ROOT / "reports"
OUT_FIG = REPORTS / "figures"; OUT_FIG.mkdir(exist_ok=True, parents=True)

ELEMENTS = ["Cu", "Au", "Co", "REE"]
TRAIN_SIZES = [10, 30, 100, 300, None]   # None = all available
SEEDS = [42, 43, 44, 45, 46]
N_TEST = 30
N_NEG  = 200


def eval_one(metal: str, n_train: int | None, seed: int, catalog: DataCatalog,
             all_pos: list, no_csv: str) -> float | None:
    cfg = METAL_CONFIGS[metal]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_pos))
    test_pos = [all_pos[i] for i in perm[:N_TEST]]
    train_pool = [all_pos[i] for i in perm[N_TEST:]]
    actual_n = len(train_pool) if n_train is None else min(n_train, len(train_pool))
    if actual_n < 10:
        return None
    train_pos = train_pool[:actual_n]

    test_ids = {s.id for s in test_pos}
    # excluded ids = everyone except chosen training (plus test)
    keep_train_ids = {s.id for s in train_pos}
    all_pos_ids = {s.id for s in all_pos}
    exclude_ids = (all_pos_ids - keep_train_ids) | test_ids  # withhold all non-train, all test

    tc = GeochemTargetConfig(
        target=metal,
        pathfinders=cfg["pathfinders"],
        ratio_features=cfg["ratio_features"],
        confidence_map=cfg["confidence_map"],
        sites_csv=cfg["sites_csv"],
    )
    agent = GeochemAgent(catalog=catalog, target_config=tc,
                          n_bg=300, bbox=WA_BBOX, seed=seed)
    agent.plan(check_bbox=WA_BBOX)
    agent.setup(exclude_ids=exclude_ids)

    forbid_xy = _forbid_xy_km(test_pos)
    nt_neg = load_no_target_sites(no_csv, all_pos_ids, forbid_xy_km=forbid_xy)
    rng2 = np.random.default_rng(seed + 1000)
    if len(nt_neg) > N_NEG:
        idxs = rng2.choice(len(nt_neg), N_NEG, replace=False)
        nt_neg = [nt_neg[i] for i in idxs]

    all_samples = test_pos + nt_neg
    labels = [1]*len(test_pos) + [0]*len(nt_neg)
    ns_list = agent.score_batch(all_samples)
    pairs = [(ns.score, l) for ns, l in zip(ns_list, labels) if ns is not None]
    if len(pairs) < 10:
        return None
    y_pred = [p[0] for p in pairs]; y_true = [p[1] for p in pairs]
    return float(roc_auc_score(y_true, y_pred))


def main() -> None:
    catalog = build_catalog()
    results = {el: {} for el in ELEMENTS}

    for metal in ELEMENTS:
        cfg = METAL_CONFIGS[metal]
        all_pos = load_mine_deposit_samples(cfg["sites_csv"], cfg["site_col"],
                                             cfg["confidence_map"])
        max_train = len(all_pos) - N_TEST
        print(f"\n{metal}: total M+D = {len(all_pos)},  test={N_TEST},  max_train={max_train}")
        for n in TRAIN_SIZES:
            actual = max_train if n is None else min(n, max_train)
            if actual < 10:
                results[metal][actual] = []
                continue
            aucs = []
            for s in SEEDS:
                try:
                    auc = eval_one(metal, n, s, catalog, all_pos, cfg["no_target_csv"])
                    if auc is not None:
                        aucs.append(auc)
                        print(f"  n_train={actual:4d}  seed={s}  AUC={auc:.3f}")
                except Exception as exc:
                    print(f"  n_train={actual} seed={s} FAILED: {exc}")
            results[metal][actual] = aucs

    with (REPORTS / "learning_curve.json").open("w") as fh:
        json.dump(results, fh, indent=2)

    # Plot
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8,"pdf.fonttype":42,"ps.fonttype":42,
        "savefig.bbox":"tight","savefig.dpi":300,
    })
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    colors = {"Cu":"#3498db","Au":"#f1c40f","Co":"#27ae60","REE":"#8e44ad"}
    for metal in ELEMENTS:
        ns = sorted(results[metal].keys())
        means = [np.mean(results[metal][n]) if results[metal][n] else np.nan for n in ns]
        stds  = [np.std(results[metal][n])  if results[metal][n] else 0       for n in ns]
        ax.errorbar(ns, means, yerr=stds, marker="o", lw=1.4, ms=4.5,
                     color=colors[metal], label=metal, capsize=3)
    ax.axhline(0.5, color="#888", ls="--", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Training positives (n_train)")
    ax.set_ylabel("AUC — True Non-Mine")
    ax.set_title("Learning curve — 5 seeds, error bars = SD",
                  fontweight="bold", fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.2, lw=0.4)
    for ext in (".pdf", ".png"):
        fig.savefig(OUT_FIG / f"fig_learning_curve{ext}")
    plt.close(fig)
    print(f"\n  → {OUT_FIG/'fig_learning_curve.pdf'}")


if __name__ == "__main__":
    main()
