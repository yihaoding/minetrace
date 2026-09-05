"""Comprehensive belief-fusion comparison: 9 metals x 4 negative strategies.

Reuses all of eval_comprehensive.py's configs and negative samplers, but scores
every test set four ways on the identical fitted model:

  tree      : CompositeNode linear AUC-weighted mean (baseline)
  bel-glob  : belief fusion, reliability = global AUC weight
  bel-local : belief fusion, reliability = per-expert local evidence strength
  bel-hybr  : belief fusion, reliability = sqrt(global * local)

For each (metal, strategy, scorer) we report AUC and the mean percentile-rank of
the *hard positives* (positives the linear tree ranks worst) — the rare-positive
recovery metric (Fork B). Summaries are broken down BY STRATEGY so the
easy (random / far_random) vs hard (true_nonmine / spatial) split is visible.

Usage::

    /group/pmc050/yding/miniconda3/envs/geochem/bin/python scripts/eval_belief_comprehensive.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.ERROR)

from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX
from domains.geochem.belief_fusion import BeliefFusionScorer, BeliefRevisionScorer
from scripts.eval_comprehensive import (
    METAL_CONFIGS, ALL_METALS, SEED, MAX_TEST, N_NEG, SPATIAL_BOUNDARY, LAYERS,
    build_catalog, build_all_mine_tree, build_all_sites_combined,
    load_mine_deposit_samples, load_no_target_sites, normalise_sites_df,
    _forbid_xy_km, gen_random_negatives, gen_far_random_negatives,
    gen_region_negatives,
)
from domains.geochem.samples import GeochemSample

SCORERS = ["tree", "global", "local", "hybrid", "routed", "revision"]
# Non-private negative strategies only (true_nonmine uses the curated
# no_<el>_sites.csv set, excluded here per request).
STRATS = ["random", "far_random", "spatial"]


def _auc(scores, labels):
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None and np.isfinite(s)]
    y = [l for _, l in pairs]
    return float(roc_auc_score(y, [s for s, _ in pairs])) if len(set(y)) >= 2 else float("nan")


def _pctile(scores):
    arr = np.array([s if (s is not None and np.isfinite(s)) else -np.inf for s in scores],
                   dtype=float)
    r = rankdata(arr, method="average")
    return (r - 1) / (len(r) - 1) if len(r) > 1 else np.zeros_like(r)


def multi_eval(agent, test_pos, negatives):
    """Score one test set with all four scorers → {scorer: (auc, hardP%ile)}."""
    if not test_pos or not negatives:
        return None
    all_test = list(test_pos) + list(negatives)
    labels = np.array([1] * len(test_pos) + [0] * len(negatives))

    ns_tree = agent.score_batch(all_test)
    scores = {"tree": [n.score if n is not None else None for n in ns_tree]}
    for mode in ("global", "local", "hybrid", "routed"):
        bf = BeliefFusionScorer(agent._tree, mode=mode)
        bf.calibrate(agent._fit_positives, agent._fit_background)
        ns = [bf.score_node(s) for s in all_test]
        scores[mode] = [n.score if n is not None else None for n in ns]
    rev = BeliefRevisionScorer(agent._tree)
    ns_rev = [rev.score_node(s) for s in all_test]
    scores["revision"] = [n.score if n is not None else None for n in ns_rev]

    pos_mask = labels == 1
    tree_pr = _pctile(scores["tree"])
    pos_pr = tree_pr[pos_mask]
    if len(pos_pr) < 4:
        hard = pos_mask
    else:
        thr = np.quantile(pos_pr, 0.25)
        hard = pos_mask.copy()
        hard[pos_mask] = pos_pr <= thr

    out = {}
    for c in SCORERS:
        out[c] = (_auc(scores[c], labels), float(np.mean(_pctile(scores[c])[hard])))
    return out


def main() -> None:
    catalog = build_catalog()
    mine_tree = build_all_mine_tree()
    all_sites = build_all_sites_combined()
    all_mine_ids: set[str] = set()
    for cfg in METAL_CONFIGS.values():
        df = normalise_sites_df(pd.read_csv(cfg["sites_csv"]))
        ids = df[df[cfg["site_col"]].fillna("").str.contains("Mine|Deposit")]["SITE_CODE"]
        all_mine_ids.update(ids.astype(str))

    grid: dict = {}   # grid[metal][strat] = {scorer: (auc, hardP)}

    for metal, cfg in METAL_CONFIGS.items():
        print(f"\n{'='*72}\n  {metal}\n{'='*72}")
        grid[metal] = {}
        all_pos = load_mine_deposit_samples(cfg["sites_csv"], cfg["site_col"],
                                            cfg["confidence_map"])
        all_pos_ids = {s.id for s in all_pos}
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(all_pos))
        n_test = min(MAX_TEST, max(0, len(all_pos) - 10))
        test_pos = [all_pos[i] for i in perm[:n_test]]
        test_ids = {s.id for s in test_pos}
        forbid = _forbid_xy_km(test_pos)

        tc = GeochemTargetConfig(target=metal, pathfinders=cfg["pathfinders"],
                                 ratio_features=cfg["ratio_features"],
                                 confidence_map=cfg["confidence_map"],
                                 sites_csv=cfg["sites_csv"], layers=LAYERS)
        agent = GeochemAgent(catalog=catalog, target_config=tc, n_bg=300,
                             bbox=WA_BBOX, seed=SEED)
        agent.plan(check_bbox=WA_BBOX)
        agent.setup(exclude_ids=test_ids)

        # random-split agent, non-private negatives only
        neg_sets = {
            "random": gen_random_negatives(agent, N_NEG, seed=SEED + 1, forbid_xy_km=forbid),
            "far_random": gen_far_random_negatives(agent, mine_tree, N_NEG, seed=SEED + 2,
                                                   forbid_xy_km=forbid),
        }

        for strat in ("random", "far_random"):
            res = multi_eval(agent, test_pos, neg_sets[strat])
            if res:
                grid[metal][strat] = res
                print(f"  [{strat:<13}] " +
                      " ".join(f"{c}:A={res[c][0]:.3f}/H={res[c][1]:.3f}" for c in SCORERS))

        # 4. spatial holdout — south train, north test
        df_full = normalise_sites_df(pd.read_csv(cfg["sites_csv"]))
        mine_df = df_full[df_full[cfg["site_col"]].fillna("").str.contains("Mine|Deposit")]
        north = [GeochemSample(site_code=str(r["SITE_CODE"]), x=float(r["X"]), y=float(r["Y"]))
                 for _, r in mine_df[mine_df.Y >= SPATIAL_BOUNDARY].iterrows()]
        south = [GeochemSample(site_code=str(r["SITE_CODE"]), x=float(r["X"]), y=float(r["Y"]))
                 for _, r in mine_df[mine_df.Y < SPATIAL_BOUNDARY].iterrows()]
        n_nt = min(MAX_TEST, len(north))
        rng4 = np.random.default_rng(SEED + 4)
        test_north = ([north[i] for i in rng4.choice(len(north), n_nt, replace=False)]
                      if len(north) > n_nt else list(north))
        if len(south) >= 10 and len(test_north) >= 5:
            agent_sp = GeochemAgent(catalog=catalog, target_config=tc, n_bg=300,
                                    bbox=WA_BBOX, seed=SEED + 10)
            agent_sp.plan(check_bbox=WA_BBOX)
            agent_sp.setup(exclude_ids={s.id for s in north})
            sp_neg = gen_region_negatives(all_sites.Y >= SPATIAL_BOUNDARY, all_sites,
                                          all_mine_ids, N_NEG, seed=SEED + 5,
                                          forbid_xy_km=_forbid_xy_km(test_north))
            res = multi_eval(agent_sp, test_north, sp_neg)
            if res:
                grid[metal]["spatial"] = res
                print(f"  [{'spatial':<13}] " +
                      " ".join(f"{c}:A={res[c][0]:.3f}/H={res[c][1]:.3f}" for c in SCORERS))

    # ── summaries by strategy ───────────────────────────────────────────────
    def mean_delta(strat, idx, scorer):
        vals = [grid[m][strat][scorer][idx] - grid[m][strat]["tree"][idx]
                for m in ALL_METALS if strat in grid.get(m, {})]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n{'='*72}\n  AUC: mean Δ vs tree, by strategy\n{'='*72}")
    print(f"  {'strategy':<14} " + " ".join(f"{c:>9}" for c in SCORERS[1:]))
    for strat in STRATS:
        print(f"  {strat:<14} " + " ".join(f"{mean_delta(strat,0,c):>+9.3f}" for c in SCORERS[1:]))

    print(f"\n{'='*72}\n  HARD-POSITIVE %ile: mean Δ vs tree, by strategy (rare-positive recovery)\n{'='*72}")
    print(f"  {'strategy':<14} " + " ".join(f"{c:>9}" for c in SCORERS[1:]))
    for strat in STRATS:
        print(f"  {strat:<14} " + " ".join(f"{mean_delta(strat,1,c):>+9.3f}" for c in SCORERS[1:]))

    # absolute AUC grand means per scorer
    print(f"\n{'='*72}\n  Absolute mean AUC across full 9x4 grid\n{'='*72}")
    for c in SCORERS:
        vals = [grid[m][s][c][0] for m in ALL_METALS for s in STRATS
                if s in grid.get(m, {})]
        print(f"  {c:<8} mean AUC = {np.nanmean(vals):.3f}")


if __name__ == "__main__":
    main()
