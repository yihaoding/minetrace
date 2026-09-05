"""Compare the linear AUC-weighted expert tree vs. Dempster-Shafer belief
fusion (3 reliability modes), on identical fitted models and test sets.

Focus (Fork B): catching rare positives — atypical-signature deposits that a
single expert detects but the averaging fusion washes out. Beyond overall AUC,
we report the mean percentile-rank of *hard positives* (the positives the
linear tree scores worst); a higher percentile means the rare positive was
lifted up the ranking.

Scorers compared:
  tree      : ns.score from CompositeNode (linear AUC-weighted mean)
  bel-glob  : belief fusion, reliability = global AUC weight
  bel-local : belief fusion, reliability = per-expert local evidence strength
  bel-hybr  : belief fusion, reliability = sqrt(global * local)

Usage::

    /group/pmc050/yding/miniconda3/envs/geochem/bin/python scripts/eval_belief_compare.py
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

logging.basicConfig(level=logging.ERROR,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX
from domains.geochem.belief_fusion import BeliefFusionScorer
from scripts.eval_multimetal import (
    METAL_CONFIGS, SEED, N_TEST_POS, N_TEST_NEG,
    build_catalog, load_positives, load_spatially_matched_neg,
    load_random_background,
)


def _auc(scores, labels):
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None and np.isfinite(s)]
    y = [l for _, l in pairs]
    p = [s for s, _ in pairs]
    return float(roc_auc_score(y, p)) if len(set(y)) >= 2 else float("nan")


def _pctile_ranks(scores):
    """Percentile rank in [0,1] per element (1.0 = highest score). NaN-safe."""
    arr = np.array([s if (s is not None and np.isfinite(s)) else -np.inf for s in scores],
                   dtype=float)
    r = rankdata(arr, method="average")
    return (r - 1) / (len(r) - 1) if len(r) > 1 else np.zeros_like(r)


def score_all(agent, test_pos, neg):
    """Return per-scorer scores for the same test set."""
    all_test = test_pos + neg
    labels = np.array([1] * len(test_pos) + [0] * len(neg))

    ns_tree = agent.score_batch(all_test)              # registers samples + tree scores
    tree = [ns.score if ns is not None else None for ns in ns_tree]

    out = {"tree": tree}
    for mode in ("global", "local", "hybrid"):
        bf = BeliefFusionScorer(agent._tree, mode=mode)
        bf.calibrate(agent._fit_positives, agent._fit_background)
        ns = [bf.score_node(s) for s in all_test]
        out[mode] = [n.score if n is not None else None for n in ns]
    return out, labels


def main() -> None:
    catalog = build_catalog()
    frames = []
    for cfg in METAL_CONFIGS.values():
        df = pd.read_csv(cfg["sites_csv"])
        if "SITE_CODE" not in df.columns and "fid" in df.columns:
            df = df.rename(columns={"fid": "SITE_CODE"})
        df["SITE_CODE"] = df["SITE_CODE"].astype(str)
        frames.append(df[["X", "Y", "SITE_CODE"]].dropna())
    all_sites = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["X", "Y"])

    cols = ["tree", "global", "local", "hybrid"]
    auc_rows, hard_rows = [], []

    for metal, cfg in METAL_CONFIGS.items():
        print(f"\n{'='*72}\n  {metal}\n{'='*72}")
        all_pos = load_positives(cfg["sites_csv"], cfg["site_col"], cfg["confidence_map"])
        n_test = N_TEST_POS if len(all_pos) >= N_TEST_POS + 10 else max(10, len(all_pos) // 5)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(all_pos))
        test_pos = [all_pos[i] for i in idx[:n_test]]
        test_ids = {s.id for s in test_pos}
        all_pos_ids = {s.id for s in all_pos}

        tc = GeochemTargetConfig(
            target=metal, pathfinders=cfg["pathfinders"],
            ratio_features=cfg["ratio_features"], confidence_map=cfg["confidence_map"],
            sites_csv=cfg["sites_csv"],
        )
        agent = GeochemAgent(catalog=catalog, target_config=tc,
                             n_bg=300, bbox=WA_BBOX, seed=SEED)
        agent.plan(check_bbox=WA_BBOX)
        agent.setup(exclude_ids=test_ids)

        random_neg = load_random_background(agent, N_TEST_NEG, seed=SEED)
        matched_neg = load_spatially_matched_neg(test_pos, all_pos_ids, all_sites,
                                                 radius_km=50.0, seed=SEED)

        for neg_kind, neg in (("random", random_neg), ("matched", matched_neg)):
            scores, labels = score_all(agent, test_pos, neg)
            pos_mask = labels == 1

            aucs = {c: _auc(scores[c], labels) for c in cols}
            auc_rows.append((metal, neg_kind, aucs))

            # hard positives = bottom quartile of positives by the TREE ranking
            tree_pr = _pctile_ranks(scores["tree"])
            pos_pr_tree = tree_pr[pos_mask]
            thresh = np.quantile(pos_pr_tree, 0.25)
            hard = pos_mask.copy()
            hard[pos_mask] = pos_pr_tree <= thresh   # hardest 25% of positives

            hp = {c: float(np.mean(_pctile_ranks(scores[c])[hard])) for c in cols}
            hard_rows.append((metal, neg_kind, hp, int(hard.sum())))

            print(f"  [{neg_kind:<7}] AUC  " +
                  "  ".join(f"{c}={aucs[c]:.3f}" for c in cols))
            print(f"            hardP%ile({int(hard.sum())})  " +
                  "  ".join(f"{c}={hp[c]:.3f}" for c in cols))

    # ── summaries ───────────────────────────────────────────────────────────
    print(f"\n{'='*72}\n  AUC SUMMARY\n{'='*72}")
    print(f"  {'metal':<5} {'neg':<8} " + " ".join(f"{c:>8}" for c in cols))
    for metal, nk, a in auc_rows:
        print(f"  {metal:<5} {nk:<8} " + " ".join(f"{a[c]:>8.3f}" for c in cols))
    for c in cols[1:]:
        d = np.mean([a[c] - a["tree"] for *_, a in auc_rows])
        print(f"  mean Δ AUC ({c} - tree): {d:+.3f}")

    print(f"\n{'='*72}\n  HARD-POSITIVE percentile-rank (rare-positive recovery)\n{'='*72}")
    print(f"  {'metal':<5} {'neg':<8} " + " ".join(f"{c:>8}" for c in cols))
    for metal, nk, hp, n in hard_rows:
        print(f"  {metal:<5} {nk:<8} " + " ".join(f"{hp[c]:>8.3f}" for c in cols))
    for c in cols[1:]:
        d = np.mean([hp[c] - hp["tree"] for *_, hp, _ in hard_rows])
        print(f"  mean Δ hardP%ile ({c} - tree): {d:+.3f}")


if __name__ == "__main__":
    main()
