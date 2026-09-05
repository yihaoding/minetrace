"""Does a model's self-reported reliability actually predict its errors?

The expert tree emits `confidence` (the fraction of the fitted expert weight
that was usable at this point) alongside every score. The baselines emit no
such thing, but each has an equivalent self-reported signal derived from the
prediction itself — the margin |p - 0.5|, plus tree-vote variance for the
random forest. This script asks the same question of all of them:

    if you keep only the points a model says it is sure about,
    does its ranking get better?

Coverage variation is generated the way deployment generates it: by withholding
data sources from the test points (0-6 of 7), then pooling every (point,
coverage) pair. Metrics per model:

  · selective AUC at 100 / 75 / 50 / 25 % coverage, ranked by its own signal
  · Spearman(signal, -|y - p|)  — does the signal track per-point error?

Note the asymmetry worth reporting: the expert tree's confidence is independent
of the score it accompanies; a margin is not — it is a transform of the score.

Usage::  python scripts/eval_confidence.py [--metals Cu Au Ni W]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".cache" / "pylibs"))
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from core.data_source import DataSourceRegistry
from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX
from scripts.eval_baselines import (
    N_TRAIN_NEG, align, expert_feature_keys, feature_frame, select_columns,
)
from scripts.eval_comprehensive import (
    METAL_CONFIGS, N_NEG, SEED, MAX_TEST, LAYERS, _forbid_xy_km, build_catalog,
    gen_random_negatives, load_mine_deposit_samples, load_no_target_sites,
)
from scripts.eval_degradation import ml_with_missing
from scripts.eval_explanation_quality import ML_MODELS, fit_models, predict

COVERAGES = (1.0, 0.75, 0.5, 0.25)
N_REP = 3
MODELS = ["expert_tree"] + ML_MODELS


def sel_auc(y, p, sig, cov):
    """AUC over the `cov` fraction of points the model is most sure about."""
    y, p, sig = np.asarray(y), np.asarray(p, float), np.asarray(sig, float)
    ok = np.isfinite(p) & np.isfinite(sig)
    y, p, sig = y[ok], p[ok], sig[ok]
    k = max(int(round(len(y) * cov)), 10)
    keep = np.argsort(-sig)[:k]
    ys, ps = y[keep], p[keep]
    if len(set(ys)) < 2:
        return None
    return float(roc_auc_score(ys, ps))


def run_metal(metal, cfg, catalog):
    all_pos = load_mine_deposit_samples(cfg["sites_csv"], cfg["site_col"], cfg["confidence_map"])
    all_pos_ids = {s.id for s in all_pos}
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_pos))
    n_test = min(MAX_TEST, max(0, len(all_pos) - 10))
    test_pos = [all_pos[i] for i in perm[:n_test]]
    test_ids = {s.id for s in test_pos}
    train_pos = [s for s in all_pos if s.id not in test_ids]
    forbid = _forbid_xy_km(test_pos)

    tc = GeochemTargetConfig(target=metal, pathfinders=cfg["pathfinders"],
                             ratio_features=cfg["ratio_features"],
                             confidence_map=cfg["confidence_map"],
                             sites_csv=cfg["sites_csv"], layers=LAYERS)
    agent = GeochemAgent(catalog=catalog, target_config=tc, n_bg=300, bbox=WA_BBOX, seed=SEED)
    agent.plan(check_bbox=WA_BBOX)
    agent.setup(exclude_ids=test_ids)

    train_neg = gen_random_negatives(agent, N_TRAIN_NEG, seed=SEED + 100, forbid_xy_km=forbid)
    neg = load_no_target_sites(cfg["no_target_csv"], all_pos_ids, forbid_xy_km=forbid)
    r3 = np.random.default_rng(SEED + 3)
    if len(neg) > N_NEG:
        neg = [neg[i] for i in r3.choice(len(neg), N_NEG, replace=False)]

    train, test = train_pos + train_neg, test_pos + neg
    y0 = np.array([1] * len(test_pos) + [0] * len(neg))
    ytr = np.array([1] * len(train_pos) + [0] * len(train_neg))
    Ftr, Fte = feature_frame(agent, train), feature_frame(agent, test)
    Xtr_all, Xte_all, all_cols = align(Ftr, Fte)
    keep = select_columns(all_cols, "expert_matched",
                          expert_feature_keys(metal, cfg["pathfinders"], cfg["ratio_features"]))
    Xtr, Xte, cols = Xtr_all[:, keep], Xte_all[:, keep], [all_cols[i] for i in keep]
    med = np.median(Xtr, axis=0)
    models = fit_models(Xtr, ytr, SEED)
    sources = list(agent.active_sources())

    pool = {m: {"y": [], "p": [], "sig": []} for m in MODELS}
    pool["random_forest_var"] = {"y": [], "p": [], "sig": []}

    for k in range(len(sources)):
        reps = [[]] if k == 0 else [
            list(np.random.default_rng(SEED + 7 * r).choice(sources, k, replace=False))
            for r in range(N_REP)]
        for drop in reps:
            saved = []
            for name in drop:
                try:
                    src = DataSourceRegistry.get(name)
                except KeyError:
                    continue
                for s in test:
                    fv = src._features.get(s.id)
                    if fv:
                        saved.append((src, s.id, fv))
                        src._features[s.id] = {kk: float("nan") for kk in fv}
            ns = agent.score_batch(test)
            sc = np.array([n.score if n is not None else np.nan for n in ns])
            cf = np.array([n.confidence if n is not None else np.nan for n in ns])
            for src, sid, fv in saved:
                src._features[sid] = fv

            pool["expert_tree"]["y"] += list(y0)
            pool["expert_tree"]["p"] += list(sc)
            pool["expert_tree"]["sig"] += list(cf)          # independent of the score

            preds = ml_with_missing(models, Xte, cols, set(drop), med)
            for m in ML_MODELS:
                p = np.asarray(preds[m], float)
                pool[m]["y"] += list(y0)
                pool[m]["p"] += list(p)
                pool[m]["sig"] += list(np.abs(p - 0.5))     # margin
            # random forest, second signal: agreement across trees
            mask = [i for i, c in enumerate(cols)
                    if ":" in c and c.split(":", 1)[0] in set(drop)]
            Xm = Xte.copy()
            if mask:
                Xm[:, mask] = med[mask]
            votes = np.stack([t.predict_proba(Xm)[:, 1] for t in models["random_forest"].estimators_])
            p_rf = votes.mean(0)
            pool["random_forest_var"]["y"] += list(y0)
            pool["random_forest_var"]["p"] += list(p_rf)
            pool["random_forest_var"]["sig"] += list(-votes.std(0))   # low variance = sure

    res = {}
    for m, dd in pool.items():
        y, p, sig = np.array(dd["y"]), np.array(dd["p"]), np.array(dd["sig"])
        ok = np.isfinite(p) & np.isfinite(sig)
        row = {f"auc@{int(c*100)}": sel_auc(y, p, sig, c) for c in COVERAGES}
        err = np.abs(y[ok] - p[ok])
        row["spearman_sig_vs_neg_err"] = float(spearmanr(sig[ok], -err).statistic)
        row["n_pooled"] = int(ok.sum())
        res[m] = row
        print(f"    {m:<20s} " + "  ".join(
            f"{k}={row[k]:.3f}" if row[k] is not None else f"{k}=n/a"
            for k in [f"auc@{int(c*100)}" for c in COVERAGES])
            + f"   rho_err={row['spearman_sig_vs_neg_err']:+.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", nargs="*", default=["Cu", "Au", "Ni", "W"])
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval_confidence.json"))
    args = ap.parse_args()
    catalog = build_catalog()
    summary = {}
    for metal in args.metals:
        print(f"\n{'='*70}\n  {metal}\n{'='*70}", flush=True)
        t0 = time.time()
        summary[metal] = run_metal(metal, METAL_CONFIGS[metal], catalog)
        print(f"  [{metal} {time.time()-t0:.0f}s]", flush=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
    print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
