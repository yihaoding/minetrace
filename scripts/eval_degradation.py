"""Extensibility test: what happens when data sources go missing at inference?

The expert tree's architectural claim is that ONE fitted model covers terrain
with uneven coverage: an expert whose source is unavailable at a query point
returns None, drops out of the aggregation, the remaining weights renormalise,
and `confidence` falls. A fixed-width model (RF / DT / logistic) cannot do this
— it needs every column, so a missing modality must be imputed, which asserts a
value the data never supported.

This script withholds sources FROM THE TEST POINTS ONLY (training stays fully
covered — the deployment case: fit where data is rich, apply where it is patchy)
and measures how each model degrades.

  · AUC vs number of withheld sources (random subsets, averaged)
  · AUC when each single source is withheld
  · whether the expert tree's own `confidence` tracks the degradation

Usage::  python scripts/eval_degradation.py [--metals Cu Au]
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
from scripts.eval_explanation_quality import ML_MODELS, fit_models, predict

N_REP = 5          # random source subsets per k


def auc_of(y, p):
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() < 10 or len(set(np.asarray(y)[ok])) < 2:
        return None
    return float(roc_auc_score(np.asarray(y)[ok], p[ok]))


def expert_with_missing(agent, test, drop_sources):
    """Blank the given sources for the test points; the tree self-adjusts."""
    saved = []
    for name in drop_sources:
        try:
            src = DataSourceRegistry.get(name)
        except KeyError:
            continue
        for s in test:
            fv = src._features.get(s.id)
            if fv:
                saved.append((src, s.id, fv))
                src._features[s.id] = {k: float("nan") for k in fv}
    ns = agent.score_batch(test)
    scores = [n.score if n is not None else np.nan for n in ns]
    confs = [n.confidence if n is not None else np.nan for n in ns]
    n_experts = [len(n.children) if n is not None else 0 for n in ns]
    for src, sid, fv in saved:
        src._features[sid] = fv
    return np.array(scores), float(np.nanmean(confs)), float(np.mean(n_experts))


def ml_with_missing(models, Xte, cols, drop_sources, med):
    """Fixed-width models cannot skip a modality — the columns must be imputed."""
    mask = [i for i, c in enumerate(cols)
            if ":" in c and c.split(":", 1)[0] in drop_sources]
    Xm = Xte.copy()
    if mask:
        Xm[:, mask] = med[mask]                     # median imputation
    Xnan = Xte.copy()
    if mask:
        Xnan[:, mask] = np.nan                      # XGBoost handles NaN natively
    out = {}
    for m in ML_MODELS:
        out[m] = predict(m, models[m], Xm)
    out["xgboost_nan"] = models["xgboost"].predict_proba(Xnan)[:, 1]
    return out


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
    y = [1] * len(test_pos) + [0] * len(neg)
    ytr = np.array([1] * len(train_pos) + [0] * len(train_neg))
    Ftr, Fte = feature_frame(agent, train), feature_frame(agent, test)
    Xtr_all, Xte_all, all_cols = align(Ftr, Fte)
    keep = select_columns(all_cols, "expert_matched",
                          expert_feature_keys(metal, cfg["pathfinders"], cfg["ratio_features"]))
    Xtr, Xte, cols = Xtr_all[:, keep], Xte_all[:, keep], [all_cols[i] for i in keep]
    med = np.median(Xtr, axis=0)
    models = fit_models(Xtr, ytr, SEED)
    sources = list(agent.active_sources())
    print(f"  sources={sources}  test={len(test)}  features={len(cols)}", flush=True)

    res = {"sources": sources, "curve": {}, "single": {}}

    # ── cumulative withholding: k random sources removed ────────────────────
    for k in range(0, len(sources)):
        reps = [[]] if k == 0 else [
            list(np.random.default_rng(SEED + 7 * r).choice(sources, k, replace=False))
            for r in range(N_REP)]
        acc = {m: [] for m in ML_MODELS + ["xgboost_nan", "expert_tree"]}
        confs, n_exp = [], []
        for drop in reps:
            sc, conf, ne = expert_with_missing(agent, test, drop)
            a = auc_of(y, sc)
            if a is not None:
                acc["expert_tree"].append(a)
            confs.append(conf); n_exp.append(ne)
            for m, p in ml_with_missing(models, Xte, cols, set(drop), med).items():
                a = auc_of(y, p)
                if a is not None:
                    acc[m].append(a)
        res["curve"][k] = {m: (float(np.mean(v)) if v else None) for m, v in acc.items()}
        res["curve"][k]["_expert_confidence"] = float(np.mean(confs))
        res["curve"][k]["_experts_active"] = float(np.mean(n_exp))
        row = res["curve"][k]
        fmt = lambda x: f"{x:.3f}" if x is not None and np.isfinite(x) else "  n/a"
        print(f"    k={k}  expert={fmt(row['expert_tree'])} (conf {row['_expert_confidence']:.2f}, "
              f"{row['_experts_active']:.1f} experts)  rf={fmt(row['random_forest'])}  "
              f"xgb={fmt(row['xgboost'])}  xgb_nan={fmt(row['xgboost_nan'])}  "
              f"lr={fmt(row['logit_l1'])}  dt={fmt(row['decision_tree'])}", flush=True)

    # ── each single source withheld ─────────────────────────────────────────
    for s in sources:
        sc, conf, ne = expert_with_missing(agent, test, [s])
        row = {"expert_tree": auc_of(y, sc), "_expert_confidence": conf}
        for m, p in ml_with_missing(models, Xte, cols, {s}, med).items():
            row[m] = auc_of(y, p)
        res["single"][s] = row
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", nargs="*", default=list(METAL_CONFIGS))
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval_degradation.json"))
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
