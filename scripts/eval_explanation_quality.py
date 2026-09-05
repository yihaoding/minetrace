"""Explanation quality, measured identically for all five models.

`eval_interpretability.py` only ever measured OUR model, so "interpretable"
stayed an adjective. This script puts the expert tree, a decision tree, a random
forest, XGBoost and L1-logistic through the SAME three tests, at the same
feature granularity, on the same split:

  A. faithfulness  — the model's claimed top-1 driver vs the feature whose
                     masking actually moves the score most (all features tested,
                     neutral value = training median for every model alike)
  B. stability     — top-3 attributed features under 10 bootstrap refits,
                     Jaccard against the full-data model's top-3
  C. parsimony     — how many features carry 80% of the attribution mass

Attribution source per model (each model's own, no cross-imposition):
  expert tree   NodeScore: child_eff_weight x feature_w x feature_z
  decision tree decision path, node weighted-impurity-decrease
  random forest TreeSHAP (shap.TreeExplainer)
  XGBoost       booster pred_contribs (exact TreeSHAP)
  L1-logistic   |coef x (x - train_mean)|

Usage::

    python scripts/eval_explanation_quality.py --metals Cu
    python scripts/eval_explanation_quality.py            # all 9
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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".cache" / "pylibs"))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import DMatrix, XGBClassifier

from core.data_source import DataSourceRegistry
from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX
from scripts.eval_baselines import (
    N_TRAIN_NEG, align, expert_feature_keys, feature_frame, select_columns,
)
from scripts.eval_comprehensive import (
    METAL_CONFIGS, N_NEG, SEED, MAX_TEST, LAYERS, _forbid_xy_km, build_catalog,
    gen_random_negatives, load_mine_deposit_samples, load_no_target_sites,
)

N_BOOT = 10
TOPK = 3
MODELS = ["expert_tree", "expert_tree_sens", "decision_tree", "random_forest",
          "xgboost", "logit_l1"]
ML_MODELS = ["decision_tree", "random_forest", "xgboost", "logit_l1"]


# ── model fitting (bootstrap-friendly: no internal CV) ───────────────────────

def fit_models(Xtr, ytr, seed):
    n_pos, n_neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                class_weight="balanced", random_state=seed).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=max(n_neg / max(n_pos, 1), 1.0),
                        eval_metric="logloss", random_state=seed, n_jobs=8,
                        tree_method="hist").fit(Xtr, ytr)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                            class_weight="balanced", max_iter=2000,
                            random_state=seed).fit(sc.transform(Xtr), ytr)
    return {"decision_tree": dt, "random_forest": rf, "xgboost": xgb,
            "logit_l1": (lr, sc)}


def predict(name, model, X):
    if name == "logit_l1":
        lr, sc = model
        return lr.predict_proba(sc.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]


# ── attributions: |contribution| per (test point, feature) ───────────────────

def attrib_decision_tree(dt, X):
    t = dt.tree_
    N = float(t.n_node_samples[0])
    dec = dt.decision_path(X)
    out = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for node in dec.indices[dec.indptr[i]:dec.indptr[i + 1]]:
            f = t.feature[node]
            if f < 0:                              # leaf
                continue
            l, r = t.children_left[node], t.children_right[node]
            n_t, n_l, n_r = t.n_node_samples[node], t.n_node_samples[l], t.n_node_samples[r]
            gain = (n_t / N) * (t.impurity[node]
                                - (n_l / n_t) * t.impurity[l]
                                - (n_r / n_t) * t.impurity[r])
            out[i, f] += abs(gain)
    return out


def attrib_shap_rf(rf, X):
    sv = shap.TreeExplainer(rf).shap_values(X, check_additivity=False)
    sv = np.array(sv)
    if sv.ndim == 3:                               # (n, f, classes) or (classes, n, f)
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
    return np.abs(sv)


def attrib_shap_xgb(xgb, X):
    contribs = xgb.get_booster().predict(DMatrix(X), pred_contribs=True)
    return np.abs(np.asarray(contribs)[:, :-1])    # drop bias column


def attrib_logit(model, X):
    lr, sc = model
    Xs = sc.transform(X)
    return np.abs(lr.coef_[0][None, :] * Xs)       # scaler centres on train mean


def attrib_expert(node_scores, cols, sensitivity=False):
    """Attribution for the expert tree, mapped onto `cols`.

    sensitivity=False → eff_weight x feature_w x |z|   (what narrative.py ranks by)
    sensitivity=True  → the above x s(1-s) of the leaf, i.e. the sigmoid
                        derivative the value-decomposition omits. A saturated
                        leaf carries large |w z| but cannot move the score, so
                        only the second form should track masking influence.
    """
    idx = {c: i for i, c in enumerate(cols)}
    bare = {}
    for c in cols:
        bare.setdefault(c.split(":", 1)[1] if ":" in c else c, i_ := idx[c])
    out = np.zeros((len(node_scores), len(cols)))
    for i, ns in enumerate(node_scores):
        if ns is None:
            continue
        tot = sum(ns.weights.values()) or 1.0
        for cname, child in ns.children.items():
            eff = ns.weights.get(cname, 0.0) / tot
            if eff <= 0:
                continue
            slope = 1.0
            if sensitivity:
                sc = float(child.score)
                slope = max(sc * (1.0 - sc), 1e-12)
            for key, (z, w, _owner) in child.all_feature_signals().items():
                j = idx.get(key, bare.get(key.split(":", 1)[-1]))
                if j is not None:
                    out[i, j] += eff * abs(w * z) * slope
    return out


# ── empirical influence: mask each feature to the training median ────────────

def influence_ml(name, model, Xte, med):
    base = predict(name, model, Xte)
    out = np.zeros_like(Xte)
    for j in range(Xte.shape[1]):
        Xm = Xte.copy()
        Xm[:, j] = med[j]
        out[:, j] = np.abs(predict(name, model, Xm) - base)
    return out


def influence_expert(agent, test_samples, cols, med, base):
    """Mask through the real model: mutate the cached feature, re-score, restore."""
    out = np.zeros((len(test_samples), len(cols)))
    for j, col in enumerate(cols):
        src_name, feat = col.split(":", 1) if ":" in col else (None, col)
        try:
            src = DataSourceRegistry.get(src_name)
        except KeyError:
            continue
        saved = {}
        for s in test_samples:
            fv = src._features.get(s.id)
            if fv is not None and feat in fv:
                saved[s.id] = fv[feat]
                fv[feat] = med[j]
        if not saved:
            continue
        ns = agent.score_batch(test_samples)
        scores = np.array([n.score if n is not None else np.nan for n in ns])
        out[:, j] = np.abs(scores - base)
        for sid, v in saved.items():
            src._features[sid][feat] = v
    return out


def expert_groups(node_scores, cols) -> dict[str, list[int]]:
    """expert name -> column indices it reads (union over test points)."""
    idx = {c: i for i, c in enumerate(cols)}
    bare = {}
    for c in cols:
        bare.setdefault(c.split(":", 1)[1] if ":" in c else c, idx[c])
    groups: dict[str, set] = {}
    for ns in node_scores:
        if ns is None:
            continue
        for cname, child in ns.children.items():
            g = groups.setdefault(cname, set())
            for key in child.all_feature_signals():
                j = idx.get(key, bare.get(key.split(":", 1)[-1]))
                if j is not None:
                    g.add(j)
    return {k: sorted(v) for k, v in groups.items() if v}


def group_influence_ml(name, model, Xte, med, groups):
    base = predict(name, model, Xte)
    out = np.zeros((Xte.shape[0], len(groups)))
    for gi, (_g, idxs) in enumerate(groups.items()):
        Xm = Xte.copy()
        Xm[:, idxs] = med[idxs]
        out[:, gi] = np.abs(predict(name, model, Xm) - base)
    return out


def group_influence_expert(agent, test_samples, cols, med, base, groups):
    out = np.zeros((len(test_samples), len(groups)))
    for gi, (_g, idxs) in enumerate(groups.items()):
        saved = []
        for j in idxs:
            col = cols[j]
            src_name, feat = col.split(":", 1) if ":" in col else (None, col)
            try:
                src = DataSourceRegistry.get(src_name)
            except KeyError:
                continue
            for s in test_samples:
                fv = src._features.get(s.id)
                if fv is not None and feat in fv:
                    saved.append((src, s.id, feat, fv[feat]))
                    fv[feat] = med[j]
        if not saved:
            continue
        ns = agent.score_batch(test_samples)
        sc = np.array([n.score if n is not None else np.nan for n in ns])
        out[:, gi] = np.abs(sc - base)
        for src, sid, feat, v in saved:
            src._features[sid][feat] = v
    return out


def group_attrib(A, groups):
    return np.column_stack([A[:, idxs].sum(1) for idxs in groups.values()])


def influence_stats(E):
    mx = np.nanmax(np.nan_to_num(E), axis=1)
    return {"max_influence_median": float(np.median(mx)),
            "max_influence_p90": float(np.percentile(mx, 90)),
            "frac_points_max_below_1e3": float(np.mean(mx < 1e-3)),
            "frac_zero_cells": float(np.mean(np.nan_to_num(E) == 0))}


# ── metrics ─────────────────────────────────────────────────────────────────

def faithfulness(A, E):
    """Claimed top-1 vs empirically most influential feature."""
    ok = ~np.all(A == 0, axis=1) & ~np.all(np.nan_to_num(E) == 0, axis=1)
    if ok.sum() == 0:
        return None
    A, E = A[ok], np.nan_to_num(E[ok])
    claim = A.argmax(1)
    emp_order = np.argsort(-E, axis=1)
    hit1 = float(np.mean(claim == emp_order[:, 0]))
    hit3 = float(np.mean([c in emp_order[i, :3] for i, c in enumerate(claim)]))
    # rank correlation between claimed and empirical importance
    from scipy.stats import spearmanr
    rho = float(np.nanmean([spearmanr(A[i], E[i]).statistic for i in range(len(A))]))
    return {"hit_at_1": hit1, "hit_at_3": hit3, "spearman": rho, "n": int(ok.sum())}


def topk_sets(A, k=TOPK):
    return [set(np.argsort(-row)[:k]) for row in A]


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def parsimony(A, frac=0.8):
    """Median number of features carrying `frac` of the attribution mass."""
    ns = []
    for row in A:
        tot = row.sum()
        if tot <= 0:
            continue
        cs = np.cumsum(np.sort(row)[::-1]) / tot
        ns.append(int(np.searchsorted(cs, frac) + 1))
    return float(np.median(ns)) if ns else None


# ── main ────────────────────────────────────────────────────────────────────

def run_metal(metal, cfg, catalog) -> dict:
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
    rng3 = np.random.default_rng(SEED + 3)
    if len(neg) > N_NEG:
        neg = [neg[i] for i in rng3.choice(len(neg), N_NEG, replace=False)]

    train, test = train_pos + train_neg, test_pos + neg
    ytr = np.array([1] * len(train_pos) + [0] * len(train_neg))
    Ftr, Fte = feature_frame(agent, train), feature_frame(agent, test)
    Xtr_all, Xte_all, all_cols = align(Ftr, Fte)
    keep = select_columns(all_cols, "expert_matched",
                          expert_feature_keys(metal, cfg["pathfinders"], cfg["ratio_features"]))
    Xtr, Xte, cols = Xtr_all[:, keep], Xte_all[:, keep], [all_cols[i] for i in keep]
    med = np.median(Xtr, axis=0)
    print(f"  train {len(train)} / test {len(test)} / features {len(cols)}", flush=True)

    models = fit_models(Xtr, ytr, SEED)
    ns_full = agent.score_batch(test)
    base_expert = np.array([n.score if n is not None else np.nan for n in ns_full])

    A = {"expert_tree": attrib_expert(ns_full, cols),
         "expert_tree_sens": attrib_expert(ns_full, cols, sensitivity=True),
         "decision_tree": attrib_decision_tree(models["decision_tree"], Xte),
         "random_forest": attrib_shap_rf(models["random_forest"], Xte),
         "xgboost": attrib_shap_xgb(models["xgboost"], Xte),
         "logit_l1": attrib_logit(models["logit_l1"], Xte)}

    t0 = time.time()
    E = {"expert_tree": influence_expert(agent, test, cols, med, base_expert)}
    E["expert_tree_sens"] = E["expert_tree"]          # same model, same masking
    for m in ML_MODELS:
        E[m] = influence_ml(m, models[m], Xte, med)
    print(f"  masking done in {time.time() - t0:.0f}s", flush=True)

    groups = expert_groups(ns_full, cols)
    print(f"  expert groups: {len(groups)}  "
          f"sizes={[len(v) for v in groups.values()]}", flush=True)
    GE = {"expert_tree": group_influence_expert(agent, test, cols, med, base_expert, groups)}
    GE["expert_tree_sens"] = GE["expert_tree"]
    for m in ML_MODELS:
        GE[m] = group_influence_ml(m, models[m], Xte, med, groups)

    res = {}
    for m in MODELS:
        res[m] = {"faithfulness": faithfulness(A[m], E[m]),
                  "faithfulness_group": faithfulness(group_attrib(A[m], groups), GE[m]),
                  "influence_stats": influence_stats(E[m]),
                  "group_influence_stats": influence_stats(GE[m]),
                  "parsimony_n80": parsimony(A[m]),
                  "attrib_nonzero_frac": float(np.mean(A[m] > 0))}
    res["_groups"] = {k: len(v) for k, v in groups.items()}

    # ── bootstrap stability ────────────────────────────────────────────────
    full_top = {m: topk_sets(A[m]) for m in MODELS}
    acc = {m: [] for m in MODELS}
    for b in range(N_BOOT):
        rb = np.random.default_rng(SEED + 1000 + b)
        idx = rb.choice(len(ytr), len(ytr), replace=True)
        if len(set(ytr[idx])) < 2:
            continue
        mb = fit_models(Xtr[idx], ytr[idx], SEED + b)
        Ab = {"decision_tree": attrib_decision_tree(mb["decision_tree"], Xte),
              "random_forest": attrib_shap_rf(mb["random_forest"], Xte),
              "xgboost": attrib_shap_xgb(mb["xgboost"], Xte),
              "logit_l1": attrib_logit(mb["logit_l1"], Xte)}
        # expert tree: refit the fitted tree on the same bootstrap sample
        bpos = [train[i] for i in idx if ytr[i] == 1]
        bneg = [train[i] for i in idx if ytr[i] == 0]
        if len(bpos) >= 5 and len(bneg) >= 5:
            agent._tree.fit(bpos, bneg)
            ns_b = agent.score_batch(test)
            Ab["expert_tree"] = attrib_expert(ns_b, cols)
            Ab["expert_tree_sens"] = attrib_expert(ns_b, cols, sensitivity=True)
        for m in MODELS:
            if m in Ab:
                bt = topk_sets(Ab[m])
                acc[m].append(float(np.mean([jaccard(a, b_) for a, b_ in zip(full_top[m], bt)])))
    agent._tree.fit(train_pos, train_neg)             # restore full fit
    for m in MODELS:
        res[m]["stability_top3_jaccard"] = float(np.mean(acc[m])) if acc[m] else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", nargs="*", default=list(METAL_CONFIGS))
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval_explanation_quality.json"))
    args = ap.parse_args()

    catalog = build_catalog()
    summary = {}
    for metal in args.metals:
        print(f"\n{'='*70}\n  {metal}\n{'='*70}", flush=True)
        t0 = time.time()
        summary[metal] = run_metal(metal, METAL_CONFIGS[metal], catalog)
        for m in MODELS:
            r = summary[metal][m]
            f = r["faithfulness"] or {}
            g = r["faithfulness_group"] or {}
            i = r["influence_stats"]
            print(f"    {m:<15s} feat: Hit@1={f.get('hit_at_1', float('nan')):.3f} "
                  f"rho={f.get('spearman', float('nan')):+.3f} | "
                  f"group: Hit@1={g.get('hit_at_1', float('nan')):.3f} "
                  f"rho={g.get('spearman', float('nan')):+.3f} | "
                  f"stab={r['stability_top3_jaccard'] or float('nan'):.3f} "
                  f"n80={r['parsimony_n80']} "
                  f"maxinf={i['max_influence_median']:.2e}", flush=True)
        print(f"  [{metal} {time.time() - t0:.0f}s]", flush=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
    print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
