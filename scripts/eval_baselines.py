"""Standard ML baselines against the expert tree, on identical splits.

Everything (positives, held-out test ids, the four negative strategies, the
5 km leakage buffer, the seeds) is imported from eval_comprehensive.py, so the
only thing that changes between arms is the model:

  expert_tree   — CompositeNode, AUC-weighted experts          (ours)
  decision_tree — DecisionTreeClassifier(max_depth=4)
  random_forest — RandomForestClassifier(500, balanced)
  xgboost       — XGBClassifier(400, depth 4, lr 0.05)
  logit_l1      — L1 logistic regression, C by 5-fold CV

Training set is the same for every arm and every test strategy: the KB
positives minus the held-out test ids, plus random WA background points with
assay coverage — i.e. exactly what the expert tree fits on.

Note the baselines get the FULL raw feature table from every active source
(~2k columns), not just the target/pathfinder subset the experts use, so this
is a generous setting for them.

Usage::

    python scripts/eval_baselines.py                 # all 9 metals
    python scripts/eval_baselines.py --metals Cu Au  # subset (smoke test)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core.data_source import DataSourceRegistry
from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX
from domains.geochem.samples import GeochemSample
from scripts.eval_comprehensive import (
    METAL_CONFIGS, SEED, MAX_TEST, N_NEG, SPATIAL_BOUNDARY, LAYERS,
    build_all_mine_tree, build_all_sites_combined, build_catalog,
    gen_far_random_negatives, gen_random_negatives, gen_region_negatives,
    load_mine_deposit_samples, load_no_target_sites, normalise_sites_df,
    _forbid_xy_km,
)

N_TRAIN_NEG = 300          # background points for model training (= agent's n_bg)
PREC_KS     = (10, 20, 50)

# Feature sets. `full` leaks exploration bias: n_samples_*km says how densely a
# spot has been sampled, and known deposits are sampled far more heavily than
# random background, so a tree can hit AUC 1.0 without using any geology.
# `no_density` removes that channel; `expert_matched` restricts the baselines to
# exactly the features the 10 experts see, which is the apples-to-apples arm.
FEATURE_SETS = ["full", "no_density", "expert_matched"]

DENSITY_PAT = ("n_samples_",)

_ENRICH_STATS = ["contrast_5_50", "contrast_10_50",
                 "5km_frac_above", "10km_frac_above", "5km_p90", "10km_p90"]
_PF_STATS = ["contrast_5_50", "contrast_10_50", "5km_frac_above", "10km_frac_above"]
_GEOPHYS_KEYS = ["mag", "mag_grad", "grav", "grav_grad",
                 "K", "Th", "U", "K_grad", "Th_grad", "U_grad", "K_Th", "Th_U", "U_K",
                 "LuHf", "SmNd", "LuHf_grad", "SmNd_grad"]
_STRUCT_KEYS = ["dist_fault_km", "fault_density_5km", "fault_density_10km",
                "dist_worm_mag_km", "dist_worm_grav_km",
                "is_rt_granitic", "is_rt_felsic", "is_rt_mafic", "is_rt_ultramafic",
                "is_rt_metased", "is_rt_sedimentary", "is_rt_metamorphic",
                "is_rt_hydrothermal", "age_ma", "is_yilgarn", "is_pilbara", "is_covered"]


def expert_feature_keys(target: str, pathfinders: dict, ratios: list[str]) -> set[str]:
    """Feature names (source prefix stripped) that the 10 experts actually read."""
    keys = {f"{target}_{st}" for st in _ENRICH_STATS}
    for el in pathfinders:
        keys |= {f"{el}_{st}" for st in _PF_STATS}
    keys |= set(ratios) | set(_GEOPHYS_KEYS) | set(_STRUCT_KEYS)
    return keys


def select_columns(cols: list[str], fs: str, expert_keys: set[str]) -> list[int]:
    if fs == "full":
        return list(range(len(cols)))
    idx = []
    for i, c in enumerate(cols):
        key = c.split(":", 1)[1] if ":" in c else c
        if any(p in key for p in DENSITY_PAT):
            continue
        if fs == "expert_matched" and key not in expert_keys:
            continue
        idx.append(i)
    return idx


# ── feature extraction ───────────────────────────────────────────────────────

def feature_frame(agent: GeochemAgent, samples: list[GeochemSample]) -> pd.DataFrame:
    """Full raw feature table for `samples`, pooled over all active sources."""
    srcs = []
    for name in agent.active_sources():
        try:
            src = DataSourceRegistry.get(name)
        except KeyError:
            continue
        if hasattr(src, "register_samples"):
            src.register_samples(samples)
        srcs.append((name, src))

    rows = []
    for s in samples:
        row: dict[str, float] = {}
        for name, src in srcs:
            fv = src.get_features(s.id)
            if not fv:
                continue
            for k, v in fv.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    row[f"{name}:{k}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows, index=[s.id for s in samples])


def align(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop all-NaN columns, align test to train, median-impute from train."""
    cols = [c for c in train.columns if train[c].notna().any()]
    tr = train[cols]
    te = test.reindex(columns=cols)
    med = tr.median()
    med = med.fillna(0.0)
    return tr.fillna(med).values, te.fillna(med).values, cols


def leakage_diag(test_pos: list[GeochemSample],
                 train_pos: list[GeochemSample]) -> dict:
    """How far is each held-out positive from the nearest TRAINING positive?

    The eval protocol buffers negatives 5 km from test positives but puts no
    buffer between train and test positives. Deposits cluster, and features are
    5/10/50 km neighbourhood statistics, so a test positive a few km from a
    training positive shares most of its feature window — a memorising model
    (RF/XGB) can exploit that, a z-score/AUC model cannot.
    """
    if not test_pos or not train_pos:
        return {}
    from scipy.spatial import cKDTree
    KM = 100.0
    tree = cKDTree(np.array([[p.x * KM, p.y * KM] for p in train_pos]))
    d, _ = tree.query(np.array([[p.x * KM, p.y * KM] for p in test_pos]), k=1)
    return {"median_km": round(float(np.median(d)), 2),
            "min_km": round(float(np.min(d)), 2),
            "p90_km": round(float(np.percentile(d, 90)), 2),
            "frac_within_5km": round(float(np.mean(d < 5)), 3),
            "frac_within_10km": round(float(np.mean(d < 10)), 3)}


# ── metrics ──────────────────────────────────────────────────────────────────

def metrics(y_true: list[int], y_score: list[float]) -> Optional[dict]:
    if len(y_true) < 10 or len(set(y_true)) < 2:
        return None
    auc = float(roc_auc_score(y_true, y_score))
    ranked = sorted(zip(y_score, y_true), reverse=True)
    prec = {k: sum(l for _, l in ranked[:k]) / k for k in PREC_KS if k <= len(ranked)}
    return {"auc": auc, "n_pos": int(sum(y_true)), "n_neg": int(len(y_true) - sum(y_true)),
            "prec": prec}


# ── models ───────────────────────────────────────────────────────────────────

def fit_predict(name: str, Xtr, ytr, Xte, seed: int) -> np.ndarray:
    n_pos, n_neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    if name == "decision_tree":
        m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                   class_weight="balanced", random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if name == "random_forest":
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                   class_weight="balanced_subsample",
                                   random_state=seed, n_jobs=-1)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if name == "xgboost":
        m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=max(n_neg / max(n_pos, 1), 1.0),
                          eval_metric="logloss", random_state=seed, n_jobs=8,
                          tree_method="hist")
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if name == "logit_l1":
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        folds = int(min(5, n_pos, n_neg))
        if folds >= 3:
            m = LogisticRegressionCV(penalty="l1", solver="liblinear",
                                     Cs=np.logspace(-3, 1, 12), cv=folds,
                                     scoring="roc_auc", class_weight="balanced",
                                     max_iter=2000, random_state=seed)
        else:
            m = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                   class_weight="balanced", max_iter=2000,
                                   random_state=seed)
        m.fit(Xtr_s, ytr)
        return m.predict_proba(Xte_s)[:, 1]
    raise ValueError(name)


MODELS = ["decision_tree", "random_forest", "xgboost", "logit_l1"]


# ── one (agent, train set) vs several test sets ──────────────────────────────

def evaluate(agent: GeochemAgent,
             train_pos: list[GeochemSample],
             train_neg: list[GeochemSample],
             scenarios: dict[str, tuple[list, list]],
             expert_keys: set[str],
             seed: int) -> dict:
    """Fit every baseline x feature set on (train_pos, train_neg); score each scenario."""
    train = train_pos + train_neg
    ytr = np.array([1] * len(train_pos) + [0] * len(train_neg))
    Ftr = feature_frame(agent, train)

    out: dict[str, dict] = {}
    for scen, (tpos, tneg) in scenarios.items():
        test = tpos + tneg
        y_true = [1] * len(tpos) + [0] * len(tneg)
        if not test:
            out[scen] = {}
            continue
        Fte = feature_frame(agent, test)
        Xtr_all, Xte_all, cols = align(Ftr, Fte)

        # ours — identical test set, scored by the fitted expert tree
        ns_list = agent.score_batch(test)
        pairs = [(ns.score, l) for ns, l in zip(ns_list, y_true) if ns is not None]
        res: dict = {"expert_tree": metrics([l for _, l in pairs], [s for s, _ in pairs])
                     if pairs else None,
                     "featuresets": {},
                     "_n_train": {"pos": int(ytr.sum()), "neg": int(len(ytr) - ytr.sum())}}

        for fs in FEATURE_SETS:
            keep = select_columns(cols, fs, expert_keys)
            if not keep:
                res["featuresets"][fs] = {"_n_features": 0}
                continue
            Xtr, Xte = Xtr_all[:, keep], Xte_all[:, keep]
            arm: dict = {"_n_features": len(keep)}
            for mdl in MODELS:
                t0 = time.time()
                try:
                    p = fit_predict(mdl, Xtr, ytr, Xte, seed)
                    arm[mdl] = metrics(y_true, list(p))
                except Exception as exc:
                    print(f"      {fs}/{mdl}: FAILED ({type(exc).__name__}: {exc})")
                    arm[mdl] = None
                if arm.get(mdl):
                    arm[mdl]["fit_s"] = round(time.time() - t0, 1)
            # what is the forest actually keying on?
            try:
                rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            class_weight="balanced_subsample",
                                            random_state=seed, n_jobs=-1).fit(Xtr, ytr)
                imp = sorted(zip([cols[i] for i in keep], rf.feature_importances_),
                             key=lambda t: -t[1])[:8]
                arm["_top_features"] = [[c, round(float(v), 4)] for c, v in imp]
            except Exception:
                pass
            res["featuresets"][fs] = arm
        out[scen] = res
    return out


def print_scenario(metal: str, scen: str, res: dict) -> None:
    n = res.get("_n_train", {})
    print(f"\n  {metal} / {scen}   (train {n.get('pos','?')}+/{n.get('neg','?')}-)")
    r = res.get("expert_tree")
    if r:
        p = r["prec"]
        print(f"    {'expert_tree':<15s} {'(10 experts)':<16s} AUC={r['auc']:.3f}"
              f"  P@10={p.get(10, float('nan')):.2f}  P@20={p.get(20, float('nan')):.2f}"
              f"  n=({r['n_pos']}+/{r['n_neg']}-)")
    for fs in FEATURE_SETS:
        arm = res.get("featuresets", {}).get(fs, {})
        nf = arm.get("_n_features", 0)
        for mdl in MODELS:
            m = arm.get(mdl)
            if not m:
                print(f"    {mdl:<15s} {fs + f' ({nf}f)':<16s} —")
                continue
            p = m["prec"]
            print(f"    {mdl:<15s} {fs + f' ({nf}f)':<16s} AUC={m['auc']:.3f}"
                  f"  P@10={p.get(10, float('nan')):.2f}  P@20={p.get(20, float('nan')):.2f}")
        top = arm.get("_top_features")
        if top:
            print(f"      RF top: " + ", ".join(f"{c.split(':')[-1]}={v}" for c, v in top[:5]))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", nargs="*", default=list(METAL_CONFIGS))
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval_baselines_summary.json"))
    args = ap.parse_args()

    catalog   = build_catalog()
    mine_tree = build_all_mine_tree()
    all_sites = build_all_sites_combined()

    all_mine_ids: set[str] = set()
    for cfg in METAL_CONFIGS.values():
        df = normalise_sites_df(pd.read_csv(cfg["sites_csv"]))
        ids = df[df[cfg["site_col"]].fillna("").str.contains("Mine|Deposit")]["SITE_CODE"]
        all_mine_ids.update(ids.astype(str))

    summary: dict[str, dict] = {}

    for metal in args.metals:
        cfg = METAL_CONFIGS[metal]
        print(f"\n{'='*76}\n  {metal}  —  expert tree vs {len(MODELS)} ML baselines\n{'='*76}")
        t_metal = time.time()

        all_pos = load_mine_deposit_samples(cfg["sites_csv"], cfg["site_col"],
                                            cfg["confidence_map"])
        all_pos_ids = {s.id for s in all_pos}
        rng  = np.random.default_rng(SEED)
        perm = rng.permutation(len(all_pos))
        n_test = min(MAX_TEST, max(0, len(all_pos) - 10))
        test_pos = [all_pos[i] for i in perm[:n_test]]
        test_ids = {s.id for s in test_pos}
        train_pos = [s for s in all_pos if s.id not in test_ids]
        forbid = _forbid_xy_km(test_pos)

        tc = GeochemTargetConfig(
            target=metal, pathfinders=cfg["pathfinders"],
            ratio_features=cfg["ratio_features"], confidence_map=cfg["confidence_map"],
            sites_csv=cfg["sites_csv"], layers=LAYERS,
        )
        agent = GeochemAgent(catalog=catalog, target_config=tc,
                             n_bg=300, bbox=WA_BBOX, seed=SEED)
        agent.plan(check_bbox=WA_BBOX)
        agent.setup(exclude_ids=test_ids)
        print(f"  positives {len(all_pos)} (train {len(train_pos)} / test {len(test_pos)})"
              f"   experts {len(agent.active_experts())}")

        # training negatives: random WA background with coverage (= agent's background)
        train_neg = gen_random_negatives(agent, N_TRAIN_NEG, seed=SEED + 100,
                                         forbid_xy_km=forbid)

        # test negatives, three strategies sharing the same test positives
        rand_neg = gen_random_negatives(agent, N_NEG, seed=SEED + 1, forbid_xy_km=forbid)
        far_neg  = gen_far_random_negatives(agent, mine_tree, N_NEG, seed=SEED + 2,
                                            forbid_xy_km=forbid)
        nt_neg   = load_no_target_sites(cfg["no_target_csv"], all_pos_ids,
                                        forbid_xy_km=forbid)
        rng3 = np.random.default_rng(SEED + 3)
        if len(nt_neg) > N_NEG:
            nt_neg = [nt_neg[i] for i in rng3.choice(len(nt_neg), N_NEG, replace=False)]

        scen = {
            "random":       (test_pos, rand_neg),
            "far_random":   (test_pos, far_neg),
            "true_nonmine": (test_pos, nt_neg),
        }
        diag = leakage_diag(test_pos, train_pos)
        print(f"  test->nearest-train-positive: median {diag.get('median_km','?')} km, "
              f"min {diag.get('min_km','?')} km, "
              f"<5km {diag.get('frac_within_5km','?')}, <10km {diag.get('frac_within_10km','?')}")
        expert_keys = expert_feature_keys(metal, cfg["pathfinders"], cfg["ratio_features"])
        res = evaluate(agent, train_pos, train_neg, scen, expert_keys, seed=SEED)
        for s in scen:
            print_scenario(metal, s, res[s])

        # ── spatial holdout: south-trained, north-tested ────────────────────
        df_full = normalise_sites_df(pd.read_csv(cfg["sites_csv"]))
        mine_df = df_full[df_full[cfg["site_col"]].fillna("").str.contains("Mine|Deposit")]
        north_all = [GeochemSample(site_code=str(r["SITE_CODE"]), x=float(r["X"]), y=float(r["Y"]))
                     for _, r in mine_df[mine_df.Y >= SPATIAL_BOUNDARY].iterrows()]
        south_all = [GeochemSample(site_code=str(r["SITE_CODE"]), x=float(r["X"]), y=float(r["Y"]))
                     for _, r in mine_df[mine_df.Y < SPATIAL_BOUNDARY].iterrows()]
        rng4 = np.random.default_rng(SEED + 4)
        n_north = min(MAX_TEST, len(north_all))
        test_north = ([north_all[i] for i in rng4.choice(len(north_all), n_north, replace=False)]
                      if len(north_all) > n_north else list(north_all))

        if len(south_all) >= 10 and len(test_north) >= 5:
            agent_sp = GeochemAgent(catalog=catalog, target_config=tc,
                                    n_bg=300, bbox=WA_BBOX, seed=SEED + 10)
            agent_sp.plan(check_bbox=WA_BBOX)
            agent_sp.setup(exclude_ids={s.id for s in north_all})
            forbid_n = _forbid_xy_km(test_north)
            sp_neg = gen_region_negatives(all_sites.Y >= SPATIAL_BOUNDARY, all_sites,
                                          all_mine_ids, N_NEG, seed=SEED + 5,
                                          forbid_xy_km=forbid_n)
            sp_train_neg = gen_random_negatives(agent_sp, N_TRAIN_NEG, seed=SEED + 110,
                                                forbid_xy_km=forbid_n)
            res_sp = evaluate(agent_sp, south_all, sp_train_neg,
                              {"spatial": (test_north, sp_neg)}, expert_keys, seed=SEED)
            res["spatial"] = res_sp["spatial"]
            d_sp = leakage_diag(test_north, south_all)
            print(f"  spatial: north-test->nearest-south-train median "
                  f"{d_sp.get('median_km','?')} km, <10km {d_sp.get('frac_within_10km','?')}")
            res.setdefault("_leakage", {})["spatial_split"] = d_sp
            print_scenario(metal, "spatial", res["spatial"])
        else:
            res["spatial"] = {}
            print(f"\n  {metal} / spatial — skipped "
                  f"(south={len(south_all)}, north={len(north_all)})")

        res.setdefault("_leakage", {})["random_split"] = diag
        summary[metal] = res
        print(f"\n  [{metal} done in {time.time() - t_metal:.0f}s]")

        with open(args.out, "w") as fh:                      # checkpoint each metal
            json.dump(summary, fh, indent=2, default=float)

    # ── final tables ─────────────────────────────────────────────────────────
    for scen in ("far_random", "true_nonmine", "spatial"):
        for fs in FEATURE_SETS:
            print(f"\n{'='*84}\n  AUC — {scen}   |   baseline feature set: {fs}\n{'='*84}")
            hdr = f"  {'metal':<6}" + "".join(f"{m:>15}" for m in ["expert_tree"] + MODELS)
            print(hdr); print("  " + "-" * (len(hdr) - 2))
            for metal, d in summary.items():
                r = d.get(scen) or {}
                arm = (r.get("featuresets") or {}).get(fs, {})
                line = f"  {metal:<6}"
                v = r.get("expert_tree")
                line += f"{v['auc']:>15.3f}" if v else f"{'—':>15}"
                for m in MODELS:
                    v = arm.get(m)
                    line += f"{v['auc']:>15.3f}" if v else f"{'—':>15}"
                print(line)
    print(f"\n  JSON: {args.out}")


if __name__ == "__main__":
    main()
