"""Interpretability evaluation — Module 1: faithfulness of expert-tree attributions.

Reuses the exact eval protocol of eval_comprehensive.py (same SEED, same
train/test split, same negative samplers) and measures, per metal:

1. additivity      — can the root score be exactly reconstructed from the
                     NodeScore tree (leaf feature_z + feature_w)?

                     NOTE (post IP-01/02/03 fix): every node now applies its own
                     _invert_score inside score_node() and no ancestor re-applies
                     it; feature_z is stored already inversion-adjusted, and
                     feature_w is keyed identically to feature_z. So the
                     reconstruction below applies NO sign flips of its own — the
                     old double-inversion mismatch no longer exists.
2. deletion curve  — mask features per-sample in order of the model-implied
                     attribution |w_norm × z| and re-score in z-space; AUC should
                     fall faster than under random-order deletion.
3. expert ablation — does the narrative's claimed dominant expert
                     (eff_weight × |score-0.5|, cf. narrative.ExpertContrib)
                     match the expert whose counterfactual removal causes the
                     largest score drop?  Hit@1 / Hit@2.
4. z sanity        — pathological |z| rates (>4/6/10/20σ), leaf-score saturation,
                     and the narrative weight-lookup miss rate (regression guard
                     for IP-02: should now be 0 for every metal, since narrative
                     reads feature_w, which is keyed exactly like feature_z).
5. deposit types   — SITE_TYPE_/SITE_SUB_T availability for the test positives
                     (feasibility check for the plausibility module).

Usage::

    METALS=Cu,Au /group/pmc050/yding/miniconda3/envs/geochem/bin/python \
        scripts/eval_interpretability.py
    # (unset METALS → all 9 metals)

Output: reports/eval_interpretability_summary.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_comprehensive as ec  # noqa: E402  (reuse protocol pieces verbatim)

from domains.geochem.experts.geochem_experts import _SIGMOID_SCALE_CORR  # noqa: E402
from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX  # noqa: E402

SEED     = ec.SEED
MAX_TEST = ec.MAX_TEST
N_NEG    = ec.N_NEG

_METALS_ENV = os.environ.get("METALS", "").strip()
METALS = ([m for m in _METALS_ENV.split(",") if m in ec.METAL_CONFIGS]
          if _METALS_ENV else list(ec.METAL_CONFIGS))

DELETION_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_RANDOM_ORDERS = 3


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# ── leaf re-scoring in z-space ────────────────────────────────────────────────
# Replicates each leaf type's score_node() arithmetic from the stored NodeScore
# plus the fitted expert object, optionally with a set of masked feature keys.
# Keys follow feature_z conventions ("src:feat" for multi-source leaves).

def leaf_rescore(expert, ns, masked: frozenset = frozenset()) -> Optional[float]:
    """Re-derive a leaf's score from its stored feature_z, optionally masking keys.

    No sign flips here: ns.feature_z is stored ALREADY direction- and
    inversion-adjusted (see NodeScore), so raw = Σ w·z / Σ w reproduces exactly
    the raw that score_node() fed to the sigmoid.
    """
    if hasattr(expert, "_states"):  # multi-source enrichment / pathfinder
        subs, ws = [], []
        for sn, st in expert._states.items():
            if st.source_weight < 1e-9:
                continue
            ts = tw = 0.0
            for f, w in st.weights.items():
                key = f"{sn}:{f}"
                if key in masked or key not in ns.feature_z:
                    continue
                ts += w * ns.feature_z[key]
                tw += w
            if tw < 1e-9:
                continue
            subs.append(_sigmoid(ts / tw))
            ws.append(st.source_weight)
        if not subs:
            return None
        w_arr = np.array(ws)
        return float(np.average(subs, weights=w_arr / w_arr.sum()))

    if hasattr(expert, "_pair_states"):  # multi-source correlation
        subs, ws = [], []
        for sn, state in expert._pair_states.items():
            sw = expert._source_weights.get(sn, 0.0)
            if sw < 1e-9:
                continue
            ts = tw = 0.0
            for (e1, e2), w in state["w"].items():
                if w < 1e-9:
                    continue
                key = f"{sn}:{e1}_{e2}"
                if key in masked or key not in ns.feature_z:
                    continue
                ts += w * ns.feature_z[key]   # stored value = sign × d × v1 × v2
                tw += w
            if tw < 1e-9:
                continue
            subs.append(_sigmoid((ts / tw) * _SIGMOID_SCALE_CORR))
            ws.append(sw)
        if not subs:
            return None
        w_arr = np.array(ws)
        return float(np.average(subs, weights=w_arr / w_arr.sum()))

    if hasattr(expert, "_shape_w"):  # single-source shape leaves (geophys/structure)
        ts = tw = 0.0
        for f, z in ns.feature_z.items():
            if f in masked:
                continue
            w = expert._shape_w.get(f, 0.0)
            ts += w * z
            tw += w
        if tw < 1e-9:
            return None
        return _sigmoid(ts / tw)

    return None  # unknown leaf type — excluded from reconstruction stats


def root_rescore(tree, root_ns, leaf_scores: dict) -> Optional[float]:
    """Replicates CompositeNode.score_node aggregation.

    No 1-s here: each child already applied its own inversion inside score_node().
    """
    total_s = total_w = 0.0
    raw_vals = []
    for child in tree.children:
        s = leaf_scores.get(child.name)
        if s is None:
            continue
        raw_vals.append(s)
        eff_w = root_ns.weights.get(child.name, 0.0)
        total_s += eff_w * s
        total_w += eff_w
    if not raw_vals:
        return None
    if total_w < 1e-9:
        return float(np.mean(raw_vals))     # composite fallback uses raw scores
    return float(np.clip(total_s / total_w, 0.0, 1.0))


# ── model-implied per-feature attribution (chain-normalized w × z) ────────────

def feature_attributions(tree, root_ns) -> list:
    """[(expert_name, feature_key, attribution)] — the linearized contribution
    each feature makes to the root score, per the model's own weight chain."""
    out = []
    total_eff = sum(root_ns.weights.get(c.name, 0.0)
                    for c in tree.children if c.name in root_ns.children)
    if total_eff < 1e-9:
        total_eff = 1.0
    for child in tree.children:
        ns = root_ns.children.get(child.name)
        if ns is None:
            continue
        w_exp = root_ns.weights.get(child.name, 0.0) / total_eff
        # ns.feature_w is the leaf-internal chain-normalized weight (source ×
        # within-source), keyed identically to feature_z. Multiplying by the
        # expert's own tree weight completes the chain from feature to root.
        for key, z in ns.feature_z.items():
            out.append((child.name, key, w_exp * ns.feature_w.get(key, 0.0) * z))
    return out


def rescore_masked(tree, root_ns, masked_by_expert: dict) -> Optional[float]:
    leaf_scores = {}
    for child in tree.children:
        ns = root_ns.children.get(child.name)
        if ns is None:
            continue
        m = masked_by_expert.get(child.name, frozenset())
        leaf_scores[child.name] = leaf_rescore(child, ns, m)
    return root_rescore(tree, root_ns, leaf_scores)


# ── per-metal analysis ────────────────────────────────────────────────────────

def analyse_metal(metal: str, cfg: dict, catalog, mine_tree) -> dict:
    print(f"\n{'='*72}\n  {metal} — interpretability module 1\n{'='*72}")

    all_pos = ec.load_mine_deposit_samples(cfg["sites_csv"], cfg["site_col"],
                                           cfg["confidence_map"])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_pos))
    n_test = min(MAX_TEST, max(0, len(all_pos) - 10))
    test_pos = [all_pos[i] for i in perm[:n_test]]
    test_ids = {s.id for s in test_pos}
    forbid_xy_km = ec._forbid_xy_km(test_pos)

    tc = GeochemTargetConfig(
        target=metal,
        pathfinders=cfg["pathfinders"],
        ratio_features=cfg["ratio_features"],
        confidence_map=cfg["confidence_map"],
        sites_csv=cfg["sites_csv"],
        layers=ec.LAYERS,
    )
    agent = GeochemAgent(catalog=catalog, target_config=tc,
                         n_bg=300, bbox=WA_BBOX, seed=SEED)
    agent.plan(check_bbox=WA_BBOX)
    agent.setup(exclude_ids=test_ids)
    tree = agent._tree

    inverted = [c.name for c in tree.children
                if getattr(c, "_invert_score", False)]

    # same negative sets as eval_comprehensive (far_random for the deletion set)
    far_neg = ec.gen_far_random_negatives(agent, mine_tree, N_NEG, seed=SEED + 2,
                                          forbid_xy_km=forbid_xy_km)
    samples = test_pos + far_neg
    labels  = [1] * len(test_pos) + [0] * len(far_neg)
    ns_list = agent.score_batch(samples)

    scored = [(ns, lbl) for ns, lbl in zip(ns_list, labels) if ns is not None]
    print(f"  scored {len(scored)}/{len(samples)} samples "
          f"({sum(l for _, l in scored)} pos)")

    # ── 1. additivity ─────────────────────────────────────────────────────────
    leaf_err, root_err, narr_gap = [], [], []
    for ns, _ in scored:
        leaf_scores = {}
        for child in tree.children:
            cns = ns.children.get(child.name)
            if cns is None:
                continue
            rec = leaf_rescore(child, cns)
            leaf_scores[child.name] = rec
            if rec is not None:
                leaf_err.append(abs(rec - cns.score))
        g = root_rescore(tree, ns, {c: cns.score for c, cns in ns.children.items()})
        if g is not None:
            root_err.append(abs(g - ns.score))
        # Narrative view: aggregate the leaf scores exactly as the narrative
        # DISPLAYS them. Post-fix these are the same scores the tree aggregates,
        # so this gap must now be ~0 — it is the IP-01 regression guard.
        tw = sum(ns.weights.get(c, 0.0) for c in ns.children)
        if tw > 1e-9:
            g_narr = sum(ns.weights.get(c, 0.0) * cns.score
                         for c, cns in ns.children.items()) / tw
            narr_gap.append(abs(g_narr - ns.score))

    additivity = {
        "leaf_recon_err_mean": float(np.mean(leaf_err)) if leaf_err else None,
        "leaf_recon_err_max":  float(np.max(leaf_err))  if leaf_err else None,
        "root_recon_err_mean": float(np.mean(root_err)) if root_err else None,
        "root_recon_err_max":  float(np.max(root_err))  if root_err else None,
        "n_inverted_experts":  len(inverted),
        "inverted_experts":    inverted,
        "narrative_view_gap_mean": float(np.mean(narr_gap)) if narr_gap else None,
        "narrative_view_gap_max":  float(np.max(narr_gap))  if narr_gap else None,
    }

    # ── 2. deletion curves ────────────────────────────────────────────────────
    per_sample = []
    for idx, (ns, lbl) in enumerate(scored):
        attrs = feature_attributions(tree, ns)
        attrs.sort(key=lambda t: abs(t[2]), reverse=True)
        per_sample.append((ns, lbl, attrs, idx))

    def curve(order_fn) -> list:
        aucs = []
        for frac in DELETION_FRACTIONS:
            preds, ys = [], []
            for ns, lbl, attrs, idx in per_sample:
                order = order_fn(attrs, idx)
                k = int(round(frac * len(order)))
                masked: dict = {}
                for exp_name, key, _ in order[:k]:
                    masked.setdefault(exp_name, set()).add(key)
                g = rescore_masked(tree, ns,
                                   {e: frozenset(v) for e, v in masked.items()})
                preds.append(0.5 if g is None else g)
                ys.append(lbl)
            aucs.append(float(roc_auc_score(ys, preds))
                        if len(set(ys)) > 1 else None)
        return aucs

    auc_attr = curve(lambda attrs, idx: attrs)

    rand_curves = []
    for r in range(N_RANDOM_ORDERS):
        def rand_order(attrs, idx, _r=r):
            rr = np.random.default_rng(SEED * 1000 + idx * 10 + _r)
            p = rr.permutation(len(attrs))
            return [attrs[i] for i in p]
        rand_curves.append(curve(rand_order))
    auc_rand = [float(np.mean([c[i] for c in rand_curves]))
                for i in range(len(DELETION_FRACTIONS))]

    # positive gap = attribution-ordered deletion destroys ranking faster
    gap = [r - a for a, r in zip(auc_attr, auc_rand)]
    deletion = {
        "fractions": DELETION_FRACTIONS,
        "auc_attr_order": auc_attr,
        "auc_random_order": auc_rand,
        "faithfulness_gap_mean": float(np.mean(gap[1:])),
    }

    # ── 3. expert-ablation consistency (positives with a claimed driver) ──────
    hits1 = hits2 = n_claim = 0
    for ns, lbl, _, _ in per_sample:
        if lbl != 1:
            continue
        active = [(c, cns) for c, cns in ns.children.items()
                  if ns.weights.get(c, 0.0) > 0.05 and cns.score > 0.52]
        if not active:
            continue
        active.sort(key=lambda t: ns.weights.get(t[0], 0.0) * abs(t[1].score - 0.5),
                    reverse=True)
        claimed = active[0][0]

        drops = []
        base_leaf = {c: cns.score for c, cns in ns.children.items()}
        for c in ns.children:
            ls = dict(base_leaf)
            ls[c] = None
            g = root_rescore(tree, ns, ls)
            drops.append((c, ns.score - (g if g is not None else 0.5)))
        drops.sort(key=lambda t: t[1], reverse=True)
        actual_rank = [c for c, _ in drops]
        n_claim += 1
        hits1 += int(actual_rank[0] == claimed)
        hits2 += int(claimed in actual_rank[:2])

    ablation = {
        "n_claimed": n_claim,
        "hit_at_1": (hits1 / n_claim) if n_claim else None,
        "hit_at_2": (hits2 / n_claim) if n_claim else None,
    }

    # ── 4. z sanity + narrative weight-lookup miss rate ───────────────────────
    all_z, sat, w_miss, w_total = [], 0, 0, 0
    n_leaf = 0
    for ns, _ in scored:
        for cns in ns.children.values():
            n_leaf += 1
            if cns.score > 0.99 or cns.score < 0.01:
                sat += 1
            # IP-02 regression guard: the narrative now resolves a feature's
            # weight through feature_w, which is keyed exactly like feature_z.
            # A miss here means a leaf failed to populate feature_w and its
            # displayed contribution (weight × z) would collapse to 0.
            for f, z in cns.feature_z.items():
                all_z.append(abs(z))
                w_total += 1
                if f not in cns.feature_w:
                    w_miss += 1
    az = np.array(all_z) if all_z else np.array([0.0])
    sanity = {
        "n_z_values": int(len(all_z)),
        "frac_abs_z_gt4":  float(np.mean(az > 4)),
        "frac_abs_z_gt6":  float(np.mean(az > 6)),
        "frac_abs_z_gt10": float(np.mean(az > 10)),
        "frac_abs_z_gt20": float(np.mean(az > 20)),
        "max_abs_z": float(az.max()),
        "leaf_saturation_frac": (sat / n_leaf) if n_leaf else None,
        "narrative_weight_miss_frac": (w_miss / w_total) if w_total else None,
    }

    # ── 5. deposit-type field availability for the plausibility module ────────
    df = ec.normalise_sites_df(pd.read_csv(cfg["sites_csv"]))
    df = df[df["SITE_CODE"].astype(str).isin(test_ids)]
    types = {}
    if "SITE_TYPE_" in df.columns:
        types["SITE_TYPE_"] = df["SITE_TYPE_"].fillna("NA").value_counts().to_dict()
    if "SITE_SUB_T" in df.columns:
        sub = df["SITE_SUB_T"].fillna("")
        types["SITE_SUB_T_nonempty"] = int((sub.str.len() > 0).sum())
        types["n_test_pos"] = int(len(df))

    print(f"  additivity: leaf max err {additivity['leaf_recon_err_max']:.2e}  "
          f"root max err {additivity['root_recon_err_max']:.2e}  "
          f"inverted={len(inverted)}  "
          f"narr gap max {additivity['narrative_view_gap_max']:.4f}")
    print(f"  deletion: attr {['%.3f' % a for a in auc_attr]}")
    print(f"            rand {['%.3f' % a for a in auc_rand]}  "
          f"gap={deletion['faithfulness_gap_mean']:.3f}")
    print(f"  ablation hit@1={ablation['hit_at_1']}  hit@2={ablation['hit_at_2']}  "
          f"(n={n_claim})")
    print(f"  z-sanity: >6σ {sanity['frac_abs_z_gt6']:.3%}  "
          f">20σ {sanity['frac_abs_z_gt20']:.3%}  max {sanity['max_abs_z']:.1f}  "
          f"saturation {sanity['leaf_saturation_frac']:.2%}  "
          f"weight-miss {sanity['narrative_weight_miss_frac']:.2%}")

    return {
        "additivity": additivity,
        "deletion": deletion,
        "ablation_consistency": ablation,
        "z_sanity": sanity,
        "deposit_types": types,
        "n_scored": len(scored),
    }


def main() -> None:
    catalog   = ec.build_catalog()
    mine_tree = ec.build_all_mine_tree()

    results = {}
    for metal in METALS:
        try:
            results[metal] = analyse_metal(metal, ec.METAL_CONFIGS[metal],
                                           catalog, mine_tree)
        except Exception as e:  # keep going; report per-metal failure
            import traceback
            traceback.print_exc()
            results[metal] = {"error": str(e)}

    out = ROOT / "reports" / "eval_interpretability_summary.json"
    with out.open("w") as fh:
        json.dump({
            "metals": METALS,
            "results": results,
            "config": {"SEED": SEED, "MAX_TEST": MAX_TEST, "N_NEG": N_NEG,
                       "deletion_fractions": DELETION_FRACTIONS,
                       "n_random_orders": N_RANDOM_ORDERS,
                       "layers": sorted(ec.LAYERS)},
        }, fh, indent=2)
    print(f"\nJSON summary: {out}")


if __name__ == "__main__":
    main()
