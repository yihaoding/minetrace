"""Multi-source geochemistry experts.

Three experts, each representing one analytical perspective.
Each aggregates evidence across ALL active assay sources — not one expert per source.

  TargetEnrichmentExpert   — is the target element locally enriched?
  TargetPathfinderExpert   — are pathfinder elements anomalous?
  ElementCorrelationExpert — do element pairs co-enrich (deposit pattern)?

At fit time : each source is fitted independently; per-source AUC used for
              source weighting (sources with better signal get more weight).
At score time: scores from all sources that cover the query point are
              aggregated by source weight → one score per expert.

This means expert count is always fixed (3), independent of which or how
many assay sources are active.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from core.data_source import DataSourceRegistry
from core.expert_tree import NodeScore

logger = logging.getLogger(__name__)

_ENRICHMENT_STATS = [
    "contrast_5_50", "contrast_10_50",
    "5km_frac_above", "10km_frac_above",
    "5km_p90", "10km_p90",
]
_PF_STATS = ["contrast_5_50", "contrast_10_50", "5km_frac_above", "10km_frac_above"]
_SIGMOID_SCALE_CORR = 3.0


# ── shared helpers ────────────────────────────────────────────────────────────

def _data_direction(feat: str, pos_rows: list[dict], bg_rows: list[dict]) -> float:
    pos_fin = np.array([fv.get(feat, np.nan) for fv in pos_rows], dtype=float)
    bg_fin  = np.array([fv.get(feat, np.nan) for fv in bg_rows],  dtype=float)
    pos_fin = pos_fin[np.isfinite(pos_fin)]
    bg_fin  = bg_fin[np.isfinite(bg_fin)]
    if len(pos_fin) > 0 and len(bg_fin) > 0:
        return 1.0 if np.median(pos_fin) >= np.median(bg_fin) else -1.0
    return 1.0


def _bg_stats(features: list[str], rows: list[dict]) -> tuple[dict, dict]:
    mean, std = {}, {}
    for f in features:
        vals = np.array([fv.get(f, np.nan) for fv in rows], dtype=float)
        fin  = vals[np.isfinite(vals)]
        mean[f] = float(np.mean(fin))  if len(fin) > 0 else 0.0
        std[f]  = float(np.std(fin))   if len(fin) > 1 else 1.0
        if std[f] < 1e-9:
            std[f] = 1.0
    return mean, std


def _weighted_z_score(
    fv: dict,
    features: list[str],
    bg_mean: dict,
    bg_std:  dict,
    weights: dict,
    direction: dict,
) -> Optional[tuple[float, dict, dict]]:
    """Weighted z-score → raw score + per-feature weights and z-scores."""
    total_s = total_w = 0.0
    z_scores: dict[str, float] = {}
    for f in features:
        v = fv.get(f, float("nan"))
        if not np.isfinite(v):
            continue
        w = weights.get(f, 0.0)
        d = direction.get(f, 1.0)
        z = (v - bg_mean.get(f, 0.0)) / bg_std.get(f, 1.0)
        z_scores[f] = float(z * d)
        total_s += w * d * z
        total_w += w
    if total_w < 1e-9:
        return None
    return total_s / total_w, {f: weights.get(f, 0.0) for f in features}, z_scores


def _chain_feature_w(
    src_weights: dict[str, float],
    per_src_z:   dict[str, dict[str, float]],
    per_src_w:   dict[str, dict[str, float]],
) -> dict[str, float]:
    """Chain-normalized per-feature weights, keyed "src:feat" to match feature_z.

    weight("sn:f") = (source_weight / Σ source_weights)
                   × (feature_weight_f / Σ feature_weights present in sn)

    Within one source this reproduces that source's raw pre-sigmoid score as
    Σ w · z; across sources it splits the source-mixing weights, which is the
    correct ranking scale for comparing features from different sources.
    """
    out: dict[str, float] = {}
    src_total = sum(src_weights.values())
    if src_total < 1e-9:
        return out
    for sn, sw in src_weights.items():
        z = per_src_z.get(sn, {})
        w = per_src_w.get(sn, {})
        feat_total = sum(w.get(f, 0.0) for f in z)   # only features actually present
        if feat_total < 1e-9:
            continue
        for f in z:
            out[f"{sn}:{f}"] = (sw / src_total) * (w.get(f, 0.0) / feat_total)
    return out


def _source_auc(
    source_name: str,
    features: list[str],
    pos_rows: list[dict],
    bg_rows:  list[dict],
    bg_mean:  dict,
    bg_std:   dict,
    weights:  dict,
    direction: dict,
) -> float:
    """Estimate this source's AUC on the training set (used for source weighting)."""
    scores, labels = [], []
    for fv, lbl in [(fv, 1) for fv in pos_rows] + [(fv, 0) for fv in bg_rows]:
        res = _weighted_z_score(fv, features, bg_mean, bg_std, weights, direction)
        if res is not None:
            scores.append(float(1.0 / (1.0 + np.exp(-res[0]))))
            labels.append(lbl)
    if len(set(labels)) < 2 or len(scores) < 10:
        return 0.5
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


# ── per-source fitted state ───────────────────────────────────────────────────

class _SourceState:
    """Fitted state for one assay source within a multi-source expert."""
    __slots__ = ("features", "bg_mean", "bg_std", "weights", "direction", "source_weight")

    def __init__(self):
        self.features:       list[str]      = []
        self.bg_mean:        dict[str, float] = {}
        self.bg_std:         dict[str, float] = {}
        self.weights:        dict[str, float] = {}
        self.direction:      dict[str, float] = {}
        self.source_weight:  float           = 0.0


# ── TargetEnrichmentExpert ────────────────────────────────────────────────────

class TargetEnrichmentExpert:
    """Is the target element locally enriched above the regional background?

    Aggregates evidence from all active assay sources.
    Features: contrast and percentile statistics at 5 km and 10 km.
    Weights: equal across features; direction learned from data per source.
    Source weighting: AUC on training set.
    """

    _invert_score: bool = False

    def __init__(self, target: str, source_names: list[str]) -> None:
        self.name = f"{target.lower()}_enrichment"
        self.required_sources: list[str] = list(source_names)
        self._target = target
        self._source_names = list(source_names)
        self._states: dict[str, _SourceState] = {}
        self._fitted = False

    def fit(self, positives: list, background: list) -> None:
        for sn in self._source_names:
            src = DataSourceRegistry.get(sn)
            features = [f"{self._target}_{s}" for s in _ENRICHMENT_STATS]
            pos_rows = [fv for s in positives  if (fv := src.get_features(s.id)) is not None]
            bg_rows  = [fv for s in background if (fv := src.get_features(s.id)) is not None]
            if not bg_rows:
                continue
            st = _SourceState()
            st.features  = features
            st.bg_mean, st.bg_std = _bg_stats(features, bg_rows)
            eq = 1.0 / len(features)
            st.weights   = {f: eq for f in features}
            st.direction = {f: _data_direction(f, pos_rows, bg_rows) for f in features}
            auc = _source_auc(sn, features, pos_rows, bg_rows,
                              st.bg_mean, st.bg_std, st.weights, st.direction)
            if auc < 0.5:
                auc = 1.0 - auc
                for f in features:
                    st.direction[f] *= -1.0
            st.source_weight = max(0.0, (auc - 0.5) * 2.0)
            self._states[sn] = st
            logger.info("  %s / %s  AUC=%.3f  src_w=%.3f",
                        self.name, sn, auc, st.source_weight)
        self._fitted = True

    def fit_kb(self, background: list) -> None:
        for sn in self._source_names:
            src = DataSourceRegistry.get(sn)
            features = [f"{self._target}_{s}" for s in _ENRICHMENT_STATS]
            bg_rows = [fv for s in background if (fv := src.get_features(s.id)) is not None]
            if not bg_rows:
                continue
            st = _SourceState()
            st.features = features
            st.bg_mean, st.bg_std = _bg_stats(features, bg_rows)
            eq = 1.0 / len(features)
            st.weights   = {f: eq for f in features}
            st.direction = {f: 1.0 for f in features}
            st.source_weight = 1.0
            self._states[sn] = st
        self._fitted = True

    def score_node(self, sample) -> Optional[NodeScore]:
        if not self._fitted:
            return None
        sub_scores: list[float] = []
        used_weights: dict[str, float] = {}   # sn → source_weight, only for sources that scored
        all_z: dict[str, float] = {}
        per_src_z: dict[str, dict[str, float]] = {}
        for sn, st in self._states.items():
            if st.source_weight < 1e-9:
                continue
            src = DataSourceRegistry.get(sn)
            fv  = src.get_features(sample.id)
            if fv is None:
                continue
            res = _weighted_z_score(fv, st.features, st.bg_mean, st.bg_std,
                                    st.weights, st.direction)
            if res is None:
                continue
            raw, _, z = res
            if self._invert_score:
                raw = -raw
                z = {k: -v for k, v in z.items()}   # keep z consistent with the reported score
            sub_scores.append(float(1.0 / (1.0 + np.exp(-raw))))
            used_weights[sn] = st.source_weight
            per_src_z[sn] = z
            for k, v in z.items():
                all_z[f"{sn}:{k}"] = v
        if not sub_scores:
            return None
        w_arr = np.array(list(used_weights.values()))
        score = float(np.average(sub_scores, weights=w_arr / w_arr.sum()))
        return NodeScore(
            name=self.name, score=score,
            weights=dict(used_weights),   # keyed by SOURCE name, only sources that scored
            feature_z=all_z,
            feature_w=_chain_feature_w(
                used_weights, per_src_z,
                {sn: self._states[sn].weights for sn in used_weights},
            ),
        )

    def score(self, sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None


# ── TargetPathfinderExpert ────────────────────────────────────────────────────

class TargetPathfinderExpert:
    """Are geologically expected pathfinder elements anomalous?

    Pathfinder weights come from TargetConfig (domain knowledge).
    Direction learned from training data per source.
    Aggregates across all active assay sources.
    """

    _invert_score: bool = False

    def __init__(
        self,
        target: str,
        source_names: list[str],
        pathfinders: dict[str, float],
        ratio_features: list[str] | None = None,
    ) -> None:
        self.name = f"{target.lower()}_pathfinder"
        self.required_sources: list[str] = list(source_names)
        self._source_names  = list(source_names)
        self._pathfinders   = pathfinders
        self._ratio_features = ratio_features or []
        self._states: dict[str, _SourceState] = {}
        self._fitted = False

    def _resolve_features(self, sample_fv: dict) -> list[str]:
        keys = []
        for elem in self._pathfinders:
            for stat in _PF_STATS:
                f = f"{elem}_{stat}"
                if f in sample_fv:
                    keys.append(f)
        for rf in self._ratio_features:
            if rf in sample_fv:
                keys.append(rf)
        return keys

    def fit(self, positives: list, background: list) -> None:
        for sn in self._source_names:
            src = DataSourceRegistry.get(sn)
            pos_rows = [fv for s in positives  if (fv := src.get_features(s.id)) is not None]
            bg_rows  = [fv for s in background if (fv := src.get_features(s.id)) is not None]
            if not bg_rows:
                continue
            features = self._resolve_features(bg_rows[0])
            if not features:
                continue
            st = _SourceState()
            st.features = features
            st.bg_mean, st.bg_std = _bg_stats(features, bg_rows)
            for elem, pw in self._pathfinders.items():
                for stat in _PF_STATS:
                    f = f"{elem}_{stat}"
                    if f in features:
                        st.weights[f]   = pw
                        st.direction[f] = _data_direction(f, pos_rows, bg_rows)
            for rf in self._ratio_features:
                if rf in features:
                    st.weights[rf]   = 1.0
                    st.direction[rf] = _data_direction(rf, pos_rows, bg_rows)
            auc = _source_auc(sn, features, pos_rows, bg_rows,
                              st.bg_mean, st.bg_std, st.weights, st.direction)
            if auc < 0.5:
                auc = 1.0 - auc
                for f in features:
                    st.direction[f] *= -1.0
            st.source_weight = max(0.0, (auc - 0.5) * 2.0)
            self._states[sn] = st
            logger.info("  %s / %s  AUC=%.3f  src_w=%.3f",
                        self.name, sn, auc, st.source_weight)
        self._fitted = True

    def fit_kb(self, background: list) -> None:
        for sn in self._source_names:
            src = DataSourceRegistry.get(sn)
            bg_rows = [fv for s in background if (fv := src.get_features(s.id)) is not None]
            if not bg_rows:
                continue
            features = self._resolve_features(bg_rows[0])
            if not features:
                continue
            st = _SourceState()
            st.features = features
            st.bg_mean, st.bg_std = _bg_stats(features, bg_rows)
            for elem, pw in self._pathfinders.items():
                for stat in _PF_STATS:
                    f = f"{elem}_{stat}"
                    if f in features:
                        st.weights[f]   = pw
                        st.direction[f] = 1.0
            for rf in self._ratio_features:
                if rf in features:
                    st.weights[rf]   = 1.0
                    st.direction[rf] = 1.0
            st.source_weight = 1.0
            self._states[sn] = st
        self._fitted = True

    def score_node(self, sample) -> Optional[NodeScore]:
        if not self._fitted:
            return None
        sub_scores: list[float] = []
        used_weights: dict[str, float] = {}
        all_z: dict[str, float] = {}
        per_src_z: dict[str, dict[str, float]] = {}
        for sn, st in self._states.items():
            if st.source_weight < 1e-9 or not st.features:
                continue
            src = DataSourceRegistry.get(sn)
            fv  = src.get_features(sample.id)
            if fv is None:
                continue
            res = _weighted_z_score(fv, st.features, st.bg_mean, st.bg_std,
                                    st.weights, st.direction)
            if res is None:
                continue
            raw, _, z = res
            if self._invert_score:
                raw = -raw
                z = {k: -v for k, v in z.items()}
            sub_scores.append(float(1.0 / (1.0 + np.exp(-raw))))
            used_weights[sn] = st.source_weight
            per_src_z[sn] = z
            for k, v in z.items():
                all_z[f"{sn}:{k}"] = v
        if not sub_scores:
            return None
        w_arr = np.array(list(used_weights.values()))
        score = float(np.average(sub_scores, weights=w_arr / w_arr.sum()))
        return NodeScore(
            name=self.name, score=score,
            weights=dict(used_weights),
            feature_z=all_z,
            feature_w=_chain_feature_w(
                used_weights, per_src_z,
                {sn: self._states[sn].weights for sn in used_weights},
            ),
        )

    def score(self, sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None


# ── ElementCorrelationExpert ──────────────────────────────────────────────────

class ElementCorrelationExpert:
    """Do element pairs co-enrich near deposits?

    co(A, B) = A_contrast_5_50 × B_contrast_5_50
    Positive and large ↔ both elements are locally elevated together.

    Pairs are derived from target + all pathfinder elements.
    Aggregates across all active assay sources.
    """

    _invert_score: bool = False

    def __init__(
        self,
        target: str,
        source_names: list[str],
        pairs: list[tuple[str, str]],
    ) -> None:
        self.name = f"{target.lower()}_correlation"
        self.required_sources: list[str] = list(source_names)
        self._source_names = list(source_names)
        self._pairs = pairs
        # fitted per source: {sn: {pair: (weight, direction)}}
        self._pair_states: dict[str, dict] = {}
        self._source_weights: dict[str, float] = {}
        self._fitted = False

    def fit(self, positives: list, background: list) -> None:
        for sn in self._source_names:
            src = DataSourceRegistry.get(sn)
            pos_rows = [fv for s in positives  if (fv := src.get_features(s.id)) is not None]
            bg_rows  = [fv for s in background if (fv := src.get_features(s.id)) is not None]
            if not pos_rows or not bg_rows:
                continue

            pair_w, pair_d = {}, {}
            for e1, e2 in self._pairs:
                k1, k2 = f"{e1}_contrast_5_50", f"{e2}_contrast_5_50"

                def _co(rows):
                    return np.array([
                        fv.get(k1, np.nan) * fv.get(k2, np.nan)
                        for fv in rows
                        if np.isfinite(fv.get(k1, np.nan)) and np.isfinite(fv.get(k2, np.nan))
                    ])

                pos_co = _co(pos_rows)
                bg_co  = _co(bg_rows)
                if len(pos_co) < 5 or len(bg_co) < 5:
                    pair_w[(e1, e2)] = 0.0
                    pair_d[(e1, e2)] = 1.0
                    continue
                all_s = np.concatenate([pos_co, bg_co])
                all_l = np.array([1] * len(pos_co) + [0] * len(bg_co))
                try:
                    auc = float(roc_auc_score(all_l, all_s))
                except Exception:
                    auc = 0.5
                d = 1.0 if auc >= 0.5 else -1.0
                auc = auc if auc >= 0.5 else 1.0 - auc
                pair_w[(e1, e2)] = max(0.0, (auc - 0.5) * 2.0)
                pair_d[(e1, e2)] = d

            self._pair_states[sn] = {"w": pair_w, "d": pair_d}

            # source-level AUC: score all training samples with this source
            scores, labels = [], []
            for fv, lbl in [(fv, 1) for fv in pos_rows] + [(fv, 0) for fv in bg_rows]:
                raw = self._score_raw(fv, pair_w, pair_d)
                if raw is not None:
                    scores.append(float(1.0 / (1.0 + np.exp(-raw * _SIGMOID_SCALE_CORR))))
                    labels.append(lbl)
            if len(set(labels)) < 2 or len(scores) < 10:
                src_auc = 0.5
            else:
                try:
                    src_auc = float(roc_auc_score(labels, scores))
                except Exception:
                    src_auc = 0.5
            if src_auc < 0.5:
                src_auc = 1.0 - src_auc
            self._source_weights[sn] = max(0.0, (src_auc - 0.5) * 2.0)
            logger.info("  %s / %s  AUC=%.3f  src_w=%.3f",
                        self.name, sn, src_auc, self._source_weights[sn])
        self._fitted = True

    def fit_kb(self, background: list) -> None:
        for sn in self._source_names:
            self._pair_states[sn] = {
                "w": {p: 1.0 / max(len(self._pairs), 1) for p in self._pairs},
                "d": {p: 1.0 for p in self._pairs},
            }
            self._source_weights[sn] = 1.0
        self._fitted = True

    def _score_raw(self, fv: dict, pair_w: dict, pair_d: dict) -> Optional[float]:
        total_s = total_w = 0.0
        for (e1, e2), w in pair_w.items():
            if w < 1e-9:
                continue
            k1, k2 = f"{e1}_contrast_5_50", f"{e2}_contrast_5_50"
            v1, v2 = fv.get(k1, np.nan), fv.get(k2, np.nan)
            if not (np.isfinite(v1) and np.isfinite(v2)):
                continue
            total_s += w * pair_d[(e1, e2)] * v1 * v2
            total_w += w
        return total_s / total_w if total_w > 1e-9 else None

    def score_node(self, sample) -> Optional[NodeScore]:
        if not self._fitted:
            return None
        sub_scores: list[float] = []
        used_weights: dict[str, float] = {}
        all_z: dict[str, float] = {}
        per_src_z: dict[str, dict[str, float]] = {}
        per_src_w: dict[str, dict[str, float]] = {}
        for sn, state in self._pair_states.items():
            sw = self._source_weights.get(sn, 0.0)
            if sw < 1e-9:
                continue
            src = DataSourceRegistry.get(sn)
            fv  = src.get_features(sample.id)
            if fv is None:
                continue
            raw = self._score_raw(fv, state["w"], state["d"])
            if raw is None:
                continue
            sign = -1.0 if self._invert_score else 1.0
            raw *= sign
            sub_scores.append(float(1.0 / (1.0 + np.exp(-raw * _SIGMOID_SCALE_CORR))))
            used_weights[sn] = sw
            z_here: dict[str, float] = {}
            w_here: dict[str, float] = {}
            for (e1, e2), w in state["w"].items():
                if w < 1e-9:
                    continue
                k1, k2 = f"{e1}_contrast_5_50", f"{e2}_contrast_5_50"
                v1, v2 = fv.get(k1, np.nan), fv.get(k2, np.nan)
                if np.isfinite(v1) and np.isfinite(v2):
                    key = f"{e1}_{e2}"
                    z_here[key] = float(sign * state["d"][(e1, e2)] * v1 * v2)
                    w_here[key] = float(w)
                    all_z[f"{sn}:{key}"] = z_here[key]
            per_src_z[sn] = z_here
            per_src_w[sn] = w_here
        if not sub_scores:
            return None
        w_arr = np.array(list(used_weights.values()))
        score = float(np.average(sub_scores, weights=w_arr / w_arr.sum()))
        return NodeScore(
            name=self.name, score=score,
            weights=dict(used_weights),
            feature_z=all_z,
            feature_w=_chain_feature_w(used_weights, per_src_z, per_src_w),
        )

    def score(self, sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None

    def knowledge_entry(self) -> dict:
        """Top co-enrichment pairs aggregated across sources."""
        agg: dict[tuple, list] = {}
        for state in self._pair_states.values():
            for pair, w in state["w"].items():
                agg.setdefault(pair, []).append(w)
        ranked = sorted(
            [(np.mean(ws), p, np.mean([self._pair_states[sn]["d"].get(p, 1.0)
                                       for sn in self._pair_states]))
             for p, ws in agg.items() if np.mean(ws) > 1e-9],
            reverse=True,
        )
        return {
            "expert": self.name,
            "top_pairs": [
                {"pair": f"{p[0]}–{p[1]}", "avg_weight": round(w, 3),
                 "pattern": "co-enriched" if d > 0 else "anti-correlated"}
                for w, p, d in ranked[:6]
            ],
        }
