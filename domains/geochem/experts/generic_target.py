"""Generic target-element experts parameterised by commodity and data source.

Two expert types, each representing a distinct analytical perspective:

  TargetEnrichmentExpert
    — Is the target element locally enriched above the regional background?
    — Features: 5/10 km contrast, fraction above threshold, p90 at two scales
    — Weights: equal across all features (all enrichment aspects matter equally)
    — Direction: learned from data (is higher always better?)

  TargetPathfinderExpert
    — Are geologically expected pathfinder elements anomalous?
    — Weights: pathfinder prior weights from GeochemTargetConfig (domain knowledge)
    — Direction: learned from data per feature

Design principle: each Expert uses domain knowledge for *which features* to weight
and *how much*, while data tells it the *direction* (some elements deplete near
deposits, e.g. pH-sensitive). AUC-based ranking across experts is left entirely
to CompositeNode — not duplicated inside each expert.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.data_source import DataSourceRegistry
from core.expert_tree import NodeScore
from core.primitives import Sample

logger = logging.getLogger(__name__)

_ENRICHMENT_STATS = [
    "contrast_5_50", "contrast_10_50",
    "5km_frac_above", "10km_frac_above",
    "5km_p90", "10km_p90",
]


def _data_direction(feat: str, pos_rows: list[dict], bg_rows: list[dict]) -> float:
    """Return +1 if positives tend to have higher values, -1 otherwise."""
    pos_vals = np.array([fv.get(feat, np.nan) for fv in pos_rows], dtype=float)
    bg_vals  = np.array([fv.get(feat, np.nan) for fv in bg_rows],  dtype=float)
    pos_fin  = pos_vals[np.isfinite(pos_vals)]
    bg_fin   = bg_vals[np.isfinite(bg_vals)]
    if len(pos_fin) > 0 and len(bg_fin) > 0:
        return 1.0 if np.median(pos_fin) >= np.median(bg_fin) else -1.0
    return 1.0


def _bg_stats(features: list[str], rows: list[dict]) -> tuple[dict, dict]:
    mean, std = {}, {}
    for f in features:
        vals = np.array([fv.get(f, np.nan) for fv in rows])
        finite = vals[np.isfinite(vals)]
        mean[f] = float(np.mean(finite))   if len(finite) > 0 else 0.0
        std[f]  = float(np.std(finite))    if len(finite) > 1 else 1.0
        if std[f] < 1e-9:
            std[f] = 1.0
    return mean, std


def _weighted_sigmoid(
    fv: dict,
    features: list[str],
    bg_mean: dict,
    bg_std:  dict,
    shape_w: dict,
    direction: dict,
    invert: bool = False,
) -> Optional[tuple[float, dict, dict]]:
    total_s = total_w = 0.0
    z_scores: dict[str, float] = {}
    for f in features:
        v = fv.get(f, float("nan"))
        if not np.isfinite(v):
            continue
        w = shape_w.get(f, 0.0)
        d = direction.get(f, 1.0)
        z = (v - bg_mean.get(f, 0.0)) / bg_std.get(f, 1.0)
        z_scores[f] = float(z * d)
        total_s += w * d * z
        total_w += w
    if total_w < 1e-9:
        return None
    raw = total_s / total_w
    if invert:
        raw = -raw
        # keep z signs consistent with the score they explain: raw = Σ w_norm · z
        z_scores = {f: -z for f, z in z_scores.items()}
    # normalized over the features actually present, so Σ feature_w · z == raw
    present_w = sum(shape_w.get(f, 0.0) for f in z_scores)
    feature_w = ({f: shape_w.get(f, 0.0) / present_w for f in z_scores}
                 if present_w > 1e-9 else {})
    return float(1.0 / (1.0 + np.exp(-raw))), feature_w, z_scores


# ── TargetEnrichmentExpert ────────────────────────────────────────────────────

class TargetEnrichmentExpert:
    """Is the target element locally enriched above the regional background?

    Uses six enrichment statistics at two scales (5 km and 10 km).
    All features are weighted equally; direction is learned from training data.
    """

    _invert_score: bool = False

    def __init__(self, target: str, source_name: str) -> None:
        self.name = f"{target.lower()}_{source_name}_enrichment"
        self.required_sources: list[str] = [source_name]
        self._target = target
        self._source_name = source_name
        self._features = [f"{target}_{s}" for s in _ENRICHMENT_STATS]
        eq = 1.0 / len(self._features)
        self._shape_w:   dict[str, float] = {f: eq for f in self._features}
        self._direction: dict[str, float] = {f: 1.0 for f in self._features}
        self._bg_mean:   dict[str, float] = {}
        self._bg_std:    dict[str, float] = {}
        self._fitted = False

    def fit(self, positives: list, background: list) -> None:
        src = DataSourceRegistry.get(self._source_name)
        pos_rows = [fv for s in positives  if (fv := src.get_features(s.id)) is not None]
        bg_rows  = [fv for s in background if (fv := src.get_features(s.id)) is not None]
        if not bg_rows:
            return
        self._bg_mean, self._bg_std = _bg_stats(self._features, bg_rows)
        # Equal weights; direction from data (e.g. depletion halos invert sign)
        eq = 1.0 / len(self._features)
        for f in self._features:
            self._shape_w[f]   = eq
            self._direction[f] = _data_direction(f, pos_rows, bg_rows)
        self._fitted = True

    def fit_kb(self, background: list[Sample]) -> None:
        src = DataSourceRegistry.get(self._source_name)
        bg_rows = [fv for s in background if (fv := src.get_features(s.id)) is not None]
        self._bg_mean, self._bg_std = _bg_stats(self._features, bg_rows)
        eq = 1.0 / len(self._features)
        for f in self._features:
            self._shape_w[f]   = eq
            self._direction[f] = 1.0
        self._fitted = True

    def score_node(self, sample) -> Optional[NodeScore]:
        if not self._fitted:
            return None
        src = DataSourceRegistry.get(self._source_name)
        fv  = src.get_features(sample.id)
        if fv is None:
            return None
        result = _weighted_sigmoid(fv, self._features, self._bg_mean, self._bg_std,
                                   self._shape_w, self._direction, self._invert_score)
        if result is None:
            return None
        sc, feature_w, z_scores = result
        return NodeScore(name=self.name, score=sc, weights=dict(feature_w),
                         feature_z=z_scores, feature_w=feature_w)

    def score(self, sample: Sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None


# ── TargetPathfinderExpert ────────────────────────────────────────────────────

class TargetPathfinderExpert:
    """Are geologically expected pathfinder elements anomalous?

    Feature weights come from the pathfinder prior dict in TargetConfig
    (domain knowledge: e.g. Au=1.5 for Cu means Au is a strong Cu pathfinder).
    Direction is learned from training data.
    """

    _invert_score: bool = False

    def __init__(
        self,
        target: str,
        source_name: str,
        pathfinders: dict[str, float],
        ratio_features: list[str] | None = None,
    ) -> None:
        self.name = f"{target.lower()}_{source_name}_pathfinder"
        self.required_sources: list[str] = [source_name]
        self._source_name = source_name
        self._pathfinders = pathfinders
        self._ratio_features = ratio_features or []
        self._feature_keys: list[str] = []
        self._shape_w:   dict[str, float] = {}
        self._direction: dict[str, float] = {}
        self._bg_mean:   dict[str, float] = {}
        self._bg_std:    dict[str, float] = {}
        self._fitted = False

    _PF_STATS = ["contrast_5_50", "contrast_10_50", "5km_frac_above", "10km_frac_above"]

    def _resolve_features(self, sample_fv: dict) -> list[str]:
        keys = []
        for elem in self._pathfinders:
            for stat in self._PF_STATS:
                fname = f"{elem}_{stat}"
                if fname in sample_fv:
                    keys.append(fname)
        for rf in self._ratio_features:
            if rf in sample_fv:
                keys.append(rf)
        return keys

    def fit(self, positives: list, background: list) -> None:
        src = DataSourceRegistry.get(self._source_name)
        pos_rows = [fv for s in positives  if (fv := src.get_features(s.id)) is not None]
        bg_rows  = [fv for s in background if (fv := src.get_features(s.id)) is not None]
        if not bg_rows:
            return
        self._feature_keys = self._resolve_features(bg_rows[0])
        self._bg_mean, self._bg_std = _bg_stats(self._feature_keys, bg_rows)
        # Pathfinder prior weights (domain knowledge); direction from data
        for elem, prior_w in self._pathfinders.items():
            for stat in self._PF_STATS:
                fname = f"{elem}_{stat}"
                if fname in self._feature_keys:
                    self._shape_w[fname]   = prior_w
                    self._direction[fname] = _data_direction(fname, pos_rows, bg_rows)
        for rf in self._ratio_features:
            if rf in self._feature_keys:
                self._shape_w[rf]   = 1.0
                self._direction[rf] = _data_direction(rf, pos_rows, bg_rows)
        self._fitted = True

    def fit_kb(self, background: list[Sample]) -> None:
        src = DataSourceRegistry.get(self._source_name)
        bg_rows = [fv for s in background if (fv := src.get_features(s.id)) is not None]
        if not bg_rows:
            return
        self._feature_keys = self._resolve_features(bg_rows[0])
        self._bg_mean, self._bg_std = _bg_stats(self._feature_keys, bg_rows)
        for elem, w in self._pathfinders.items():
            for stat in self._PF_STATS:
                fname = f"{elem}_{stat}"
                if fname in self._feature_keys:
                    self._shape_w[fname]   = w
                    self._direction[fname] = 1.0
        for rf in self._ratio_features:
            if rf in self._feature_keys:
                self._shape_w[rf]   = 1.0
                self._direction[rf] = 1.0
        self._fitted = True

    def score_node(self, sample) -> Optional[NodeScore]:
        if not self._fitted or not self._feature_keys:
            return None
        src = DataSourceRegistry.get(self._source_name)
        fv  = src.get_features(sample.id)
        if fv is None:
            return None
        result = _weighted_sigmoid(fv, self._feature_keys, self._bg_mean, self._bg_std,
                                   self._shape_w, self._direction, self._invert_score)
        if result is None:
            return None
        sc, feature_w, z_scores = result
        return NodeScore(name=self.name, score=sc, weights=dict(feature_w),
                         feature_z=z_scores, feature_w=feature_w)

    def score(self, sample: Sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None
