"""CommodityAffinityExpert — L1 Expert based on commodity co-occurrence.

Hypothesis: Sites with commodity assemblages similar to known Cu deposits
are more likely to be anomalous.

Score = sigmoid(z-score of co_occurrence_with_cu relative to background).
Also incorporates pathfinder_count and is_polymetallic as secondary signals.
"""
from __future__ import annotations

import logging

import numpy as np

from core.data_source import DataSourceRegistry
from core.primitives import Sample

logger = logging.getLogger(__name__)

_FEATURE_WEIGHTS = {
    "co_occurrence_with_cu": 2.0,
    "pathfinder_count": 1.0,
    "is_polymetallic": 0.5,
}


class CommodityAffinityExpert:
    """Scores samples by commodity similarity to Cu deposit profile."""

    name: str = "commodity_affinity"
    required_sources: list[str] = ["site_catalog"]

    def __init__(self) -> None:
        self._bg_mean: dict[str, float] = {}
        self._bg_std: dict[str, float] = {}
        self._fitted = False

    def fit_kb(self, background_samples: list[Sample]) -> None:
        src = DataSourceRegistry.get("site_catalog")
        feats = list(_FEATURE_WEIGHTS)
        values: dict[str, list[float]] = {f: [] for f in feats}
        for s in background_samples:
            fv = src.get_features(s.id)
            if fv is None:
                continue
            for f in feats:
                if f in fv:
                    values[f].append(fv[f])
        for f in feats:
            arr = np.array(values[f])
            self._bg_mean[f] = float(arr.mean()) if len(arr) > 0 else 0.0
            self._bg_std[f] = float(arr.std()) if len(arr) > 1 else 1.0
            if self._bg_std[f] < 1e-9:
                self._bg_std[f] = 1.0
        self._fitted = True

    def score(self, sample: Sample) -> float | None:
        if not self._fitted:
            return None
        src = DataSourceRegistry.get("site_catalog")
        fv = src.get_features(sample.id)
        if fv is None:
            return None
        weighted_z = 0.0
        total_weight = 0.0
        for f, w in _FEATURE_WEIGHTS.items():
            if f in fv:
                z = (fv[f] - self._bg_mean[f]) / self._bg_std[f]
                weighted_z += w * z
                total_weight += w
        if total_weight < 1e-9:
            return None
        raw_z = weighted_z / total_weight
        return float(1.0 / (1.0 + np.exp(-raw_z)))
