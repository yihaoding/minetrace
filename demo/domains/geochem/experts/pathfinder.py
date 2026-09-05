"""PathfinderExpert — L1 Expert based on Cu pathfinder element enrichment.

Hypothesis: Cu deposits are accompanied by characteristic pathfinder elements
(Au, Mo, Ag for porphyry; Pb, Zn, Co for others). Elevated pathfinder
enrichment relative to background is a strong anomaly indicator.

Score uses the 10km/50km contrast and frac_above features for Au, Mo, Ag.
"""
from __future__ import annotations

import logging

import numpy as np

from core.data_source import DataSourceRegistry
from core.primitives import Sample

logger = logging.getLogger(__name__)

# Pathfinders with their weights (geochemical relevance to Cu systems)
_PATHFINDER_WEIGHTS: dict[str, float] = {
    "Au": 2.0,   # strongest Cu indicator (esp. porphyry)
    "Mo": 1.5,   # diagnostic of porphyry Cu
    "Ag": 1.0,   # broad Cu indicator
    "Co": 0.8,   # relevant for sediment-hosted Cu
    "Bi": 0.7,
}

_STATS = ["contrast_5_50", "contrast_10_50", "5km_frac_above", "10km_frac_above"]


class PathfinderExpert:
    """Scores samples by combined pathfinder element enrichment."""

    name: str = "pathfinder"
    required_sources: list[str] = ["sediment"]

    def __init__(self) -> None:
        self._bg_mean: dict[str, float] = {}
        self._bg_std: dict[str, float] = {}
        self._feature_list: list[tuple[str, float]] = []  # (feature_name, weight)
        self._fitted = False

    def fit_kb(self, background_samples: list[Sample]) -> None:
        src = DataSourceRegistry.get("sediment")
        self._feature_list = [
            (f"{elem}_{stat}", w)
            for elem, w in _PATHFINDER_WEIGHTS.items()
            for stat in _STATS
        ]
        values: dict[str, list[float]] = {f: [] for f, _ in self._feature_list}
        for s in background_samples:
            fv = src.get_features(s.id)
            if fv is None:
                continue
            for f, _ in self._feature_list:
                v = fv.get(f, np.nan)
                if np.isfinite(v):
                    values[f].append(v)
        for f, _ in self._feature_list:
            arr = np.array(values[f])
            self._bg_mean[f] = float(arr.mean()) if len(arr) > 0 else 0.0
            self._bg_std[f] = float(arr.std()) if len(arr) > 1 else 1.0
            if self._bg_std[f] < 1e-9:
                self._bg_std[f] = 1.0
        self._fitted = True

    def score(self, sample: Sample) -> float | None:
        if not self._fitted:
            return None
        src = DataSourceRegistry.get("sediment")
        fv = src.get_features(sample.id)
        if fv is None:
            return None
        weighted_z, total_w = 0.0, 0.0
        for f, w in self._feature_list:
            v = fv.get(f, np.nan)
            if np.isfinite(v):
                z = (v - self._bg_mean[f]) / self._bg_std[f]
                weighted_z += w * z
                total_w += w
        if total_w < 1e-9:
            return None
        raw_z = weighted_z / total_w
        return float(1.0 / (1.0 + np.exp(-raw_z)))
