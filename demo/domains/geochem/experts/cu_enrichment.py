"""CuEnrichmentExpert — L1 Expert based on local Cu enrichment vs background.

Hypothesis: Cu deposit sites show elevated Cu concentrations within 10 km
relative to the regional background (50 km).

Score uses the Cu_contrast_10_50 feature (log-ratio local/background) plus
the local fraction of samples above the regional 75th percentile.
"""
from __future__ import annotations

import logging

import numpy as np

from core.data_source import DataSourceRegistry
from core.primitives import Sample

logger = logging.getLogger(__name__)

_FEATURES = ["Cu_contrast_5_50", "Cu_contrast_10_50", "Cu_5km_frac_above", "Cu_10km_frac_above", "Cu_5km_p90", "Cu_10km_p90"]


class CuEnrichmentExpert:
    """Scores samples by Cu enrichment relative to regional background."""

    name: str = "cu_enrichment"
    required_sources: list[str] = ["sediment"]

    def __init__(self) -> None:
        self._bg_mean: dict[str, float] = {}
        self._bg_std: dict[str, float] = {}
        self._fitted = False

    def fit_kb(self, background_samples: list[Sample]) -> None:
        src = DataSourceRegistry.get("sediment")
        values: dict[str, list[float]] = {f: [] for f in _FEATURES}
        for s in background_samples:
            fv = src.get_features(s.id)
            if fv is None:
                continue
            for f in _FEATURES:
                v = fv.get(f, np.nan)
                if np.isfinite(v):
                    values[f].append(v)
        for f in _FEATURES:
            arr = np.array(values[f])
            self._bg_mean[f] = float(arr.mean()) if len(arr) > 0 else 0.0
            self._bg_std[f] = float(arr.std()) if len(arr) > 1 else 1.0
            if self._bg_std[f] < 1e-9:
                self._bg_std[f] = 1.0
        self._fitted = True
        logger.debug("CuEnrichmentExpert fitted on %d background samples", len(background_samples))

    def score(self, sample: Sample) -> float | None:
        if not self._fitted:
            return None
        src = DataSourceRegistry.get("sediment")
        fv = src.get_features(sample.id)
        if fv is None:
            return None
        z_scores = []
        for f in _FEATURES:
            v = fv.get(f, np.nan)
            if np.isfinite(v):
                z = (v - self._bg_mean[f]) / self._bg_std[f]
                z_scores.append(z)
        if not z_scores:
            return None
        raw_z = float(np.mean(z_scores))
        return float(1.0 / (1.0 + np.exp(-raw_z)))
