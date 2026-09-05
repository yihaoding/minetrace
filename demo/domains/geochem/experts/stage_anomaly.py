"""StageAnomalyExpert — L1 Expert based on SITE_STAGE score.

Hypothesis: Sites at advanced development stages (Operating, Under Development)
in a spatial cluster signal a productive district. The stage_score feature
encodes stage maturity; high scores relative to background are anomalous.

Score = sigmoid(z-score of stage_score relative to background).
"""
from __future__ import annotations

import logging

import numpy as np

from core.data_source import DataSourceRegistry
from core.primitives import Sample

logger = logging.getLogger(__name__)


class StageAnomalyExpert:
    """Scores samples by development stage relative to background."""

    name: str = "stage_anomaly"
    required_sources: list[str] = ["site_catalog"]

    def __init__(self) -> None:
        self._bg_mean: float = 0.0
        self._bg_std: float = 1.0
        self._fitted = False

    def fit_kb(self, background_samples: list[Sample]) -> None:
        src = DataSourceRegistry.get("site_catalog")
        values: list[float] = []
        for s in background_samples:
            fv = src.get_features(s.id)
            if fv is not None and "stage_score" in fv:
                values.append(fv["stage_score"])
        arr = np.array(values)
        self._bg_mean = float(arr.mean()) if len(arr) > 0 else 0.0
        self._bg_std = float(arr.std()) if len(arr) > 1 else 1.0
        if self._bg_std < 1e-9:
            self._bg_std = 1.0
        self._fitted = True

    def score(self, sample: Sample) -> float | None:
        if not self._fitted:
            return None
        src = DataSourceRegistry.get("site_catalog")
        fv = src.get_features(sample.id)
        if fv is None or "stage_score" not in fv:
            return None
        z = (fv["stage_score"] - self._bg_mean) / self._bg_std
        return float(1.0 / (1.0 + np.exp(-z)))
