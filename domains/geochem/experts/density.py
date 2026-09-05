"""DensityExpert — L1 Expert based on local site density.

Hypothesis: Cu deposits cluster spatially. A site in a dense cluster of other
sites (within 5 km and 20 km) is more anomalous than an isolated site.

Score = combined z-score of local_density_5km and local_density_20km relative
to the background distribution.
"""
from __future__ import annotations

import logging

import numpy as np

from core.data_source import DataSourceRegistry
from core.primitives import Sample

logger = logging.getLogger(__name__)


class DensityExpert:
    """Scores samples by local spatial density anomaly."""

    name: str = "density"
    required_sources: list[str] = ["site_catalog"]

    def __init__(self) -> None:
        self._bg_mean: dict[str, float] = {}
        self._bg_std: dict[str, float] = {}
        self._fitted = False

    def fit_kb(self, background_samples: list[Sample]) -> None:
        src = DataSourceRegistry.get("site_catalog")
        feats = ["local_density_5km", "local_density_20km"]
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
        logger.debug("DensityExpert fitted on %d background samples", len(background_samples))

    def score(self, sample: Sample) -> float | None:
        if not self._fitted:
            logger.warning("DensityExpert.score called before fit_kb")
            return None
        src = DataSourceRegistry.get("site_catalog")
        fv = src.get_features(sample.id)
        if fv is None:
            return None
        feats = ["local_density_5km", "local_density_20km"]
        z_scores = []
        for f in feats:
            if f in fv:
                z = (fv[f] - self._bg_mean[f]) / self._bg_std[f]
                z_scores.append(z)
        if not z_scores:
            return None
        raw_z = float(np.mean(z_scores))
        # sigmoid to map z-score → [0, 1]
        return float(1.0 / (1.0 + np.exp(-raw_z)))
