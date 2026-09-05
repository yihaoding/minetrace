"""Geophysics experts — four leaf nodes for the geophysics data source.

Each expert covers one physical signal type (magnetics, gravity, radiometrics,
geochronology). Within each expert all features are weighted equally; the
*direction* (does higher value → more prospective?) is learned from training data.
AUC-based ranking *across* experts is left to CompositeNode.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.data_source import DataSourceRegistry
from core.expert_tree import NodeScore

logger = logging.getLogger(__name__)

# ── shared base ───────────────────────────────────────────────────────────────

class _GeophysicsExpert:
    """Base for geophysics leaf experts.

    All features within one expert are weighted equally.
    Direction (+1 / −1) is learned from positives vs background medians.
    """

    _invert_score: bool = False

    def __init__(self, name: str, source_name: str, feature_keys: list[str]) -> None:
        self.name = name
        self.required_sources: list[str] = [source_name]
        self._source_name = source_name
        self._feature_keys = feature_keys
        self._bg_mean:   dict[str, float] = {}
        self._bg_std:    dict[str, float] = {}
        self._shape_w:   dict[str, float] = {}
        self._direction: dict[str, float] = {}
        self._fitted = False

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(self, positives: list, background: list) -> None:
        src = DataSourceRegistry.get(self._source_name)
        pos_rows = [fv for s in positives  if (fv := src.get_features(s.id)) is not None]
        bg_rows  = [fv for s in background if (fv := src.get_features(s.id)) is not None]
        if not bg_rows:
            logger.warning("%s: no background samples — skip fit", self.name)
            return

        eq = 1.0 / max(len(self._feature_keys), 1)
        for f in self._feature_keys:
            bg_vals  = np.array([fv.get(f, np.nan) for fv in bg_rows])
            pos_vals = np.array([fv.get(f, np.nan) for fv in pos_rows]) if pos_rows else np.array([])
            finite_bg = bg_vals[np.isfinite(bg_vals)]
            self._bg_mean[f] = float(np.mean(finite_bg))  if len(finite_bg) > 0 else 0.0
            self._bg_std[f]  = float(np.std(finite_bg))   if len(finite_bg) > 1 else 1.0
            if self._bg_std[f] < 1e-9:
                self._bg_std[f] = 1.0
            self._shape_w[f] = eq
            if pos_rows:
                pm = np.nanmedian(pos_vals)
                bm = np.nanmedian(bg_vals)
                self._direction[f] = 1.0 if (np.isfinite(pm) and np.isfinite(bm) and pm >= bm) else -1.0
            else:
                self._direction[f] = 1.0

        self._fitted = True

    def fit_kb(self, background: list) -> None:
        """Legacy path: equal weights, direction=+1."""
        src = DataSourceRegistry.get(self._source_name)
        rows = [fv for s in background if (fv := src.get_features(s.id)) is not None]
        if not rows:
            return
        eq = 1.0 / max(len(self._feature_keys), 1)
        for f in self._feature_keys:
            vals = np.array([fv.get(f, np.nan) for fv in rows])
            finite = vals[np.isfinite(vals)]
            self._bg_mean[f]  = float(np.mean(finite)) if len(finite) > 0 else 0.0
            self._bg_std[f]   = float(np.std(finite))  if len(finite) > 1 else 1.0
            if self._bg_std[f] < 1e-9:
                self._bg_std[f] = 1.0
            self._shape_w[f]   = eq
            self._direction[f] = 1.0
        self._fitted = True

    # ── scoring ───────────────────────────────────────────────────────────────

    def score_node(self, sample) -> Optional[NodeScore]:
        if not self._fitted: return None
        src = DataSourceRegistry.get(self._source_name)
        fv  = src.get_features(sample.id)
        if fv is None: return None

        total_s = total_w = 0.0
        z_scores: dict[str, float] = {}
        for f in self._feature_keys:
            v = fv.get(f, float("nan"))
            if not np.isfinite(v): continue
            w = self._shape_w.get(f, 0.0)
            d = self._direction.get(f, 1.0)
            z = (v - self._bg_mean.get(f, 0.0)) / self._bg_std.get(f, 1.0)
            z_scores[f] = float(z * d)
            total_s += w * d * z
            total_w += w

        if total_w < 1e-9: return None
        raw = total_s / total_w
        if self._invert_score: raw = -raw
        sc = float(1.0 / (1.0 + np.exp(-raw)))
        return NodeScore(
            name=self.name, score=sc,
            weights={f: self._shape_w.get(f, 0.0) for f in self._feature_keys},
            feature_z=z_scores,
        )

    def score(self, sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None


# ── concrete experts ──────────────────────────────────────────────────────────

class MagneticExpert(_GeophysicsExpert):
    """Magnetic anomaly (1VD) value + horizontal gradient magnitude."""

    def __init__(self, source_name: str = "geophysics") -> None:
        super().__init__(
            name=f"magnetic_{source_name}",
            source_name=source_name,
            feature_keys=["mag", "mag_grad"],
        )


class GravityExpert(_GeophysicsExpert):
    """Bouguer gravity anomaly value + gradient."""

    def __init__(self, source_name: str = "geophysics") -> None:
        super().__init__(
            name=f"gravity_{source_name}",
            source_name=source_name,
            feature_keys=["grav", "grav_grad"],
        )


class RadiometricExpert(_GeophysicsExpert):
    """K, Th, U values + gradients + K/Th, Th/U, U/K ratios."""

    def __init__(self, source_name: str = "geophysics") -> None:
        super().__init__(
            name=f"radiometric_{source_name}",
            source_name=source_name,
            feature_keys=["K", "Th", "U", "K_grad", "Th_grad", "U_grad",
                          "K_Th", "Th_U", "U_K"],
        )


class GeochronExpert(_GeophysicsExpert):
    """Lu-Hf and Sm-Nd crustal model ages + gradients."""

    def __init__(self, source_name: str = "geophysics") -> None:
        super().__init__(
            name=f"geochronology_{source_name}",
            source_name=source_name,
            feature_keys=["LuHf", "SmNd", "LuHf_grad", "SmNd_grad"],
        )
