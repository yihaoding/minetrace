"""SedimentSource — DataSource backed by real geochemical assay data.

For any target point (known mine site or random background coordinate),
computes geochemical features by aggregating samples in the neighbourhood
at three spatial scales.  Works for any assay CSV (sediment or rockchips)
by passing a custom ``name`` and ``elements`` list.

Feature groups per target point:
  1. Per-element per-scale statistics (median, p90, CV in log-space,
     fraction above regional threshold)
  2. Multi-scale contrast (log ratio 5km / 50km, 10km / 50km)
  3. Element ratios (selected pairs present in elements list)
  4. Spatial coherence of top samples within 10 km
  5. Global PCA scores (top-5 PCs, median in 5 km and 10 km nbhd)
  6. Sample coverage at each scale

All feature computation is done at construction time for a fixed set of
target samples, so get_features() is O(1) dict lookup.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ELEMENTS = ["Cu", "Au", "Mo", "Ag", "Pb", "Zn", "Co", "Bi", "Ni", "W", "As", "Sn", "Cr", "Sb"]
SCALES_KM = [5, 10, 50]
N_PCA_COMPONENTS = 5
MIN_SAMPLES_PER_SCALE = 3

# Candidate element-ratio pairs; only those with both elements present are computed.
CANDIDATE_RATIOS: list[tuple[str, str]] = [
    ("Cu", "Mo"), ("Cu", "Pb"), ("Au", "Cu"), ("Zn", "Pb"), ("Co", "Ni"),
    ("W", "Sn"), ("W", "Mo"), ("Au", "As"), ("Ni", "Co"), ("Ni", "Cr"),
    ("Sb", "As"), ("Bi", "Sb"),
]

# Default lat/lon → km conversion constants (WA centre lat ≈ -25°).
# Override via SedimentSource(center_lat=...) for other regions.
_KM_PER_DEG_LAT = 111.0
_KM_PER_DEG_LON_DEFAULT = 111.0 * np.cos(np.radians(25.0))  # ≈ 100.6 km/°


def _to_km(
    lon: float | np.ndarray,
    lat: float | np.ndarray,
    km_per_deg_lon: float = _KM_PER_DEG_LON_DEFAULT,
) -> tuple:
    """Convert (lon, lat) to (x_km, y_km) in an approximate Euclidean space."""
    return lon * km_per_deg_lon, lat * _KM_PER_DEG_LAT


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_log1p(arr: np.ndarray) -> np.ndarray:
    """log1p with clipping of negatives (measurement artefacts)."""
    return np.log1p(np.clip(arr, 0.0, None))


def _cv(arr: np.ndarray) -> float:
    """Coefficient of variation; returns NaN for empty or zero-mean arrays."""
    if len(arr) < 2:
        return np.nan
    mu = np.mean(arr)
    if abs(mu) < 1e-12:
        return np.nan
    return float(np.std(arr) / abs(mu))


# ── main class ────────────────────────────────────────────────────────────────

class SedimentSource:
    """Geochemical feature source backed by any assay CSV (sediment or rockchips).

    Parameters
    ----------
    csv_path       : path to assay CSV (sediment or rockchips)
    target_samples : GeochemSample list for pre-computation (optional)
    name           : registry key, e.g. "sediment" or "rockchips"
    elements       : element symbols to include; defaults to DEFAULT_ELEMENTS
    """

    def __init__(
        self,
        csv_path: str | Path,
        target_samples: list | None = None,
        name: str = "sediment",
        elements: list[str] | None = None,
        center_lat: float = -25.0,
    ) -> None:
        self.name = name
        self._km_per_deg_lon = 111.0 * np.cos(np.radians(abs(center_lat)))
        self._elements: list[str] = elements if elements is not None else list(DEFAULT_ELEMENTS)
        self._element_cols: list[str] = [f"{e}_ppm" for e in self._elements]
        # active ratios: only pairs where both elements are present
        self._ratios: list[tuple[str, str]] = [
            (a, b) for a, b in CANDIDATE_RATIOS if a in self._elements and b in self._elements
        ]
        self._features: dict[str, dict[str, float]] = {}
        self._load_sediment(csv_path)
        self._fit_pca()
        self._compute_thresholds()
        if target_samples:
            self.register_samples(target_samples)

    # ── DataSource protocol ────────────────────────────────────────────────

    @property
    def feature_names(self) -> list[str]:
        names = []
        for scale in SCALES_KM:
            for elem in self._elements:
                for stat in ("median", "p90", "cv", "frac_above"):
                    names.append(f"{elem}_{scale}km_{stat}")
        for elem in self._elements:
            names.append(f"{elem}_contrast_5_50")
            names.append(f"{elem}_contrast_10_50")
        for a, b in self._ratios:
            names.append(f"ratio_{a}_{b}")
        names.append("spatial_coherence_10km")
        for scale in [5, 10]:
            for pc in range(N_PCA_COMPONENTS):
                names.append(f"PC{pc+1}_{scale}km_median")
        for scale in SCALES_KM:
            names.append(f"n_samples_{scale}km")
        return names

    def get_features(self, sample_id: str) -> dict[str, float] | None:
        return self._features.get(sample_id)

    def is_available_for(self, sample_id: str) -> bool:
        return sample_id in self._features

    # ── registration ──────────────────────────────────────────────────────

    def register_samples(self, samples: list) -> None:
        """Precompute and cache features for a list of GeochemSample objects."""
        logger.info("Computing sediment features for %d target points …", len(samples))
        for s in samples:
            if s.id not in self._features:
                self._features[s.id] = self._compute(s.x, s.y)
        logger.info("Done. %d points registered.", len(self._features))

    def compute_for_point(self, x: float, y: float) -> dict[str, float]:
        """On-demand feature computation for an arbitrary coordinate."""
        return self._compute(x, y)

    # ── internal ──────────────────────────────────────────────────────────

    def _load_sediment(self, csv_path: str | Path) -> None:
        logger.info("Loading %s data from %s …", self.name, csv_path)
        needed_cols = ["X", "Y"] + self._element_cols
        present = [c for c in needed_cols if c in pd.read_csv(csv_path, nrows=0).columns]
        df = pd.read_csv(csv_path, usecols=present)
        df = df.replace(-9999, np.nan)
        for col in self._element_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        self._sed_xy = df[["X", "Y"]].values.astype(float)
        self._sed_vals: dict[str, np.ndarray] = {}
        for col in self._element_cols:
            elem = col.replace("_ppm", "")
            self._sed_vals[elem] = df[col].values.astype(float) if col in df.columns else np.full(len(df), np.nan)

        x_km, y_km = _to_km(self._sed_xy[:, 0], self._sed_xy[:, 1], self._km_per_deg_lon)
        self._sed_xy_km = np.column_stack([x_km, y_km])
        self._tree = cKDTree(self._sed_xy_km)
        logger.info("%s KD-tree built on %d samples.", self.name, len(df))

    def _fit_pca(self) -> None:
        mat = np.column_stack([_safe_log1p(self._sed_vals[e]) for e in self._elements])
        mask = np.all(np.isfinite(mat), axis=1)
        if mask.sum() < N_PCA_COMPONENTS + 1:
            logger.warning("Too few complete rows for PCA; skipping.")
            self._pca = None
            self._pca_scaler = None
            return
        self._pca_scaler = StandardScaler()
        mat_scaled = self._pca_scaler.fit_transform(mat[mask])
        self._pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
        self._pca.fit(mat_scaled)
        mat_all = np.column_stack([_safe_log1p(self._sed_vals[e]) for e in self._elements])
        finite_mask = np.all(np.isfinite(mat_all), axis=1)
        self._pc_scores = np.full((len(self._sed_xy), N_PCA_COMPONENTS), np.nan)
        if finite_mask.any():
            self._pc_scores[finite_mask] = self._pca.transform(
                self._pca_scaler.transform(mat_all[finite_mask])
            )
        logger.info("Global PCA fitted (explained variance: %s).",
                    np.round(self._pca.explained_variance_ratio_, 3))

    def _compute_thresholds(self) -> None:
        self._thresholds: dict[str, float] = {}
        for elem in self._elements:
            vals = _safe_log1p(self._sed_vals[elem])
            finite = vals[np.isfinite(vals)]
            self._thresholds[elem] = float(np.percentile(finite, 75)) if len(finite) > 0 else 0.0

    def _query_nbhd(self, x: float, y: float, radius_km: float) -> np.ndarray:
        """Return indices of sediment samples within radius_km of (x, y)."""
        x_km, y_km = _to_km(x, y, self._km_per_deg_lon)
        qpt = np.array([[x_km, y_km]])
        idxs = self._tree.query_ball_point(qpt, r=radius_km)[0]
        return np.array(idxs, dtype=int)

    def _elem_stats(self, elem: str, idxs: np.ndarray) -> dict[str, float]:
        """Compute per-element statistics for a neighbourhood index set."""
        raw = self._sed_vals[elem][idxs]
        valid = raw[np.isfinite(raw) & (raw >= 0)]
        if len(valid) < MIN_SAMPLES_PER_SCALE:
            return {"median": np.nan, "p90": np.nan, "cv": np.nan, "frac_above": np.nan}
        log_vals = _safe_log1p(valid)
        frac = float(np.mean(log_vals > self._thresholds[elem]))
        return {
            "median": float(np.median(log_vals)),
            "p90": float(np.percentile(log_vals, 90)),
            "cv": _cv(log_vals),
            "frac_above": frac,
        }

    def _spatial_coherence(self, idxs: np.ndarray) -> float:
        """Mean pairwise distance of top-5 samples (by primary element) within neighbourhood."""
        if len(idxs) < 3:
            return np.nan
        primary = self._sed_vals[self._elements[0]][idxs]
        valid_mask = np.isfinite(primary)
        if valid_mask.sum() < 3:
            return np.nan
        top_k = np.argsort(primary[valid_mask])[-min(5, valid_mask.sum()):]
        coords = self._sed_xy[idxs[valid_mask][top_k]]
        if len(coords) < 2:
            return np.nan
        dists = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dists.append(np.linalg.norm(coords[i] - coords[j]))
        return float(np.mean(dists)) if dists else np.nan

    def _pc_stats(self, idxs: np.ndarray) -> np.ndarray:
        """Median PC scores for neighbourhood samples; NaN if PCA unavailable."""
        if self._pca is None or len(idxs) == 0:
            return np.full(N_PCA_COMPONENTS, np.nan)
        scores = self._pc_scores[idxs]
        finite_rows = scores[np.all(np.isfinite(scores), axis=1)]
        if len(finite_rows) == 0:
            return np.full(N_PCA_COMPONENTS, np.nan)
        return np.median(finite_rows, axis=0)

    def _compute(self, x: float, y: float) -> dict[str, float]:
        """Compute all features for one target coordinate."""
        feat: dict[str, float] = {}

        # ── per-element per-scale stats ──────────────────────────────────
        nbhd: dict[int, np.ndarray] = {}
        scale_stats: dict[int, dict[str, dict[str, float]]] = {}
        for scale in SCALES_KM:
            idxs = self._query_nbhd(x, y, scale)
            nbhd[scale] = idxs
            scale_stats[scale] = {elem: self._elem_stats(elem, idxs) for elem in self._elements}
            feat[f"n_samples_{scale}km"] = float(len(idxs))
            for elem in self._elements:
                for stat, val in scale_stats[scale][elem].items():
                    feat[f"{elem}_{scale}km_{stat}"] = val

        # ── multi-scale contrast ──────────────────────────────────────────
        def _contrast(elem: str, s_local: int, s_bg: int) -> float:
            m_local = scale_stats[s_local][elem]["median"]
            m_bg = scale_stats[s_bg][elem]["median"]
            if np.isfinite(m_local) and np.isfinite(m_bg) and abs(m_bg) > 1e-9:
                return float(m_local - m_bg)
            return np.nan

        for elem in self._elements:
            feat[f"{elem}_contrast_5_50"] = _contrast(elem, 5, 50)
            feat[f"{elem}_contrast_10_50"] = _contrast(elem, 10, 50)

        # ── element ratios (log-space at 10 km) ───────────────────────────
        def _log_ratio(e1: str, e2: str, scale: int = 10) -> float:
            a = scale_stats[scale][e1]["median"]
            b = scale_stats[scale][e2]["median"]
            if np.isfinite(a) and np.isfinite(b):
                return float(a - b)
            return np.nan

        for a, b in self._ratios:
            feat[f"ratio_{a}_{b}"] = _log_ratio(a, b)

        # ── spatial coherence at 10 km (5 km often too few points) ───────
        feat["spatial_coherence_10km"] = self._spatial_coherence(nbhd[10])

        # ── PCA scores at 5 km and 10 km ──────────────────────────────────
        for scale in [5, 10]:
            pc_med = self._pc_stats(nbhd[scale])
            for i, val in enumerate(pc_med):
                feat[f"PC{i+1}_{scale}km_median"] = float(val)

        return feat
