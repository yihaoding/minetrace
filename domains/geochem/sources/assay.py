"""AssaySource — unified DataSource for all geochemical assay CSV types.

Backed by any of the 5 GSWA assay files (sediment, rockchips, drillhole,
shallowdrill, soil).  All 5 share the same column schema so the same
spatial-aggregation logic applies.

Feature groups per target point:
  1. Per-element per-scale statistics (median, p90, CV in log-space,
     fraction above regional threshold)
  2. Multi-scale contrast (log ratio 5km / 50km, 10km / 50km)
  3. Element ratios (selected pairs present in elements list)
  4. Spatial coherence of top samples within 10 km
  5. Global PCA scores (top-5 PCs, median in 5 km and 10 km nbhd)
  6. Sample coverage at each scale

All feature computation is done at registration time for a fixed set of
target samples, so get_features() is O(1) dict lookup.

Key difference from SedimentSource:
  - ``sample_type_filter`` parameter: if set (e.g. "DRILL"), only rows
    where SAMPLETYPE == that value are loaded.
  - ``has_coverage(x, y)`` helper for quick coverage checks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ELEMENTS = [
    "Cu", "Au", "Mo", "Ag", "Pb", "Zn", "Co", "Bi",
    "Ni", "W", "As", "Sn", "Cr", "Sb",
]
SCALES_KM = [5, 10, 50]
N_PCA_COMPONENTS = 5
MIN_SAMPLES_PER_SCALE = 3

CANDIDATE_RATIOS: List[Tuple[str, str]] = [
    ("Cu", "Mo"), ("Cu", "Pb"), ("Au", "Cu"), ("Zn", "Pb"), ("Co", "Ni"),
    ("W", "Sn"), ("W", "Mo"), ("Au", "As"), ("Ni", "Co"), ("Ni", "Cr"),
    ("Sb", "As"), ("Bi", "Sb"),
]

_KM_PER_DEG_LAT = 111.0
_KM_PER_DEG_LON_DEFAULT = 111.0 * np.cos(np.radians(25.0))  # ~100.6 km/°


def _to_km(
    lon: "float | np.ndarray",
    lat: "float | np.ndarray",
    km_per_deg_lon: float = _KM_PER_DEG_LON_DEFAULT,
) -> tuple:
    return lon * km_per_deg_lon, lat * _KM_PER_DEG_LAT


def _safe_log1p(arr: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(arr, 0.0, None))


def _cv(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return np.nan
    mu = np.mean(arr)
    if abs(mu) < 1e-12:
        return np.nan
    return float(np.std(arr) / abs(mu))


# ── main class ────────────────────────────────────────────────────────────────

class AssaySource:
    """Geochemical feature source backed by any GSWA assay CSV.

    Parameters
    ----------
    csv_path           : path to assay CSV
    target_samples     : optional list of GeochemSample for pre-computation
    name               : registry key, e.g. "sediment", "drillhole"
    elements           : element symbols to include; defaults to DEFAULT_ELEMENTS
    sample_type_filter : if set, only rows with SAMPLETYPE == this value are loaded
                         (e.g. "DRILL", "SED", "ROCK", "SHALL", "SOIL")
    center_lat         : reference latitude for lon→km conversion
    """

    def __init__(
        self,
        csv_path: "str | Path",
        target_samples: Optional[list] = None,
        name: str = "assay",
        elements: Optional[List[str]] = None,
        sample_type_filter: Optional[str] = None,
        center_lat: float = -25.0,
    ) -> None:
        self.name = name
        self._km_per_deg_lon = 111.0 * np.cos(np.radians(abs(center_lat)))
        self._elements: List[str] = elements if elements is not None else list(DEFAULT_ELEMENTS)
        self._element_cols: List[str] = [f"{e}_ppm" for e in self._elements]
        self._ratios: List[Tuple[str, str]] = [
            (a, b) for a, b in CANDIDATE_RATIOS
            if a in self._elements and b in self._elements
        ]
        self._features: dict = {}
        self._load_assay(csv_path, sample_type_filter)
        self._fit_pca()
        self._compute_thresholds()
        if target_samples:
            self.register_samples(target_samples)

    # ── DataSource protocol ────────────────────────────────────────────────

    @property
    def feature_names(self) -> List[str]:
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

    def get_features(self, sample_id: str) -> Optional[dict]:
        return self._features.get(sample_id)

    def is_available_for(self, sample_id: str) -> bool:
        return sample_id in self._features

    # ── registration ──────────────────────────────────────────────────────

    def register_samples(self, samples: list) -> None:
        """Precompute and cache features for a list of GeochemSample objects."""
        logger.info("%s: computing features for %d target points …", self.name, len(samples))
        for s in samples:
            if s.id not in self._features:
                self._features[s.id] = self._compute(s.x, s.y)
        logger.info("%s: %d points registered.", self.name, len(self._features))

    def compute_for_point(self, x: float, y: float) -> dict:
        """On-demand feature computation for an arbitrary coordinate."""
        return self._compute(x, y)

    def has_coverage(
        self,
        x: float,
        y: float,
        min_samples: int = 3,
        radius_km: float = 10.0,
    ) -> bool:
        """Return True if there are at least min_samples within radius_km."""
        idxs = self._query_nbhd(x, y, radius_km)
        return len(idxs) >= min_samples

    # ── internal ──────────────────────────────────────────────────────────

    def _load_assay(self, csv_path: "str | Path", sample_type_filter: Optional[str]) -> None:
        logger.info("Loading %s data from %s …", self.name, csv_path)
        # peek at columns
        header_df = pd.read_csv(csv_path, nrows=0)
        all_cols = list(header_df.columns)
        needed = ["X", "Y"] + self._element_cols
        if "SAMPLETYPE" in all_cols:
            needed = ["X", "Y", "SAMPLETYPE"] + self._element_cols
        present = [c for c in needed if c in all_cols]
        df = pd.read_csv(csv_path, usecols=present)
        df = df.replace(-9999, np.nan)

        if sample_type_filter is not None and "SAMPLETYPE" in df.columns:
            before = len(df)
            df = df[df["SAMPLETYPE"] == sample_type_filter].copy()
            logger.info(
                "%s: filtered to SAMPLETYPE='%s': %d / %d rows",
                self.name, sample_type_filter, len(df), before,
            )

        for col in self._element_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        self._assay_xy = df[["X", "Y"]].values.astype(float)
        self._assay_vals: dict = {}
        for col in self._element_cols:
            elem = col.replace("_ppm", "")
            self._assay_vals[elem] = (
                df[col].values.astype(float)
                if col in df.columns
                else np.full(len(df), np.nan)
            )

        x_km, y_km = _to_km(
            self._assay_xy[:, 0], self._assay_xy[:, 1], self._km_per_deg_lon
        )
        self._assay_xy_km = np.column_stack([x_km, y_km])
        self._tree = cKDTree(self._assay_xy_km)
        logger.info("%s: KD-tree built on %d samples.", self.name, len(df))

    def _fit_pca(self) -> None:
        mat = np.column_stack([_safe_log1p(self._assay_vals[e]) for e in self._elements])
        # keep only columns with at least 10% valid values
        col_valid_frac = np.mean(np.isfinite(mat), axis=0)
        usable = col_valid_frac >= 0.10
        if usable.sum() < N_PCA_COMPONENTS + 1:
            logger.warning(
                "%s: only %d element columns have ≥10%% data; skipping PCA.",
                self.name, int(usable.sum()),
            )
            self._pca = None
            self._pca_scaler = None
            self._pc_scores = np.full((len(self._assay_xy), N_PCA_COMPONENTS), np.nan)
            return
        mat_sub = mat[:, usable].copy()
        # median-impute missing values per column
        col_medians = np.nanmedian(mat_sub, axis=0)
        for j in range(mat_sub.shape[1]):
            bad = ~np.isfinite(mat_sub[:, j])
            if bad.any():
                mat_sub[bad, j] = col_medians[j]
        self._pca_scaler = StandardScaler()
        mat_scaled = self._pca_scaler.fit_transform(mat_sub)
        n_comp = min(N_PCA_COMPONENTS, int(usable.sum()))
        self._pca = PCA(n_components=n_comp, random_state=42)
        self._pca.fit(mat_scaled)
        self._pc_scores = np.full((len(self._assay_xy), N_PCA_COMPONENTS), np.nan)
        self._pc_scores[:, :n_comp] = self._pca.transform(mat_scaled)
        logger.info(
            "%s: PCA fitted on %d/%d elements (explained variance: %s).",
            self.name, int(usable.sum()), len(self._elements),
            np.round(self._pca.explained_variance_ratio_, 3),
        )

    def _compute_thresholds(self) -> None:
        self._thresholds: dict = {}
        for elem in self._elements:
            vals = _safe_log1p(self._assay_vals[elem])
            finite = vals[np.isfinite(vals)]
            self._thresholds[elem] = float(np.percentile(finite, 75)) if len(finite) > 0 else 0.0

    def _query_nbhd(self, x: float, y: float, radius_km: float) -> np.ndarray:
        x_km, y_km = _to_km(x, y, self._km_per_deg_lon)
        qpt = np.array([[x_km, y_km]])
        idxs = self._tree.query_ball_point(qpt, r=radius_km)[0]
        return np.array(idxs, dtype=int)

    def _elem_stats(self, elem: str, idxs: np.ndarray) -> dict:
        raw = self._assay_vals[elem][idxs]
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
        if len(idxs) < 3:
            return np.nan
        primary = self._assay_vals[self._elements[0]][idxs]
        valid_mask = np.isfinite(primary)
        if valid_mask.sum() < 3:
            return np.nan
        top_k = np.argsort(primary[valid_mask])[-min(5, valid_mask.sum()):]
        coords = self._assay_xy[idxs[valid_mask][top_k]]
        if len(coords) < 2:
            return np.nan
        dists = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dists.append(np.linalg.norm(coords[i] - coords[j]))
        return float(np.mean(dists)) if dists else np.nan

    def _pc_stats(self, idxs: np.ndarray) -> np.ndarray:
        if self._pca is None or len(idxs) == 0:
            return np.full(N_PCA_COMPONENTS, np.nan)
        scores = self._pc_scores[idxs]
        finite_rows = scores[np.all(np.isfinite(scores), axis=1)]
        if len(finite_rows) == 0:
            return np.full(N_PCA_COMPONENTS, np.nan)
        return np.median(finite_rows, axis=0)

    def _compute(self, x: float, y: float) -> dict:
        """Compute all features for one target coordinate."""
        feat: dict = {}

        nbhd: dict = {}
        scale_stats: dict = {}
        for scale in SCALES_KM:
            idxs = self._query_nbhd(x, y, scale)
            nbhd[scale] = idxs
            scale_stats[scale] = {elem: self._elem_stats(elem, idxs) for elem in self._elements}
            feat[f"n_samples_{scale}km"] = float(len(idxs))
            for elem in self._elements:
                for stat, val in scale_stats[scale][elem].items():
                    feat[f"{elem}_{scale}km_{stat}"] = val

        def _contrast(elem: str, s_local: int, s_bg: int) -> float:
            m_local = scale_stats[s_local][elem]["median"]
            m_bg = scale_stats[s_bg][elem]["median"]
            if np.isfinite(m_local) and np.isfinite(m_bg) and abs(m_bg) > 1e-9:
                return float(m_local - m_bg)
            return np.nan

        for elem in self._elements:
            feat[f"{elem}_contrast_5_50"] = _contrast(elem, 5, 50)
            feat[f"{elem}_contrast_10_50"] = _contrast(elem, 10, 50)

        def _log_ratio(e1: str, e2: str, scale: int = 10) -> float:
            a = scale_stats[scale][e1]["median"]
            b = scale_stats[scale][e2]["median"]
            if np.isfinite(a) and np.isfinite(b):
                return float(a - b)
            return np.nan

        for a, b in self._ratios:
            feat[f"ratio_{a}_{b}"] = _log_ratio(a, b)

        feat["spatial_coherence_10km"] = self._spatial_coherence(nbhd[10])

        for scale in [5, 10]:
            pc_med = self._pc_stats(nbhd[scale])
            for i, val in enumerate(pc_med):
                feat[f"PC{i+1}_{scale}km_median"] = float(val)

        return feat
