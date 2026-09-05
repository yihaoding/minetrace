"""GeophysicsSource — DataSource backed by geophysical rasters.

For each registered target point, samples 7 rasters and computes:
  - Point values  : mag, grav, K, Th, U, LuHf, SmNd
  - Gradients     : {name}_grad  (magnitude of horizontal gradient, nT/km or mGal/km)
  - Derived ratios: K_Th, Th_U, U_K

Gradients use central differences at ±DELTA_DEG (~1 km); falls back to
one-sided differences at raster edges / nodata boundaries.

All features are cached by sample_id after the first computation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# nodata sentinels used by the GSWA rasters
_NODATA_LARGE = -3.0e+38   # LuHf / SmNd .bil files
_NODATA_SMALL = -99999.0   # all .ers files

# ~1 km in degrees at WA centre latitude (≈ -25°)
_DELTA_DEG = 1.0 / 111.0
_DEG_KM    = 111.0
_LON_KM    = 111.0 * np.cos(np.radians(25.0))   # ≈ 100.6 km/°

# Raster names exposed as features
RASTER_NAMES = ("mag", "grav", "K", "Th", "U", "LuHf", "SmNd")

# Derived ratio pairs  (numerator, denominator, output_name)
RATIO_PAIRS = (
    ("K",  "Th", "K_Th"),
    ("Th", "U",  "Th_U"),
    ("U",  "K",  "U_K"),
)


def _is_valid(v: float) -> bool:
    return (not np.isnan(v)) and (v > _NODATA_LARGE) and (v != _NODATA_SMALL)


class GeophysicsSource:
    """DataSource backed by the 7 GSWA geophysical rasters.

    Parameters
    ----------
    raster_paths : dict mapping raster name → file path
                   Expected keys: mag, grav, K, Th, U, LuHf, SmNd
    name         : registry key (default "geophysics")
    """

    def __init__(
        self,
        raster_paths: dict[str, str | Path],
        name: str = "geophysics",
    ) -> None:
        self.name = name
        self._features: dict[str, dict[str, float]] = {}
        self._raster_paths = {rname: str(p) for rname, p in raster_paths.items()}
        self._open_handles()

    def _open_handles(self) -> None:
        try:
            import rasterio  # type: ignore
        except ImportError as e:
            raise ImportError("rasterio is required for GeophysicsSource") from e

        self._handles: dict[str, "rasterio.DatasetReader"] = {}
        for rname, path in self._raster_paths.items():
            try:
                self._handles[rname] = rasterio.open(path)
            except Exception as exc:
                logger.warning("Cannot open raster %s (%s): %s", rname, path, exc)

        logger.info("GeophysicsSource: opened %d/%d rasters",
                    len(self._handles), len(self._raster_paths))

    # ── pickling: raster handles cannot be serialised; reopen from paths ──────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_handles", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open_handles()

    # ── DataSource protocol ───────────────────────────────────────────────────

    @property
    def feature_names(self) -> list[str]:
        names = []
        for r in RASTER_NAMES:
            names.append(r)
            names.append(f"{r}_grad")
        for _, _, out in RATIO_PAIRS:
            names.append(out)
        return names

    def get_features(self, sample_id: str) -> dict[str, float] | None:
        return self._features.get(sample_id)

    def is_available_for(self, sample_id: str) -> bool:
        return sample_id in self._features

    # ── registration ─────────────────────────────────────────────────────────

    def register_samples(self, samples: list) -> None:
        new = [s for s in samples if s.id not in self._features]
        logger.info("GeophysicsSource: computing features for %d new points", len(new))
        for s in new:
            self._features[s.id] = self._compute(s.x, s.y)

    def compute_for_point(self, x: float, y: float) -> dict[str, float]:
        return self._compute(x, y)

    # ── internal computation ──────────────────────────────────────────────────

    def _sample_raster(self, handle, lon: float, lat: float) -> float:
        try:
            v = float(list(handle.sample([(lon, lat)]))[0][0])
            return v if _is_valid(v) else np.nan
        except Exception:
            return np.nan

    def _compute(self, lon: float, lat: float) -> dict[str, float]:
        feats: dict[str, float] = {}
        raw: dict[str, float] = {}

        for rname, handle in self._handles.items():
            v_c  = self._sample_raster(handle, lon,                lat)
            v_px = self._sample_raster(handle, lon + _DELTA_DEG,   lat)
            v_mx = self._sample_raster(handle, lon - _DELTA_DEG,   lat)
            v_py = self._sample_raster(handle, lon,                 lat + _DELTA_DEG)
            v_my = self._sample_raster(handle, lon,                 lat - _DELTA_DEG)

            raw[rname]   = v_c
            feats[rname] = v_c

            # horizontal gradient (nT/km or mGal/km)
            if _is_valid(v_px) and _is_valid(v_mx):
                dx = (v_px - v_mx) / (2.0 * _DELTA_DEG * _LON_KM)
            elif _is_valid(v_px) and _is_valid(v_c):
                dx = (v_px - v_c) / (_DELTA_DEG * _LON_KM)
            elif _is_valid(v_c) and _is_valid(v_mx):
                dx = (v_c - v_mx) / (_DELTA_DEG * _LON_KM)
            else:
                dx = np.nan

            if _is_valid(v_py) and _is_valid(v_my):
                dy = (v_py - v_my) / (2.0 * _DELTA_DEG * _DEG_KM)
            elif _is_valid(v_py) and _is_valid(v_c):
                dy = (v_py - v_c) / (_DELTA_DEG * _DEG_KM)
            elif _is_valid(v_c) and _is_valid(v_my):
                dy = (v_c - v_my) / (_DELTA_DEG * _DEG_KM)
            else:
                dy = np.nan

            if _is_valid(dx) and _is_valid(dy):
                feats[f"{rname}_grad"] = float(np.sqrt(dx ** 2 + dy ** 2))
            else:
                feats[f"{rname}_grad"] = np.nan

        # derived radiometric ratios
        for num, den, out in RATIO_PAIRS:
            n, d = raw.get(num, np.nan), raw.get(den, np.nan)
            if _is_valid(n) and _is_valid(d) and abs(d) > 1e-12:
                feats[out] = float(n / d)
            else:
                feats[out] = np.nan

        return feats
