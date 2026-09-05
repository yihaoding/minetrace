"""StructureSource — DataSource backed by structural and geological vectors.

For each registered target point computes:

  Structural distances:
    dist_fault_km        : distance to nearest fault / shear zone (km)
    fault_density_5km    : fault-line density within 5 km (pts per km²)
    fault_density_10km   : fault-line density within 10 km
    dist_worm_mag_km     : distance to nearest magnetic-gradient worm (km)
    dist_worm_grav_km    : distance to nearest gravity-gradient worm (km)

  Geological attributes (one-hot rock type + continuous):
    is_rt_granitic       : rock type contains granitic
    is_rt_felsic         : felsic volcanic / intrusive (including meta-)
    is_rt_mafic          : mafic intrusive or volcanic (including meta-)
    is_rt_ultramafic     : ultramafic (including meta-)
    is_rt_metased        : metasedimentary
    is_rt_sedimentary    : sedimentary (unmetamorphosed)
    is_rt_metamorphic    : metamorphic (protolith unknown)
    is_rt_hydrothermal   : hydrothermal / carbonatite / lamproite
    age_ma               : MAX_AGE_MA (Ma), NaN if unknown
    is_yilgarn           : 1 if in Yilgarn Craton
    is_pilbara           : 1 if in Pilbara Craton
    is_covered           : 1 if overlain by Cenozoic cover
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEG_KM = 111.0
_LON_KM = 111.0 * np.cos(np.radians(25.0))

_STEP_DEG_FAULT = 0.005   # ~0.55 km — fault lines need finer sampling
_STEP_DEG_WORM  = 0.01    # ~1.1 km

# One-hot groups: name → substring patterns (any match → 1)
_ROCK_GROUPS: dict[str, list[str]] = {
    "is_rt_granitic":    ["granitic"],
    "is_rt_felsic":      ["felsic volcanic", "felsic intrusive", "meta-igneous felsic"],
    "is_rt_mafic":       ["mafic intrusive", "mafic volcanic", "meta-igneous mafic"],
    "is_rt_ultramafic":  ["ultramafic"],
    "is_rt_metased":     ["metasedimentary"],
    "is_rt_sedimentary": ["sedimentary siliciclastic", "sedimentary carbonate",
                          "sedimentary other"],
    "is_rt_metamorphic": ["metamorphic protolith"],
    "is_rt_hydrothermal":["hydrothermal", "carbonatite", "lamproite"],
}


def _lines_to_kdtree(gdf, step_deg: float):
    from scipy.spatial import cKDTree
    pts: list[list[float]] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            n = max(2, int(line.length / step_deg))
            for ti in np.linspace(0, 1, n):
                p = line.interpolate(ti, normalized=True)
                pts.append([p.x * _LON_KM, p.y * _DEG_KM])
    if not pts:
        raise ValueError("No geometry points found")
    return cKDTree(np.array(pts))


class StructureSource:
    """DataSource backed by GSWA structural and geological vector layers."""

    def __init__(
        self,
        fault_path: str | Path,
        worm_mag_path: str | Path,
        worm_grav_path: str | Path,
        geology_path: str | Path,
        cenozoic_path: str | Path,
        name: str = "structure",
    ) -> None:
        self.name = name
        self._features: dict[str, dict[str, float]] = {}

        import geopandas as gpd

        logger.info("StructureSource: loading vector layers …")

        fault  = gpd.read_file(str(fault_path))
        worm_m = gpd.read_file(str(worm_mag_path))
        worm_g = gpd.read_file(str(worm_grav_path))

        logger.info("  building KD-trees …")
        self._fault_tree = _lines_to_kdtree(fault,  _STEP_DEG_FAULT)
        self._wmag_tree  = _lines_to_kdtree(worm_m, _STEP_DEG_WORM)
        self._wgrav_tree = _lines_to_kdtree(worm_g, _STEP_DEG_WORM)

        self._geology  = gpd.read_file(str(geology_path))
        self._cenozoic = gpd.read_file(str(cenozoic_path))
        if self._geology.crs is None:
            self._geology  = self._geology.set_crs("EPSG:7844")
        if self._cenozoic.crs is None:
            self._cenozoic = self._cenozoic.set_crs("EPSG:7844")

        logger.info("StructureSource: ready.")

    # ── DataSource protocol ───────────────────────────────────────────────────

    @property
    def feature_names(self) -> list[str]:
        names = ["dist_fault_km", "fault_density_5km", "fault_density_10km",
                 "dist_worm_mag_km", "dist_worm_grav_km"]
        names += list(_ROCK_GROUPS.keys())
        names += ["age_ma", "is_yilgarn", "is_pilbara", "is_covered"]
        return names

    def get_features(self, sample_id: str) -> dict[str, float] | None:
        return self._features.get(sample_id)

    def is_available_for(self, sample_id: str) -> bool:
        return sample_id in self._features

    # ── registration ─────────────────────────────────────────────────────────

    def register_samples(self, samples: list) -> None:
        new = [s for s in samples if s.id not in self._features]
        logger.info("StructureSource: computing features for %d new points", len(new))
        for s in new:
            self._features[s.id] = self._compute(s.x, s.y)

    def compute_for_point(self, x: float, y: float) -> dict[str, float]:
        return self._compute(x, y)

    # ── internal computation ──────────────────────────────────────────────────

    def _compute(self, lon: float, lat: float) -> dict[str, float]:
        import geopandas as gpd
        from shapely.geometry import Point

        feats: dict[str, float] = {}
        pt_km = [lon * _LON_KM, lat * _DEG_KM]

        # --- structural distances ---
        d, _ = self._fault_tree.query([pt_km], k=1)
        feats["dist_fault_km"] = float(d[0])

        for radius, key in [(5.0, "fault_density_5km"), (10.0, "fault_density_10km")]:
            idx = self._fault_tree.query_ball_point(pt_km, radius)
            feats[key] = len(idx) / (np.pi * radius ** 2)

        d, _ = self._wmag_tree.query([pt_km], k=1)
        feats["dist_worm_mag_km"] = float(d[0])

        d, _ = self._wgrav_tree.query([pt_km], k=1)
        feats["dist_worm_grav_km"] = float(d[0])

        # --- geological attributes via spatial join ---
        pt_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:7844")

        join = gpd.sjoin(
            pt_gdf,
            self._geology[["ROCKTYPE1", "MAX_AGE_MA", "CRATON", "geometry"]],
            how="left", predicate="within",
        )

        if len(join) > 0 and not join["ROCKTYPE1"].isna().all():
            rt = str(join["ROCKTYPE1"].iloc[0]).lower()
            for fname, patterns in _ROCK_GROUPS.items():
                feats[fname] = float(any(p in rt for p in patterns))
            age_raw = join["MAX_AGE_MA"].iloc[0]
            try:
                feats["age_ma"] = float(age_raw) if pd.notna(age_raw) else np.nan
            except (TypeError, ValueError):
                feats["age_ma"] = np.nan
            craton = str(join["CRATON"].iloc[0]).lower()
            feats["is_yilgarn"] = float("yilgarn" in craton)
            feats["is_pilbara"] = float("pilbara" in craton)
        else:
            for fname in _ROCK_GROUPS:
                feats[fname] = np.nan
            feats["age_ma"] = feats["is_yilgarn"] = feats["is_pilbara"] = np.nan

        cov = gpd.sjoin(pt_gdf, self._cenozoic[["geometry"]], how="left",
                        predicate="within")
        feats["is_covered"] = float(
            len(cov) > 0 and not cov["index_right"].isna().all()
        )

        return feats
