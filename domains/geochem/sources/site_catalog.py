"""SiteCatalogSource — DataSource plugin for GSWA site catalog CSV.

Derives pseudo-geochemical features from a site catalog (not raw assay data).
Implements core.DataSource; registers itself under the name "site_catalog".

Features produced per site:
  commodity_richness      — number of distinct commodities present
  co_occurrence_with_cu   — Jaccard similarity with Cu sites' commodity sets
  stage_score             — numeric encoding of SITE_STAGE (0–5)
  local_density_5km       — count of other sites within 5 km
  local_density_20km      — count of other sites within 20 km
  pathfinder_count        — count of Cu pathfinders present (Mo/Au/Ag/Co/Bi/Re)
  is_polymetallic         — 1.0 if ≥3 commodities, else 0.0
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from domains.geochem.samples import STAGE_SCORE

logger = logging.getLogger(__name__)

# Cu pathfinder elements tracked in this catalog
_PATHFINDERS = {"Mo", "Au", "Ag", "Co", "Bi"}
_PATHFINDER_COLS = ["Mo_SITES", "Au_SITES", "Ag_SITES", "Co_SITES", "Bi_SITES"]

# Radius constants in degrees (approx: 1° ≈ 111 km at this latitude)
_R5_DEG = 5.0 / 111.0
_R20_DEG = 20.0 / 111.0


class SiteCatalogSource:
    """DataSource built from the GSWA Cu sites CSV.

    Load once; all feature vectors are precomputed at construction time.
    Thread-safe for reads after __init__.
    """

    name: str = "site_catalog"

    def __init__(self, csv_path: str | Path) -> None:
        self._df = pd.read_csv(csv_path)
        self._features: dict[str, dict[str, float]] = {}
        self._build_features()
        logger.info("SiteCatalogSource loaded %d sites", len(self._df))

    # ------------------------------------------------------------------
    # DataSource protocol
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return [
            "commodity_richness",
            "co_occurrence_with_cu",
            "stage_score",
            "local_density_5km",
            "local_density_20km",
            "pathfinder_count",
            "is_polymetallic",
        ]

    def get_features(self, sample_id: str) -> dict[str, float] | None:
        return self._features.get(sample_id)

    def is_available_for(self, sample_id: str) -> bool:
        return sample_id in self._features

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build_features(self) -> None:
        df = self._df
        commodity_cols = [c for c in df.columns if c.endswith("_SITES")]

        # --- commodity presence matrix (binary) ----------------------
        # Each cell is 1 if the string value is not NaN and not empty
        presence = df[commodity_cols].notna() & (df[commodity_cols] != "")
        richness = presence.sum(axis=1).astype(float)

        # --- co-occurrence with Cu ------------------------------------
        cu_mask = df["Cu_SITES"].notna() & (df["Cu_SITES"] != "")
        cu_presence = presence[cu_mask]
        # mean commodity vector across Cu sites (Jaccard numerator proxy)
        cu_mean = cu_presence.mean(axis=0).values  # shape (n_cols,)
        cu_norm = np.linalg.norm(cu_mean) + 1e-9
        row_vecs = presence.values.astype(float)
        row_norms = np.linalg.norm(row_vecs, axis=1, keepdims=True) + 1e-9
        co_occ = (row_vecs @ cu_mean) / (row_norms.ravel() * cu_norm)

        # --- stage score ---------------------------------------------
        stage_scores = df["SITE_STAGE"].map(STAGE_SCORE).fillna(0).astype(float)

        # --- spatial densities (KD-tree on lon/lat approx) -----------
        coords = df[["X", "Y"]].values  # (N, 2)
        tree = cKDTree(coords)
        density_5 = (
            np.array(tree.query_ball_point(coords, r=_R5_DEG, return_length=True), dtype=float) - 1
        )  # exclude self
        density_20 = (
            np.array(tree.query_ball_point(coords, r=_R20_DEG, return_length=True), dtype=float) - 1
        )

        # --- pathfinder count ----------------------------------------
        pf_cols_present = [c for c in _PATHFINDER_COLS if c in df.columns]
        pf_presence = df[pf_cols_present].notna() & (df[pf_cols_present] != "")
        pathfinder_count = pf_presence.sum(axis=1).astype(float)

        # --- polymetallic flag ---------------------------------------
        is_poly = (richness >= 3).astype(float)

        # --- assemble ------------------------------------------------
        for idx, row in df.iterrows():
            sid = str(row["SITE_CODE"])
            self._features[sid] = {
                "commodity_richness": float(richness.iloc[idx]),
                "co_occurrence_with_cu": float(co_occ[idx]),
                "stage_score": float(stage_scores.iloc[idx]),
                "local_density_5km": float(density_5[idx]),
                "local_density_20km": float(density_20[idx]),
                "pathfinder_count": float(pathfinder_count.iloc[idx]),
                "is_polymetallic": float(is_poly.iloc[idx]),
            }

    # ------------------------------------------------------------------
    # Convenience loaders
    # ------------------------------------------------------------------

    def load_all_samples(self) -> list:
        """Return all rows as GeochemSample objects (for pipeline bootstrap)."""
        from domains.geochem.samples import GeochemSample

        samples = []
        for _, row in self._df.iterrows():
            commo_raw = str(row.get("SITE_COMMO", ""))
            commo = [c.strip() for c in commo_raw.split(",") if c.strip()]
            s = GeochemSample(
                site_code=str(row["SITE_CODE"]),
                x=float(row["X"]),
                y=float(row["Y"]),
                site_commo=commo,
                site_stage=str(row.get("SITE_STAGE", "")),
                site_type=str(row.get("SITE_TYPE_", "")),
                raw_features=self._features.get(str(row["SITE_CODE"]), {}),
            )
            samples.append(s)
        return samples

    def load_cu_instances(self) -> list:
        """Return high-confidence Cu positive sites as GeochemInstance objects."""
        from domains.geochem.samples import GeochemSample, GeochemInstance, is_high_confidence_cu

        instances = []
        for _, row in self._df.iterrows():
            cu_val = str(row.get("Cu_SITES", ""))
            if is_high_confidence_cu(cu_val):
                sid = str(row["SITE_CODE"])
                commo_raw = str(row.get("SITE_COMMO", ""))
                commo = [c.strip() for c in commo_raw.split(",") if c.strip()]
                s = GeochemSample(
                    site_code=sid,
                    x=float(row["X"]),
                    y=float(row["Y"]),
                    site_commo=commo,
                    site_stage=str(row.get("SITE_STAGE", "")),
                    site_type=str(row.get("SITE_TYPE_", "")),
                    raw_features=self._features.get(sid, {}),
                )
                instances.append(
                    GeochemInstance(
                        sample=s,
                        confidence=1.0,
                        metadata={"cu_sites": cu_val},
                    )
                )
        return instances
