"""Smoke-test: build the full expert tree for Cu and verify NodeScore output.

Tests the end-to-end flow:
  1. Load Cu mine sites as positives + random background samples
  2. Register all data sources (SedimentSource, GeophysicsSource, StructureSource)
  3. Build GeochemNode, GeophysicsNode, StructureNode → root CompositeNode
  4. Fit the tree on training data
  5. Score a test point and print the full NodeScore tree

Usage:
  python scripts/smoke_test_tree.py
"""
from __future__ import annotations

import logging
import os
import sys

# make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("smoke_test")

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(BASE, "datasets")

CU_SITES_CSV  = os.path.join(DATASETS, "geochemical/states/sites/gswa_cu_sites.csv")
SEDIMENT_CSV  = os.path.join(DATASETS, "geochemical/states/state_geochemical/gswa_all_sediment.csv")

RASTER_PATHS = {
    "mag":  os.path.join(DATASETS, "selected_layers/rasters/magnetics/WA_80m_Mag_Merge_1VD_v1_2020.ers"),
    "grav": os.path.join(DATASETS, "selected_layers/rasters/gravity/WA_400m_Grav_Merge_v1_2020.ers"),
    "K":    os.path.join(DATASETS, "selected_layers/rasters/radiometrics/WA_80m_K_Merge_v1_2018.ers"),
    "Th":   os.path.join(DATASETS, "selected_layers/rasters/radiometrics/WA_80m_Th_Merge_v1_2018.ers"),
    "U":    os.path.join(DATASETS, "selected_layers/rasters/radiometrics/WA_80m_U_Merge_v1_2018.ers"),
    "LuHf": os.path.join(DATASETS, "selected_layers/rasters/geochronology/WA_LuHf_TDM2_masked.bil"),
    "SmNd": os.path.join(DATASETS, "selected_layers/rasters/geochronology/WA_SmNd_TDM2_masked.bil"),
}

STRUCTURE_PATHS = dict(
    fault_path   = os.path.join(DATASETS, "selected_layers/vectors/structures/500k_interpstrucl20.shp"),
    worm_mag_path= os.path.join(DATASETS, "selected_layers/vectors/geophysics_worms/worm_mag.shp"),
    worm_grav_path=os.path.join(DATASETS, "selected_layers/vectors/geophysics_worms/worm_grav.shp"),
    geology_path = os.path.join(DATASETS, "selected_layers/vectors/geology_context/GeologyMERGED.shp"),
    cenozoic_path= os.path.join(DATASETS, "selected_layers/vectors/geology_context/500k_cenozoicp20.shp"),
)

# ── imports ───────────────────────────────────────────────────────────────────

from core.data_source import DataSourceRegistry
from core.expert_tree import CompositeNode
from domains.geochem.samples import GeochemSample, is_high_confidence_cu
from domains.geochem.sources.sediment_source import SedimentSource
from domains.geochem.sources.geophysics import GeophysicsSource
from domains.geochem.sources.geology import StructureSource
from domains.geochem.experts.generic_target import TargetEnrichmentExpert, TargetPathfinderExpert
from domains.geochem.experts.geophysics_experts import (
    MagneticExpert, GravityExpert, RadiometricExpert, GeochronExpert,
)
from domains.geochem.experts.structure_experts import FaultExpert, WormExpert, GeologyExpert

import pandas as pd


# ── 1. load Cu mine/deposit sites as positives ───────────────────────────────

def load_cu_positives(csv_path: str, max_n: int = 100) -> list[GeochemSample]:
    df = pd.read_csv(csv_path)
    pos = df[df["Cu_SITES"].apply(is_high_confidence_cu)]
    pos = pos.head(max_n)
    samples = []
    for _, row in pos.iterrows():
        samples.append(GeochemSample(
            site_code=str(row["SITE_CODE"]),
            x=float(row["X"]),
            y=float(row["Y"]),
        ))
    logger.info("Loaded %d Cu positives", len(samples))
    return samples


def load_background(csv_path: str, max_n: int = 200, seed: int = 42) -> list[GeochemSample]:
    df = pd.read_csv(csv_path)
    # exclude Cu mine/deposit sites
    is_cu = df["Cu_SITES"].apply(is_high_confidence_cu) if "Cu_SITES" in df.columns else pd.Series(False, index=df.index)
    df_bg = df[~is_cu]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df_bg), size=min(max_n, len(df_bg)), replace=False)
    rows = df_bg.iloc[idx]
    samples = []
    for _, row in rows.iterrows():
        samples.append(GeochemSample(
            site_code=str(row["SITE_CODE"]),
            x=float(row["X"]),
            y=float(row["Y"]),
        ))
    logger.info("Loaded %d background samples", len(samples))
    return samples


# ── 2. build and register data sources ───────────────────────────────────────

def build_sources(positives, background):
    all_samples = positives + background

    logger.info("=== Building SedimentSource ===")
    sed = SedimentSource(
        csv_path=SEDIMENT_CSV,
        name="sediment",
        elements=["Cu", "Mo", "Au", "Ag", "Pb", "Zn", "Co", "As", "Bi"],
    )
    sed.register_samples(all_samples)
    DataSourceRegistry.register(sed)

    logger.info("=== Building GeophysicsSource ===")
    geo = GeophysicsSource(raster_paths=RASTER_PATHS, name="geophysics")
    geo.register_samples(all_samples)
    DataSourceRegistry.register(geo)

    logger.info("=== Building StructureSource ===")
    struct = StructureSource(**STRUCTURE_PATHS, name="structure")
    struct.register_samples(all_samples)
    DataSourceRegistry.register(struct)

    return sed, geo, struct


# ── 3. build expert tree ──────────────────────────────────────────────────────

def build_cu_tree() -> CompositeNode:
    geochem_node = CompositeNode("geochem_cu", children=[
        TargetEnrichmentExpert(target="Cu", source_name="sediment"),
        TargetPathfinderExpert(
            target="Cu",
            source_name="sediment",
            pathfinders={"Mo": 1.0, "Au": 0.8, "Ag": 0.6, "Co": 0.5, "Bi": 0.4},
        ),
    ])

    geophys_node = CompositeNode("geophysics_cu", children=[
        MagneticExpert(source_name="geophysics"),
        GravityExpert(source_name="geophysics"),
        RadiometricExpert(source_name="geophysics"),
        GeochronExpert(source_name="geophysics"),
    ])

    structure_node = CompositeNode("structure_cu", children=[
        FaultExpert(source_name="structure"),
        WormExpert(source_name="structure"),
        GeologyExpert(source_name="structure"),
    ])

    root = CompositeNode("root_cu", children=[
        geochem_node, geophys_node, structure_node,
    ])
    return root


# ── 4. print NodeScore tree ───────────────────────────────────────────────────

def print_node_score(ns, indent: int = 0) -> None:
    prefix = "  " * indent
    wstr = ""
    if ns.weights:
        top = sorted(ns.weights.items(), key=lambda x: -x[1])[:3]
        wstr = "  weights: " + ", ".join(f"{k}={v:.3f}" for k, v in top)
    print(f"{prefix}{ns.name}: {ns.score:.4f}{wstr}")
    for child in ns.children.values():
        print_node_score(child, indent + 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    DataSourceRegistry.clear()

    logger.info("Loading Cu sites …")
    positives  = load_cu_positives(CU_SITES_CSV, max_n=80)
    background = load_background(CU_SITES_CSV,   max_n=160)

    logger.info("Building data sources …")
    build_sources(positives, background)

    logger.info("Building expert tree …")
    tree = build_cu_tree()

    logger.info("Fitting tree …")
    tree.fit(positives, background)

    logger.info("\n=== NodeScore tree for first test positive ===")
    test_sample = positives[0]
    ns = tree.score_node(test_sample)
    if ns is None:
        logger.error("score_node returned None for %s", test_sample.id)
        return

    print_node_score(ns)

    print("\n--- Flat scores ---")
    for k, v in sorted(ns.flat_scores().items()):
        print(f"  {k}: {v:.4f}")

    logger.info("\n=== Scoring %d positives + %d background ===", len(positives), len(background))
    pos_scores = [tree.score_node(s) for s in positives]
    bg_scores  = [tree.score_node(s) for s in background]
    pos_vals = [ns.score for ns in pos_scores if ns is not None]
    bg_vals  = [ns.score for ns in bg_scores  if ns is not None]

    from sklearn.metrics import roc_auc_score
    if len(set([1]*len(pos_vals) + [0]*len(bg_vals))) == 2:
        y = [1]*len(pos_vals) + [0]*len(bg_vals)
        scores = pos_vals + bg_vals
        auc = roc_auc_score(y, scores)
        print(f"\nIn-sample AUC (train = test, indicative only): {auc:.4f}")
        print(f"Mean score — positives: {np.mean(pos_vals):.4f}  background: {np.mean(bg_vals):.4f}")
    else:
        logger.warning("Not enough class diversity to compute AUC")


if __name__ == "__main__":
    main()
