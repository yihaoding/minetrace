"""Demo script: GeochemAgent end-to-end for Cu prospectivity.

Builds a DataCatalog for all 5 assay types + geophysics + geology,
creates a GeochemAgent, plans, sets up, scores test points and saves
a markdown report.

Usage::

    cd <project root>
    /group/pmc050/yding/miniconda3/envs/geochem/bin/python scripts/run_geochem_agent.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_geochem_agent")

# ── imports ───────────────────────────────────────────────────────────────────
from core.catalog import DataCatalog, SourceSpec
from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX
from domains.geochem.narrative import describe
from domains.geochem.samples import GeochemSample

# ── path constants ────────────────────────────────────────────────────────────
DATA_ROOT = ROOT / "datasets"
GEOCHEM_DIR = DATA_ROOT / "geochemical" / "states" / "state_geochemical"
SITES_DIR   = DATA_ROOT / "geochemical" / "states" / "sites"
RASTER_DIR  = DATA_ROOT / "selected_layers" / "rasters"
VECTOR_DIR  = DATA_ROOT / "selected_layers" / "vectors"

SITES_CSV = str(SITES_DIR / "gswa_cu_sites.csv")

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ── build catalog ─────────────────────────────────────────────────────────────

def build_catalog() -> DataCatalog:
    catalog = DataCatalog()

    # ── assay sources (5 types) ──────────────────────────────────────────────
    assay_specs = [
        SourceSpec(
            name="sediment",
            source_type="assay_spatial",
            modality="geochemistry",
            subtype="sediment",
            path=str(GEOCHEM_DIR / "gswa_all_sediment.csv"),
            loader_kwargs={"sample_type_filter": "STREA"},
            description="GSWA stream sediment geochemistry (~157k rows)",
        ),
        SourceSpec(
            name="rockchips",
            source_type="assay_spatial",
            modality="geochemistry",
            subtype="rockchip",
            path=str(GEOCHEM_DIR / "gswa_all_rockchips.csv"),
            loader_kwargs={"sample_type_filter": "ROCKC"},
            description="GSWA rock chip geochemistry (~402k rows)",
        ),
        SourceSpec(
            name="drillhole",
            source_type="assay_spatial",
            modality="geochemistry",
            subtype="drillhole",
            path=str(GEOCHEM_DIR / "gswa_all_drillhole_maxgrade.csv"),
            loader_kwargs={"sample_type_filter": "DRILL"},
            description="GSWA drillhole max grade",
        ),
        SourceSpec(
            name="shallowdrill",
            source_type="assay_spatial",
            modality="geochemistry",
            subtype="shallowdrill",
            path=str(GEOCHEM_DIR / "gswa_all_shallowdrill.csv"),
            loader_kwargs={"sample_type_filter": "SHALL"},
            description="GSWA shallow drill geochemistry",
        ),
        SourceSpec(
            name="soil",
            source_type="assay_spatial",
            modality="geochemistry",
            subtype="soil",
            path=str(GEOCHEM_DIR / "gswa_all_surfsoilgeochem.csv"),
            loader_kwargs={"sample_type_filter": "SOIL"},
            description="GSWA surface soil geochemistry",
        ),
    ]
    for spec in assay_specs:
        catalog.register(spec)

    # ── geophysics rasters ────────────────────────────────────────────────────
    raster_specs = [
        ("mag",   "magnetics",    "WA_80m_Mag_Merge_1VD_v1_2020.ers",   0.08),
        ("grav",  "gravity",      "WA_400m_Grav_Merge_v1_2020.ers",     0.40),
        ("K",     "radiometrics", "WA_80m_K_Merge_v1_2018.ers",         0.08),
        ("Th",    "radiometrics", "WA_80m_Th_Merge_v1_2018.ers",        0.08),
        ("U",     "radiometrics", "WA_80m_U_Merge_v1_2018.ers",         0.08),
        ("LuHf",  "geochronology","WA_LuHf_TDM2_masked.bil",            0.10),
        ("SmNd",  "geochronology","WA_SmNd_TDM2_masked.bil",            0.10),
    ]
    for rkey, subdir, fname, res in raster_specs:
        path = RASTER_DIR / subdir / fname
        catalog.register(SourceSpec(
            name=f"raster_{rkey}",
            source_type="raster",
            modality="geophysics",
            subtype=rkey,
            path=str(path),
            resolution_km=res,
            loader_kwargs={"raster_key": rkey},
            description=f"GSWA {rkey} raster",
        ))

    # ── geology / structure vectors ───────────────────────────────────────────
    vector_specs = [
        ("fault",    "structures",        "500k_interpstrucl20.shp"),
        ("worm_mag", "geophysics_worms",  "worm_mag.shp"),
        ("worm_grav","geophysics_worms",  "worm_grav.shp"),
        ("geology",  "geology_context",   "GeologyMERGED.shp"),
        ("cenozoic", "geology_context",   "500k_cenozoicp20.shp"),
    ]
    for layer_key, subdir, fname in vector_specs:
        path = VECTOR_DIR / subdir / fname
        catalog.register(SourceSpec(
            name=f"vector_{layer_key}",
            source_type="vector",
            modality="geology",
            subtype=layer_key,
            path=str(path),
            loader_kwargs={"layer_key": layer_key},
            description=f"GSWA {layer_key} vector layer",
        ))

    return catalog


# ── target config ─────────────────────────────────────────────────────────────

def cu_target_config() -> GeochemTargetConfig:
    return GeochemTargetConfig(
        target="Cu",
        pathfinders={
            "Au": 1.5, "Mo": 1.5, "Ag": 1.0,
            "Co": 0.8, "Bi": 0.8, "Pb": 0.5,
        },
        ratio_features=["ratio_Cu_Mo", "ratio_Au_Cu", "ratio_Co_Ni"],
        confidence_map={
            "Mine":    1.0,
            "Deposit": 0.9,
        },
        sites_csv=SITES_CSV,
    )


# ── test points ───────────────────────────────────────────────────────────────

def make_test_samples():
    """A small set of test points: 3 known Cu areas + 2 background."""
    return [
        # Known Cu deposits (accurate coordinates)
        GeochemSample(site_code="TEST_DeGrussa",    x=119.00, y=-25.98),  # VMS Cu-Au, Murchison
        GeochemSample(site_code="TEST_Nifty",       x=121.68, y=-21.68),  # IOCG Cu, Paterson
        GeochemSample(site_code="TEST_Telfer",      x=122.06, y=-21.72),  # Cu-Au, Paterson
        # Background (remote, low-prospectivity areas)
        GeochemSample(site_code="TEST_BG_SW",       x=115.50, y=-32.00),
        GeochemSample(site_code="TEST_BG_NE",       x=127.00, y=-17.00),
    ]


# ── report generation ─────────────────────────────────────────────────────────

def write_markdown_report(
    plan,
    test_samples,
    node_scores,
    target: str,
    out_path: Path,
) -> None:
    lines = [
        f"# GeochemAgent Demo Report — {target} Prospectivity",
        "",
        "## Analysis Plan",
        "```",
        plan.summary(),
        "```",
        "",
        "## Scored Test Points",
        "",
    ]

    for sample, ns in zip(test_samples, node_scores):
        if ns is None:
            lines.append(f"### {sample.id}  ({sample.x:.3f}, {sample.y:.3f})")
            lines.append("_No score returned (insufficient coverage)._")
            lines.append("")
            continue

        narrative = describe(ns, target=target, x=sample.x, y=sample.y)
        lines.append(narrative.to_markdown())
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved → %s", out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== GeochemAgent Demo ===")

    # 1. Build catalog
    catalog = build_catalog()
    print(catalog.summary())
    print()

    # 2. Target config
    tc = cu_target_config()

    # 3. Create agent
    agent = GeochemAgent(
        catalog=catalog,
        target_config=tc,
        n_bg=300,
        bbox=WA_BBOX,
        seed=42,
    )

    # 4. Plan (with coverage check)
    logger.info("Planning …")
    plan = agent.plan(check_bbox=WA_BBOX)
    print(plan.summary())
    print()

    # 5. Setup (load data + fit tree)
    logger.info("Setting up agent (loading data, building expert tree) …")
    agent.setup()
    logger.info("Active sources : %s", agent.active_sources())
    logger.info("Active experts : %s", agent.active_experts())
    print()

    # 6. Score test points
    test_samples = make_test_samples()
    logger.info("Scoring %d test points …", len(test_samples))
    node_scores = agent.score_batch(test_samples)

    # 7. Print summaries
    for sample, ns in zip(test_samples, node_scores):
        if ns is None:
            print(f"{sample.id:30s}  score=None")
        else:
            flat = ns.flat_scores()
            root_score = flat.get(ns.name, ns.score)
            print(f"{sample.id:30s}  score={root_score:.3f}")
            narrative = describe(ns, target=tc.target, x=sample.x, y=sample.y)
            print(narrative.to_text())
    print()

    # 8. Individual case: inject direct drillhole evidence for test point 0
    logger.info("Individual case mode: injecting direct evidence for %s", test_samples[0].id)
    direct_ev = {"Cu_ppm": 1800.0, "Au_ppm": 0.85, "Mo_ppm": 95.0}
    ns_case = agent.score_case(test_samples[0], direct_evidence=direct_ev)
    if ns_case is not None:
        print(f"\n[Case mode] {test_samples[0].id}  combined score={ns_case.score:.3f}")
        flat = ns_case.flat_scores()
        for k, v in flat.items():
            print(f"  {k}: {v:.3f}")
    print()

    # 9. Save markdown report
    report_path = REPORTS_DIR / "geochem_agent_demo.md"
    write_markdown_report(plan, test_samples, node_scores, tc.target, report_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
