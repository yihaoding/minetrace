"""demo_au_narrative.py — Au prospectivity narrative on 10 points, FULL stack.

Uses ProspectivityModel with all three layers enabled:
  - geochem (sediment + rockchip + drillhole + shallowdrill + soil)
  - geophys (magnetics + gravity + radiometrics + geochronology rasters)
  - structure (faults + worms + geology + cenozoic cover)

Outputs:
  - stdout: per-point narrative
  - demo/sessions/au_10_points_narrative.md: markdown report

Usage:
    cd /group/pmc050/yding/gad_reasoning
    conda run -n geochem python demo/scripts/demo_au_narrative.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# demo/ is self-contained: code (demo/core, demo/domains) + data (demo/datasets)
# all resolve under DEMO_ROOT so the folder can be packaged and run standalone.
DEMO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DEMO_ROOT))

from core.catalog import DataCatalog, SourceSpec
from domains.geochem.narrative import describe
from domains.geochem.prospectivity_model import ProspectivityModel, TargetConfig, WA_BBOX
from domains.geochem.samples import GeochemSample

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DATA_ROOT = DEMO_ROOT / "datasets"
GEOCHEM_DIR = DATA_ROOT / "geochemical" / "states" / "state_geochemical"
SITES_DIR = DATA_ROOT / "geochemical" / "states" / "sites"
RASTER_DIR = DATA_ROOT / "selected_layers" / "rasters"
VECTOR_DIR = DATA_ROOT / "selected_layers" / "vectors"

AU_SITES_CSV = SITES_DIR / "gswa_au_site.csv"

OUT_DIR = DEMO_ROOT / "sessions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Au"
PATHFINDERS = {"As": 2.0, "Sb": 1.5, "Ag": 1.5, "Bi": 1.0, "W": 0.8, "Cu": 0.5}
RATIO_FEATS = ["ratio_Au_Cu", "ratio_Au_As", "ratio_Sb_As"]
CONFIDENCE = {"Mine": 1.0, "Deposit": 0.9}

SEED = 42
N_KNOWN_DEMO = 5
N_RANDOM_DEMO = 5


def build_catalog() -> DataCatalog:
    """Same layer registration as eval_comprehensive.py (full 3-layer stack)."""
    catalog = DataCatalog()
    for name, subtype, fname, st in [
        ("sediment",     "sediment",     "gswa_all_sediment.csv",          "STREA"),
        ("rockchips",    "rockchip",     "gswa_all_rockchips.csv",         "ROCKC"),
        ("drillhole",    "drillhole",    "gswa_all_drillhole_maxgrade.csv","DRILL"),
        ("shallowdrill", "shallowdrill", "gswa_all_shallowdrill.csv",      "SHALL"),
        ("soil",         "soil",         "gswa_all_surfsoilgeochem.csv",   "SOIL"),
    ]:
        catalog.register(SourceSpec(
            name=name, source_type="assay_spatial", modality="geochemistry",
            subtype=subtype, path=str(GEOCHEM_DIR / fname),
            loader_kwargs={"sample_type_filter": st},
        ))
    for rkey, subdir, fname, res in [
        ("mag",  "magnetics",     "WA_80m_Mag_Merge_1VD_v1_2020.ers", 0.08),
        ("grav", "gravity",       "WA_400m_Grav_Merge_v1_2020.ers",   0.40),
        ("K",    "radiometrics",  "WA_80m_K_Merge_v1_2018.ers",       0.08),
        ("Th",   "radiometrics",  "WA_80m_Th_Merge_v1_2018.ers",      0.08),
        ("U",    "radiometrics",  "WA_80m_U_Merge_v1_2018.ers",       0.08),
        ("LuHf", "geochronology", "WA_LuHf_TDM2_masked.bil",          0.10),
        ("SmNd", "geochronology", "WA_SmNd_TDM2_masked.bil",          0.10),
    ]:
        catalog.register(SourceSpec(
            name=f"raster_{rkey}", source_type="raster", modality="geophysics",
            subtype=rkey, path=str(RASTER_DIR / subdir / fname),
            resolution_km=res, loader_kwargs={"raster_key": rkey},
        ))
    for layer_key, subdir, fname in [
        ("fault",     "structures",        "500k_interpstrucl20.shp"),
        ("worm_mag",  "geophysics_worms",  "worm_mag.shp"),
        ("worm_grav", "geophysics_worms",  "worm_grav.shp"),
        ("geology",   "geology_context",   "GeologyMERGED.shp"),
        ("cenozoic",  "geology_context",   "500k_cenozoicp20.shp"),
    ]:
        catalog.register(SourceSpec(
            name=f"vector_{layer_key}", source_type="vector", modality="geology",
            subtype=layer_key, path=str(VECTOR_DIR / subdir / fname),
            loader_kwargs={"layer_key": layer_key},
        ))
    return catalog


def pick_known_au_sites(rng: np.random.Generator, k: int) -> list[tuple[str, float, float]]:
    df = pd.read_csv(AU_SITES_CSV)
    df = df[df["Au_SITES"].astype(str).str.contains("Mine|Deposit", regex=True, na=False)]
    df = df.dropna(subset=["X", "Y", "SHORT_NAME"]).drop_duplicates(subset=["SHORT_NAME"])

    lat_min, lat_max = WA_BBOX[2], WA_BBOX[3]
    band_edges = np.linspace(lat_min, lat_max, k + 1)
    picks: list[tuple[str, float, float]] = []
    for i in range(k):
        lo, hi = band_edges[i], band_edges[i + 1]
        band = df[(df["Y"] >= lo) & (df["Y"] < hi)]
        if len(band) == 0:
            continue
        row = band.sample(1, random_state=int(rng.integers(0, 10_000))).iloc[0]
        picks.append((str(row["SHORT_NAME"]), float(row["X"]), float(row["Y"])))
    while len(picks) < k:
        row = df.sample(1, random_state=int(rng.integers(0, 10_000))).iloc[0]
        name = str(row["SHORT_NAME"])
        if any(p[0] == name for p in picks):
            continue
        picks.append((name, float(row["X"]), float(row["Y"])))
    return picks[:k]


def pick_random_points(rng: np.random.Generator, k: int) -> list[tuple[str, float, float]]:
    picks: list[tuple[str, float, float]] = []
    for _ in range(k):
        lon = float(rng.uniform(WA_BBOX[0], WA_BBOX[1]))
        lat = float(rng.uniform(WA_BBOX[2], WA_BBOX[3]))
        picks.append((f"Random_{len(picks) + 1}", lon, lat))
    return picks


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Building catalog (5 geochem + 7 raster + 5 vector sources) ...")
    catalog = build_catalog()

    tc = TargetConfig(
        target=TARGET,
        pathfinders=PATHFINDERS,
        ratio_features=RATIO_FEATS,
        confidence_map=CONFIDENCE,
        sites_csv=str(AU_SITES_CSV),
        layers=frozenset({"geochem", "geophys", "structure"}),
    )

    print(f"Picking {N_KNOWN_DEMO} known Au sites + {N_RANDOM_DEMO} random points ...")
    known = pick_known_au_sites(rng, N_KNOWN_DEMO)
    random_pts = pick_random_points(rng, N_RANDOM_DEMO)
    demo_points = [(n, lon, lat, "known") for n, lon, lat in known] + \
                  [(n, lon, lat, "random") for n, lon, lat in random_pts]

    print("Instantiating ProspectivityModel (Au, all 3 layers) ...")
    model = ProspectivityModel(catalog=catalog, config=tc, n_bg=300, bbox=WA_BBOX, seed=SEED)

    print("plan() — probing coverage ...")
    model.plan(check_bbox=WA_BBOX)

    print("setup() — loading sources, fitting expert tree (this may take 1-3 min) ...")
    model.setup()

    print(f"\nActive sources : {model.active_sources()}")
    print(f"Active experts : {model.active_experts()}\n")

    demo_samples = [
        GeochemSample(site_code=name.replace(" ", "_").replace("/", "_"), x=lon, y=lat)
        for name, lon, lat, _ in demo_points
    ]
    print("Scoring 10 demo points ...")
    node_scores = model.score_batch(demo_samples)

    md_sections: list[str] = [
        "# Au Prospectivity — 10-Point Narrative (full 3-layer stack)",
        f"\nTarget: **{TARGET}**",
        f"Active experts: `{model.active_experts()}`",
        f"Active sources: `{model.active_sources()}`",
        f"Pathfinders: `{PATHFINDERS}`",
        f"Ratio features: `{RATIO_FEATS}`",
        "",
    ]

    rows = []
    for (name, lon, lat, kind), ns in zip(demo_points, node_scores):
        if ns is None:
            print(f"\n[{name} ({kind})] — no score (missing features)")
            rows.append((name, kind, lon, lat, None, "no coverage"))
            continue
        narr = describe(ns, target=TARGET, x=lon, y=lat)

        print(f"\n{'=' * 72}")
        print(f"  [{kind.upper()}] {name}  ({lon:.3f}, {lat:.3f})")
        print(f"{'=' * 72}")
        print(narr.to_text())

        rows.append((name, kind, lon, lat, ns.score, narr.tier))

        md_sections.append("---\n")
        md_sections.append(f"### [{kind.upper()}] {name}  ({lon:.3f}, {lat:.3f})\n")
        md_sections.append(narr.to_markdown())
        md_sections.append("")

    print("\n" + "=" * 72)
    print("  SUMMARY TABLE (3-layer)")
    print("=" * 72)
    print(f"  {'Name':<24} {'Kind':<8} {'Lon':>8} {'Lat':>8} {'g-score':>9} {'Tier':<12}")
    for name, kind, lon, lat, score, tier in rows:
        if score is None:
            print(f"  {name[:24]:<24} {kind:<8} {lon:>8.3f} {lat:>8.3f} {'-':>9} {tier:<12}")
        else:
            print(f"  {name[:24]:<24} {kind:<8} {lon:>8.3f} {lat:>8.3f} {score:>9.3f} {tier:<12}")

    out = OUT_DIR / "au_10_points_narrative.md"
    out.write_text("\n".join(md_sections), encoding="utf-8")
    print(f"\nMarkdown report saved to: {out}")


if __name__ == "__main__":
    main()
