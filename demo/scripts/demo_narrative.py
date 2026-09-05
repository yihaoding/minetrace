"""demo_narrative.py — Stage 2 single-point semantic description demo.

Standalone copy under demo/. Imports resolve against the demo-local copies
of core/ and domains/ (demo/core, demo/domains); data still reads from the
parent project's datasets/ directory.

Usage:
    cd /group/pmc050/yding/gad_reasoning/demo
    python scripts/demo_narrative.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# demo/ is self-contained: code + data both resolve under DEMO_ROOT.
DEMO_ROOT = Path(__file__).resolve().parent.parent       # .../gad_reasoning/demo
sys.path.insert(0, str(DEMO_ROOT))

from core.data_source import DataSourceRegistry
from core.expert import ExpertRegistry
from core.expert_tree import CompositeNode
from domains.geochem.experts.generic_target import TargetEnrichmentExpert, TargetPathfinderExpert
from domains.geochem.narrative import describe
from domains.geochem.samples import GeochemInstance, GeochemSample
from domains.geochem.sources.sediment_source import SedimentSource

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

GEO_DIR  = DEMO_ROOT / "datasets/geochemical/states/state_geochemical"
SITES_DIR = DEMO_ROOT / "datasets/geochemical/states/sites"
REPORTS  = DEMO_ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

SED_CSV  = GEO_DIR / "gswa_all_sediment.csv"
ROCK_CSV = GEO_DIR / "gswa_all_rockchips.csv"

TARGET   = "Cu"
PATHFINDERS = {"Au": 2.0, "Mo": 1.5, "Ag": 1.0, "Co": 0.8, "Bi": 0.7}
RATIO_FEATS = ["ratio_Cu_Mo", "ratio_Cu_Pb", "ratio_Co_Ni"]
N_BG    = 300

# ── demo points: a few known mines + random background ────────────────────────
DEMO_POINTS = [
    # name,           lon,      lat
    ("Chatsworth",    128.380, -26.141),
    ("Abbotts (51)",  118.368, -26.279),
    ("Yalgoo Copper", 116.516, -28.726),
    ("Random BG 1",   122.0,   -24.0),
    ("Random BG 2",   117.0,   -30.0),
]


def main() -> None:
    print("Loading geochemical data …")

    # register data sources
    sed_src  = SedimentSource(SED_CSV,  name="sediment",  center_lat=-25.0)
    rock_src = SedimentSource(ROCK_CSV, name="rockchips", center_lat=-25.0)
    DataSourceRegistry.register(sed_src)
    DataSourceRegistry.register(rock_src)

    # build demo samples
    samples = [
        GeochemSample(site_code=name.replace(" ", "_").replace("/", "_"),
                      x=lon, y=lat)
        for name, lon, lat in DEMO_POINTS
    ]
    sed_src.register_samples(samples)
    rock_src.register_samples(samples)

    # build background for null distribution
    rng = np.random.default_rng(42)
    bg_samples: list[GeochemSample] = []
    while len(bg_samples) < N_BG:
        bx = float(rng.uniform(114.5, 128.5))
        by = float(rng.uniform(-33.5, -15.0))
        if (sed_src.compute_for_point(bx, by).get("n_samples_10km", 0) >= 3
                and rock_src.compute_for_point(bx, by).get("n_samples_10km", 0) >= 3):
            sid = f"BG_{len(bg_samples):04d}"
            bg_samples.append(GeochemSample(site_code=sid, x=bx, y=by))
    sed_src.register_samples(bg_samples)
    rock_src.register_samples(bg_samples)

    # load KB positives for fitting
    import pandas as pd
    cu_df = pd.read_csv(SITES_DIR / "gswa_cu_sites.csv")
    kb_instances: list[GeochemInstance] = []
    for _, row in cu_df.iterrows():
        val = str(row.get("Cu_SITES", ""))
        if "Mine" in val or "Deposit" in val:
            sid = str(row.get("SITE_CODE", row.get("site_code", "")))
            lon = float(row.get("X", row.get("x", 0)))
            lat = float(row.get("Y", row.get("y", 0)))
            s = GeochemSample(site_code=sid, x=lon, y=lat)
            kb_instances.append(GeochemInstance(sample=s, confidence=1.0))

    print(f"KB positives: {len(kb_instances)}")
    kb_samples = [inst.sample for inst in kb_instances]
    sed_src.register_samples(kb_samples)
    rock_src.register_samples(kb_samples)

    # build and fit expert tree
    tree = CompositeNode(
        name="cu_tree",
        children=[
            TargetEnrichmentExpert(TARGET, "sediment"),
            TargetPathfinderExpert(TARGET, "sediment", PATHFINDERS, RATIO_FEATS),
            TargetEnrichmentExpert(TARGET, "rockchips"),
            TargetPathfinderExpert(TARGET, "rockchips", PATHFINDERS, RATIO_FEATS),
        ],
    )
    print("Fitting expert tree …")
    tree.fit(kb_samples, bg_samples)

    # score and describe each demo point
    md_sections: list[str] = [
        "# Stage 2 — Single-Point Semantic Description Demo",
        f"Target: **{TARGET}**  |  Expert tree: 4 leaf experts (sediment + rockchips)\n",
    ]

    for name, lon, lat in DEMO_POINTS:
        sid = name.replace(" ", "_").replace("/", "_")
        s   = GeochemSample(site_code=sid, x=lon, y=lat)
        ns  = tree.score_node(s)
        if ns is None:
            print(f"\n[{name}] — no score (missing features)")
            continue
        narr = describe(ns, target=TARGET, x=lon, y=lat)

        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        print(narr.to_text())

        md_sections.append(f"---\n")
        md_sections.append(narr.to_markdown())

    # save markdown report
    out = REPORTS / "stage2_narrative_demo.md"
    out.write_text("\n".join(md_sections), encoding="utf-8")
    print(f"\nMarkdown report saved to: {out}")


if __name__ == "__main__":
    main()
