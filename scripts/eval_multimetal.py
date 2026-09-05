"""Multi-metal evaluation: Cu, Au, Ni, W prospectivity.

For each metal: random-split hold-out + random negatives + spatially matched
negatives. Compares composite AUC and per-expert AUC across metals.

Usage::

    /group/pmc050/yding/miniconda3/envs/geochem/bin/python scripts/eval_multimetal.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("eval_multimetal")

from core.catalog import DataCatalog, SourceSpec
from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX
from domains.geochem.samples import GeochemSample

DATA_ROOT   = ROOT / "datasets"
GEOCHEM_DIR = DATA_ROOT / "geochemical" / "states" / "state_geochemical"
SITES_DIR   = DATA_ROOT / "geochemical" / "states" / "sites"
RASTER_DIR  = DATA_ROOT / "selected_layers" / "rasters"
VECTOR_DIR  = DATA_ROOT / "selected_layers" / "vectors"

SEED = 42
N_TEST_POS = 60    # held-out positives per metal (W has only 86 total)
N_TEST_NEG = 200   # negatives

# ── Metal configurations ──────────────────────────────────────────────────────

METAL_CONFIGS = {
    "Cu": dict(
        sites_csv=str(SITES_DIR / "gswa_cu_sites.csv"),
        site_col="Cu_SITES",
        pathfinders={"Au": 1.5, "Mo": 1.5, "Ag": 1.0, "Co": 0.8, "Bi": 0.8, "Pb": 0.5},
        ratio_features=["ratio_Cu_Mo", "ratio_Au_Cu", "ratio_Co_Ni"],
        confidence_map={"Mine": 1.0, "Deposit": 0.9},
    ),
    "Au": dict(
        sites_csv=str(SITES_DIR / "gswa_au_site.csv"),
        site_col="Au_SITES",
        pathfinders={"As": 2.0, "Sb": 1.5, "Ag": 1.5, "Bi": 1.0, "W": 0.8, "Cu": 0.5},
        ratio_features=["ratio_Au_Cu", "ratio_Au_As", "ratio_Sb_As"],
        confidence_map={"Mine": 1.0, "Deposit": 0.9},
    ),
    "Ni": dict(
        sites_csv=str(SITES_DIR / "gswa_ni_site.csv"),
        site_col="Ni_SITES",
        pathfinders={"Co": 2.0, "Cr": 1.5, "Cu": 1.0, "Au": 0.5},
        ratio_features=["ratio_Co_Ni", "ratio_Ni_Co", "ratio_Ni_Cr"],
        confidence_map={"Mine": 1.0, "Deposit": 0.9},
    ),
    "W": dict(
        sites_csv=str(SITES_DIR / "gswa_w_site.csv"),
        site_col="W_SITES",
        pathfinders={"Mo": 2.0, "Bi": 1.5, "Sn": 1.5, "Cu": 1.0, "As": 0.8},
        ratio_features=["ratio_W_Mo", "ratio_W_Sn"],
        confidence_map={"Mine": 1.0, "Deposit": 0.9},
    ),
}


# ── Catalog ───────────────────────────────────────────────────────────────────

def build_catalog() -> DataCatalog:
    catalog = DataCatalog()
    for name, subtype, fname, st in [
        ("sediment",    "sediment",    "gswa_all_sediment.csv",          "STREA"),
        ("rockchips",   "rockchip",    "gswa_all_rockchips.csv",         "ROCKC"),
        ("drillhole",   "drillhole",   "gswa_all_drillhole_maxgrade.csv","DRILL"),
        ("shallowdrill","shallowdrill","gswa_all_shallowdrill.csv",       "SHALL"),
        ("soil",        "soil",        "gswa_all_surfsoilgeochem.csv",    "SOIL"),
    ]:
        catalog.register(SourceSpec(
            name=name, source_type="assay_spatial", modality="geochemistry",
            subtype=subtype, path=str(GEOCHEM_DIR / fname),
            loader_kwargs={"sample_type_filter": st},
        ))
    for rkey, subdir, fname, res in [
        ("mag","magnetics","WA_80m_Mag_Merge_1VD_v1_2020.ers",0.08),
        ("grav","gravity","WA_400m_Grav_Merge_v1_2020.ers",0.40),
        ("K","radiometrics","WA_80m_K_Merge_v1_2018.ers",0.08),
        ("Th","radiometrics","WA_80m_Th_Merge_v1_2018.ers",0.08),
        ("U","radiometrics","WA_80m_U_Merge_v1_2018.ers",0.08),
        ("LuHf","geochronology","WA_LuHf_TDM2_masked.bil",0.10),
        ("SmNd","geochronology","WA_SmNd_TDM2_masked.bil",0.10),
    ]:
        catalog.register(SourceSpec(
            name=f"raster_{rkey}", source_type="raster", modality="geophysics",
            subtype=rkey, path=str(RASTER_DIR / subdir / fname),
            resolution_km=res, loader_kwargs={"raster_key": rkey},
        ))
    for layer_key, subdir, fname in [
        ("fault","structures","500k_interpstrucl20.shp"),
        ("worm_mag","geophysics_worms","worm_mag.shp"),
        ("worm_grav","geophysics_worms","worm_grav.shp"),
        ("geology","geology_context","GeologyMERGED.shp"),
        ("cenozoic","geology_context","500k_cenozoicp20.shp"),
    ]:
        catalog.register(SourceSpec(
            name=f"vector_{layer_key}", source_type="vector", modality="geology",
            subtype=layer_key, path=str(VECTOR_DIR / subdir / fname),
            loader_kwargs={"layer_key": layer_key},
        ))
    return catalog


# ── Data loading ──────────────────────────────────────────────────────────────

def load_positives(sites_csv: str, site_col: str,
                   confidence_map: dict) -> list[GeochemSample]:
    df = pd.read_csv(sites_csv)
    site_code_col = "SITE_CODE" if "SITE_CODE" in df.columns else "fid"
    samples = []
    for _, row in df.iterrows():
        val = str(row.get(site_col, ""))
        for kw in confidence_map:
            if kw in val:
                samples.append(GeochemSample(
                    site_code=str(row[site_code_col]),
                    x=float(row["X"]), y=float(row["Y"]),
                ))
                break
    return samples


def load_spatially_matched_neg(
    test_pos: list[GeochemSample],
    pos_ids: set,
    all_sites_df: pd.DataFrame,
    radius_km: float,
    seed: int,
) -> list[GeochemSample]:
    KM = 100.0
    non_pos = all_sites_df[~all_sites_df["SITE_CODE"].isin(pos_ids)].reset_index(drop=True)
    neg_xy = np.column_stack([non_pos.X.values * KM, non_pos.Y.values * KM])
    tree = cKDTree(neg_xy)
    rng = np.random.default_rng(seed + 7)
    matched, used = [], set()
    for s in test_pos:
        cands = [i for i in tree.query_ball_point([s.x * KM, s.y * KM], r=radius_km)
                 if i not in used]
        if not cands:
            _, idx = tree.query([s.x * KM, s.y * KM], k=5)
            cands = [i for i in (idx if np.ndim(idx) > 0 else [idx]) if i not in used]
        if not cands:
            continue
        pick = int(rng.choice(cands))
        used.add(pick)
        row = non_pos.iloc[pick]
        matched.append(GeochemSample(
            site_code=str(row["SITE_CODE"]), x=float(row["X"]), y=float(row["Y"])
        ))
    return matched


def load_random_background(agent: GeochemAgent, n: int, seed: int) -> list[GeochemSample]:
    rng = np.random.default_rng(seed + 1)
    x_min, x_max, y_min, y_max = WA_BBOX
    bg, attempts = [], 0
    while len(bg) < n and attempts < n * 500:
        attempts += 1
        x, y = float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))
        for src_name in agent.active_sources():
            try:
                src = agent._registry.get(src_name)
                if hasattr(src, "has_coverage") and src.has_coverage(x, y):
                    bg.append(GeochemSample(site_code=f"BG_{len(bg):04d}", x=x, y=y))
                    break
            except KeyError:
                pass
    return bg


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval(agent, test_pos, neg_samples) -> Optional[dict]:
    all_test = test_pos + neg_samples
    labels   = [1] * len(test_pos) + [0] * len(neg_samples)
    ns_list  = agent.score_batch(all_test)
    pairs    = [(ns, lbl) for ns, lbl in zip(ns_list, labels) if ns is not None]
    if len(pairs) < 10 or len(set(lbl for _, lbl in pairs)) < 2:
        return None
    y_true = [lbl for _, lbl in pairs]
    y_pred = [ns.score for ns, _ in pairs]
    auc = float(roc_auc_score(y_true, y_pred))
    paired = sorted(zip(y_pred, y_true), reverse=True)
    prec = {}
    for k in (10, 20, 50):
        if k <= len(paired):
            top = [l for _, l in paired[:k]]
            prec[k] = sum(top) / k
    return {"auc": auc, "n": len(pairs), "prec": prec}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    catalog = build_catalog()

    # build combined all-metal sites for spatial matching
    frames = []
    for metal, cfg in METAL_CONFIGS.items():
        df = pd.read_csv(cfg["sites_csv"])
        if "SITE_CODE" not in df.columns and "fid" in df.columns:
            df = df.rename(columns={"fid": "SITE_CODE"})
        df["SITE_CODE"] = df["SITE_CODE"].astype(str)
        frames.append(df[["X", "Y", "SITE_CODE"]].dropna())
    all_sites = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["X", "Y"])

    results = {}

    for metal, cfg in METAL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {metal} Prospectivity")
        print(f"{'='*60}")

        # load positives
        all_pos = load_positives(cfg["sites_csv"], cfg["site_col"], cfg["confidence_map"])
        print(f"  Total {metal} Mine/Deposit: {len(all_pos)}")
        if len(all_pos) < N_TEST_POS + 10:
            n_test = max(10, len(all_pos) // 5)
            print(f"  (small dataset, using {n_test} test positives)")
        else:
            n_test = N_TEST_POS

        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(all_pos))
        test_pos = [all_pos[i] for i in idx[:n_test]]
        test_ids = {s.id for s in test_pos}
        all_pos_ids = {s.id for s in all_pos}

        # build agent
        tc = GeochemTargetConfig(
            target=metal,
            pathfinders=cfg["pathfinders"],
            ratio_features=cfg["ratio_features"],
            confidence_map=cfg["confidence_map"],
            sites_csv=cfg["sites_csv"],
        )
        agent = GeochemAgent(
            catalog=catalog, target_config=tc, n_bg=300, bbox=WA_BBOX, seed=SEED
        )
        agent.plan(check_bbox=WA_BBOX)
        agent.setup(exclude_ids=test_ids)
        print(f"  Active experts: {agent.active_experts()}")

        # negatives
        random_neg  = load_random_background(agent, N_TEST_NEG, seed=SEED)
        matched_neg = load_spatially_matched_neg(
            test_pos, all_pos_ids, all_sites, radius_km=50.0, seed=SEED
        )

        # evaluate
        r_rand    = run_eval(agent, test_pos, random_neg)
        r_matched = run_eval(agent, test_pos, matched_neg)

        results[metal] = {"random": r_rand, "matched": r_matched}

        if r_rand:
            p = r_rand["prec"]
            print(f"  Random neg   — AUC={r_rand['auc']:.3f}  "
                  f"P@10={p.get(10,''):.2f}  P@20={p.get(20,''):.2f}  P@50={p.get(50,''):.2f}")
        if r_matched:
            p = r_matched["prec"]
            print(f"  Matched neg  — AUC={r_matched['auc']:.3f}  "
                  f"P@10={p.get(10,''):.2f}  P@20={p.get(20,''):.2f}  P@50={p.get(50,''):.2f}")

    # summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metal':<6}  {'AUC (random)':>14}  {'AUC (matched 50km)':>18}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*18}")
    for metal, res in results.items():
        r = res["random"]["auc"] if res["random"] else float("nan")
        m = res["matched"]["auc"] if res["matched"] else float("nan")
        print(f"  {metal:<6}  {r:>14.3f}  {m:>18.3f}")


if __name__ == "__main__":
    main()
