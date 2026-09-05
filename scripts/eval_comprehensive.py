"""Comprehensive multi-metal evaluation: 4 negative types × 9 metals.

Positives: each element draws 30 random test positives from `enriched_<el>_sites.csv`
(rows where `<el>_SITES contains Mine|Deposit`). Remaining positives go to training.

Negative strategies
-------------------
1. random       — random WA coords with assay coverage (inflated baseline)
2. far_random   — random WA coords ≥50 km from any known Mine/Deposit
3. true_nonmine — element-specific `no_<el>_sites.csv` (sites that are Mine/Deposit
                  for OTHER metals but where target metal is absent)
4. spatial      — south-trained agent, north test region (independent spatial holdout)

All negative samplers reject any candidate within 5 km of a test positive
(spatial-leakage buffer).

Usage::

    /group/pmc050/yding/miniconda3/envs/geochem/bin/python scripts/eval_comprehensive.py
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
logger = logging.getLogger("eval_comprehensive")

import json
import os

from core.catalog import DataCatalog, SourceSpec
from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX
from domains.geochem.samples import GeochemSample

# Expert-layer ablation: set ABLATE_LAYERS to a + separated subset of
# {geochem,geophys,structure}. Examples:
#   ABLATE_LAYERS=geochem            → only geochem experts
#   ABLATE_LAYERS=geochem+geophys    → geochem + geophysics (no structure)
#   (unset)                          → all three (baseline)
_ABLATE_RAW = os.environ.get("ABLATE_LAYERS", "geochem+geophys+structure").strip()
LAYERS = frozenset(l for l in _ABLATE_RAW.split("+") if l in {"geochem","geophys","structure"})
ABLATE_TAG = "_".join(sorted(LAYERS)) if len(LAYERS) < 3 else "full"

DATA_ROOT   = ROOT / "datasets"
GEOCHEM_DIR = DATA_ROOT / "geochemical" / "states" / "state_geochemical"
SITES_DIR   = DATA_ROOT / "geochemical" / "states" / "sites"
ENRICH_DIR  = DATA_ROOT / "geochemical" / "states" / "sites_enriched"
NO_DIR      = SITES_DIR / "no_gold"
RASTER_DIR  = DATA_ROOT / "selected_layers" / "rasters"
VECTOR_DIR  = DATA_ROOT / "selected_layers" / "vectors"

SEED        = 42
MAX_TEST    = 30    # held-out positives per scenario
N_NEG       = 200   # number of negatives per evaluation
SPATIAL_BOUNDARY = -25.0   # Y < -25 → south (train); Y >= -25 → north (test)
KM_PER_DEG  = 100.0        # approximate deg→km for distance calculations
FAR_KM      = 50.0         # minimum distance for "far_random" negatives
BUFFER_KM   = 5.0          # min distance from any test positive (leakage guard)

# ── Metal configurations ──────────────────────────────────────────────────────

_CONFIDENCE = {"Mine": 1.0, "Deposit": 0.9}

def _ec(el: str) -> str:
    return str(ENRICH_DIR / f"enriched_{el}_sites.csv")

def _nc(fname: str) -> str:
    return str(NO_DIR / fname)

METAL_CONFIGS = {
    "Cu": dict(
        sites_csv=_ec("Cu"), site_col="Cu_SITES", no_target_csv=_nc("no_cu_sites.csv"),
        pathfinders={"Au": 1.5, "Mo": 1.5, "Ag": 1.0, "Co": 0.8, "Bi": 0.8, "Pb": 0.5},
        ratio_features=["ratio_Cu_Mo", "ratio_Au_Cu", "ratio_Co_Ni"],
        confidence_map=_CONFIDENCE,
    ),
    "Au": dict(
        sites_csv=_ec("Au"), site_col="Au_SITES", no_target_csv=_nc("no_gold_sites.csv"),
        pathfinders={"As": 2.0, "Sb": 1.5, "Ag": 1.5, "Bi": 1.0, "W": 0.8, "Cu": 0.5},
        ratio_features=["ratio_Au_Cu", "ratio_Au_As", "ratio_Sb_As"],
        confidence_map=_CONFIDENCE,
    ),
    "Ni": dict(
        sites_csv=_ec("Ni"), site_col="Ni_SITES", no_target_csv=_nc("no_ni_sites.csv"),
        pathfinders={"Co": 2.0, "Cr": 1.5, "Cu": 1.0, "Au": 0.5},
        ratio_features=["ratio_Co_Ni", "ratio_Ni_Co", "ratio_Ni_Cr"],
        confidence_map=_CONFIDENCE,
    ),
    "W": dict(
        sites_csv=_ec("W"), site_col="W_SITES", no_target_csv=_nc("no_w_sites.csv"),
        pathfinders={"Mo": 2.0, "Bi": 1.5, "Sn": 1.5, "Cu": 1.0, "As": 0.8},
        ratio_features=["ratio_W_Mo", "ratio_W_Sn"],
        confidence_map=_CONFIDENCE,
    ),
    "Sn": dict(
        sites_csv=_ec("Sn"), site_col="Sn_SITES", no_target_csv=_nc("no_sn_sites.csv"),
        pathfinders={"W": 1.8, "Mo": 1.0, "Bi": 1.0, "Li": 0.8, "Ta": 0.8, "Cu": 0.5},
        ratio_features=["ratio_Sn_W", "ratio_Sn_Mo"],
        confidence_map=_CONFIDENCE,
    ),
    "Co": dict(
        sites_csv=_ec("Co"), site_col="Co_SITES", no_target_csv=_nc("no_co_sites.csv"),
        pathfinders={"Ni": 2.0, "Cu": 1.2, "Cr": 1.0, "Mn": 0.8, "Fe": 0.5},
        ratio_features=["ratio_Co_Ni", "ratio_Co_Mn"],
        confidence_map=_CONFIDENCE,
    ),
    "Ta": dict(
        sites_csv=_ec("Ta"), site_col="Ta_SITES", no_target_csv=_nc("no_ta_sites.csv"),
        pathfinders={"Nb": 2.0, "Li": 1.5, "Sn": 1.0, "Cs": 0.8, "Be": 0.5},
        ratio_features=["ratio_Ta_Nb", "ratio_Li_Ta"],
        confidence_map=_CONFIDENCE,
    ),
    "Mn": dict(
        sites_csv=_ec("Mn"), site_col="Mn_SITES", no_target_csv=_nc("no_mn_sites.csv"),
        pathfinders={"Fe": 1.5, "Co": 1.0, "Ni": 0.8, "Ba": 0.5},
        ratio_features=["ratio_Mn_Fe", "ratio_Co_Mn"],
        confidence_map=_CONFIDENCE,
    ),
    "REE": dict(
        sites_csv=_ec("REE"), site_col="REE_SITES", no_target_csv=_nc("no_ree_sites.csv"),
        pathfinders={"Th": 1.5, "U": 1.0, "P": 1.0, "Y": 1.0, "Sr": 0.8, "Ba": 0.8},
        ratio_features=["ratio_Th_U"],
        confidence_map=_CONFIDENCE,
    ),
}

ALL_METALS = list(METAL_CONFIGS.keys())


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


# ── Site loading helpers ──────────────────────────────────────────────────────

def normalise_sites_df(df: pd.DataFrame) -> pd.DataFrame:
    if "SITE_CODE" not in df.columns and "fid" in df.columns:
        df = df.rename(columns={"fid": "SITE_CODE"})
    df["SITE_CODE"] = df["SITE_CODE"].astype(str).str.strip()
    return df.dropna(subset=["X", "Y"])


def load_mine_deposit_samples(sites_csv: str, site_col: str,
                               confidence_map: dict) -> list[GeochemSample]:
    df = normalise_sites_df(pd.read_csv(sites_csv))
    samples = []
    for _, row in df.iterrows():
        val = str(row.get(site_col, ""))
        for kw in confidence_map:
            if kw in val:
                samples.append(GeochemSample(
                    site_code=str(row["SITE_CODE"]),
                    x=float(row["X"]), y=float(row["Y"]),
                ))
                break
    return samples


def load_no_target_sites(no_csv: str, exclude_ids: set,
                          forbid_xy_km: Optional[np.ndarray] = None,
                          buffer_km: float = BUFFER_KM) -> list[GeochemSample]:
    """Load sites where target element is absent but the site is a Mine/Deposit
    for some OTHER metal. Reads no_<el>_sites.csv (lowercase schema). Applies
    5 km buffer against test positives if `forbid_xy_km` is given."""
    df = pd.read_csv(no_csv)
    # lowercase schema: site_code, X, Y
    df["site_code"] = df["site_code"].astype(str).str.strip()
    df = df.dropna(subset=["X", "Y"])
    df = df[~df["site_code"].isin(exclude_ids)]
    out: list[GeochemSample] = []
    tree = cKDTree(forbid_xy_km) if forbid_xy_km is not None and len(forbid_xy_km) else None
    for _, r in df.iterrows():
        x, y = float(r["X"]), float(r["Y"])
        if tree is not None:
            d, _ = tree.query([x * KM_PER_DEG, y * KM_PER_DEG], k=1)
            if d < buffer_km:
                continue
        out.append(GeochemSample(site_code=str(r["site_code"]), x=x, y=y))
    return out


def _forbid_xy_km(test_pos: list[GeochemSample]) -> np.ndarray:
    """Return Nx2 km-scale coords for spatial-buffer queries (or empty)."""
    if not test_pos:
        return np.empty((0, 2))
    return np.array([[p.x * KM_PER_DEG, p.y * KM_PER_DEG] for p in test_pos])


def build_all_mine_tree() -> cKDTree:
    """KD-tree of all Mine/Deposit sites across all 4 metals (for far_random check)."""
    rows = []
    for metal, cfg in METAL_CONFIGS.items():
        df = normalise_sites_df(pd.read_csv(cfg["sites_csv"]))
        mine = df[df[cfg["site_col"]].fillna("").str.contains("Mine|Deposit")]
        for _, r in mine.iterrows():
            rows.append([float(r["X"]) * KM_PER_DEG, float(r["Y"]) * KM_PER_DEG])
    xy = np.array(rows)
    return cKDTree(xy)


def build_all_sites_combined() -> pd.DataFrame:
    """Unified X,Y,SITE_CODE frame from all 4 metal CSVs."""
    frames = []
    for metal, cfg in METAL_CONFIGS.items():
        df = normalise_sites_df(pd.read_csv(cfg["sites_csv"]))
        frames.append(df[["X", "Y", "SITE_CODE"]])
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["X", "Y"])
        .dropna(subset=["X", "Y"])
        .reset_index(drop=True)
    )


# ── Negative generators ───────────────────────────────────────────────────────

def gen_random_negatives(agent: GeochemAgent, n: int, seed: int,
                          forbid_xy_km: Optional[np.ndarray] = None,
                          buffer_km: float = BUFFER_KM) -> list[GeochemSample]:
    """Random WA points with at least one assay source coverage and ≥buffer_km
    from any test positive (if `forbid_xy_km` is given)."""
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = WA_BBOX
    forbid_tree = cKDTree(forbid_xy_km) if forbid_xy_km is not None and len(forbid_xy_km) else None
    bg: list[GeochemSample] = []
    attempts = 0
    while len(bg) < n and attempts < n * 500:
        attempts += 1
        x, y = float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))
        if forbid_tree is not None:
            d, _ = forbid_tree.query([x * KM_PER_DEG, y * KM_PER_DEG], k=1)
            if d < buffer_km:
                continue
        for src_name in agent.active_sources():
            try:
                src = agent._registry.get(src_name)
                if hasattr(src, "has_coverage") and src.has_coverage(x, y):
                    bg.append(GeochemSample(site_code=f"RBG_{len(bg):04d}", x=x, y=y))
                    break
            except KeyError:
                pass
    return bg


def gen_far_random_negatives(
    agent: GeochemAgent,
    mine_tree: cKDTree,
    n: int,
    seed: int,
    min_km: float = FAR_KM,
    forbid_xy_km: Optional[np.ndarray] = None,
    buffer_km: float = BUFFER_KM,
) -> list[GeochemSample]:
    """Random WA points: assay coverage AND ≥min_km from any known Mine/Deposit
    AND ≥buffer_km from any test positive."""
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = WA_BBOX
    forbid_tree = cKDTree(forbid_xy_km) if forbid_xy_km is not None and len(forbid_xy_km) else None
    bg: list[GeochemSample] = []
    attempts = 0
    while len(bg) < n and attempts < n * 2000:
        attempts += 1
        x, y = float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))
        if mine_tree.query([x * KM_PER_DEG, y * KM_PER_DEG], k=1)[0] < min_km:
            continue
        if forbid_tree is not None:
            d, _ = forbid_tree.query([x * KM_PER_DEG, y * KM_PER_DEG], k=1)
            if d < buffer_km:
                continue
        for src_name in agent.active_sources():
            try:
                src = agent._registry.get(src_name)
                if hasattr(src, "has_coverage") and src.has_coverage(x, y):
                    bg.append(GeochemSample(site_code=f"FAR_{len(bg):04d}", x=x, y=y))
                    break
            except KeyError:
                pass
    return bg


def gen_spatially_matched_negatives(
    test_pos: list[GeochemSample],
    all_mine_ids: set,
    all_sites_df: pd.DataFrame,
    radius_km: float,
    seed: int,
) -> list[GeochemSample]:
    """One non-mine site within radius_km per test positive (fallback: nearest)."""
    non_mine = all_sites_df[~all_sites_df["SITE_CODE"].isin(all_mine_ids)].reset_index(drop=True)
    neg_xy = np.column_stack([non_mine.X.values * KM_PER_DEG,
                               non_mine.Y.values * KM_PER_DEG])
    tree = cKDTree(neg_xy)
    rng = np.random.default_rng(seed)
    matched: list[GeochemSample] = []
    used: set[int] = set()
    for pos in test_pos:
        cands = [i for i in tree.query_ball_point(
            [pos.x * KM_PER_DEG, pos.y * KM_PER_DEG], r=radius_km
        ) if i not in used]
        if not cands:
            _, idxs = tree.query([pos.x * KM_PER_DEG, pos.y * KM_PER_DEG], k=10)
            cands = [i for i in (idxs if np.ndim(idxs) > 0 else [idxs]) if i not in used]
        if not cands:
            continue
        pick = int(rng.choice(cands))
        used.add(pick)
        row = non_mine.iloc[pick]
        matched.append(GeochemSample(
            site_code=str(row["SITE_CODE"]), x=float(row["X"]), y=float(row["Y"])
        ))
    return matched


def gen_region_negatives(
    region_mask: pd.Series,
    all_sites_df: pd.DataFrame,
    mine_ids: set,
    n: int,
    seed: int,
    forbid_xy_km: Optional[np.ndarray] = None,
    buffer_km: float = BUFFER_KM,
) -> list[GeochemSample]:
    """Non-mine sites from a specific geographic region with 5 km buffer
    against test positives."""
    pool = all_sites_df[region_mask & ~all_sites_df["SITE_CODE"].isin(mine_ids)].copy()
    if forbid_xy_km is not None and len(forbid_xy_km):
        tree = cKDTree(forbid_xy_km)
        xy_km = np.column_stack([pool["X"].values * KM_PER_DEG,
                                  pool["Y"].values * KM_PER_DEG])
        d, _ = tree.query(xy_km, k=1)
        pool = pool[d >= buffer_km]
    rng = np.random.default_rng(seed)
    if len(pool) > n:
        pool = pool.sample(n, random_state=int(rng.integers(0, 9999)))
    return [
        GeochemSample(site_code=str(r["SITE_CODE"]), x=float(r["X"]), y=float(r["Y"]))
        for _, r in pool.iterrows()
    ]


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval(agent: GeochemAgent, test_pos: list, negatives: list) -> Optional[dict]:
    all_test = test_pos + negatives
    labels   = [1] * len(test_pos) + [0] * len(negatives)
    ns_list  = agent.score_batch(all_test)
    pairs    = [(ns.score, lbl) for ns, lbl in zip(ns_list, labels) if ns is not None]
    if len(pairs) < 10 or len({lbl for _, lbl in pairs}) < 2:
        return None
    y_true = [lbl for _, lbl in pairs]
    y_pred = [sc  for sc,  _ in pairs]
    auc = float(roc_auc_score(y_true, y_pred))
    paired_sorted = sorted(zip(y_pred, y_true), reverse=True)
    prec = {}
    for k in (10, 20, 50):
        if k <= len(paired_sorted):
            prec[k] = sum(lbl for _, lbl in paired_sorted[:k]) / k
    return {
        "auc": auc,
        "n": len(pairs),
        "n_pos": sum(y_true),
        "n_neg": len(y_true) - sum(y_true),
        "prec": prec,
        # Raw per-sample (score, label) pairs for downstream ROC / score-dist plots
        "y_pred": y_pred,
        "y_true": y_true,
    }


def print_result(label: str, r: Optional[dict]) -> None:
    if r is None:
        print(f"  {label:<22s}  —  (insufficient data)")
        return
    p = r["prec"]
    p10 = f"{p[10]:.2f}" if 10 in p else "  — "
    p20 = f"{p[20]:.2f}" if 20 in p else "  — "
    print(f"  {label:<22s}  AUC={r['auc']:.3f}  "
          f"n=({r['n_pos']}+/{r['n_neg']}-)"
          f"  P@10={p10}  P@20={p20}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    catalog   = build_catalog()
    mine_tree = build_all_mine_tree()
    all_sites = build_all_sites_combined()

    # Collect all Mine/Deposit SITE_CODEs across all metals (for negative filtering)
    all_mine_ids: set[str] = set()
    for cfg in METAL_CONFIGS.values():
        df = normalise_sites_df(pd.read_csv(cfg["sites_csv"]))
        ids = df[df[cfg["site_col"]].fillna("").str.contains("Mine|Deposit")]["SITE_CODE"]
        all_mine_ids.update(ids.astype(str))

    summary: dict[str, dict] = {}

    for metal, cfg in METAL_CONFIGS.items():
        print(f"\n{'='*72}")
        print(f"  {metal}  Prospectivity — 4 negative strategies")
        print(f"{'='*72}")

        all_pos = load_mine_deposit_samples(cfg["sites_csv"], cfg["site_col"],
                                            cfg["confidence_map"])
        all_pos_ids = {s.id for s in all_pos}
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(all_pos))

        # ── shared agent (random split) ────────────────────────────────────
        n_test = min(MAX_TEST, max(0, len(all_pos) - 10))
        test_pos_rand = [all_pos[i] for i in perm[:n_test]]
        test_ids_rand = {s.id for s in test_pos_rand}
        forbid_xy_km  = _forbid_xy_km(test_pos_rand)
        print(f"  Total positives: {len(all_pos)}   "
              f"test={len(test_pos_rand)}  train={len(all_pos)-len(test_pos_rand)}")

        tc = GeochemTargetConfig(
            target=metal,
            pathfinders=cfg["pathfinders"],
            ratio_features=cfg["ratio_features"],
            confidence_map=cfg["confidence_map"],
            sites_csv=cfg["sites_csv"],
            layers=LAYERS,
        )
        agent = GeochemAgent(catalog=catalog, target_config=tc,
                             n_bg=300, bbox=WA_BBOX, seed=SEED)
        agent.plan(check_bbox=WA_BBOX)
        agent.setup(exclude_ids=test_ids_rand)
        print(f"  Active experts: {len(agent.active_experts())}")

        # ── 1. Random ─────────────────────────────────────────────────────
        rand_neg = gen_random_negatives(agent, N_NEG, seed=SEED + 1,
                                         forbid_xy_km=forbid_xy_km)
        r_random = run_eval(agent, test_pos_rand, rand_neg)

        # ── 2. Far random (≥50 km from any mine) ──────────────────────────
        far_neg = gen_far_random_negatives(agent, mine_tree, N_NEG, seed=SEED + 2,
                                            forbid_xy_km=forbid_xy_km)
        r_far = run_eval(agent, test_pos_rand, far_neg)
        print(f"  Far random negatives found: {len(far_neg)}")

        # ── 3. True non-mine (no_<el>_sites.csv — same area, target absent) ──
        nt_neg = load_no_target_sites(cfg["no_target_csv"], all_pos_ids,
                                       forbid_xy_km=forbid_xy_km)
        rng3 = np.random.default_rng(SEED + 3)
        if len(nt_neg) > N_NEG:
            idxs = rng3.choice(len(nt_neg), N_NEG, replace=False)
            nt_neg = [nt_neg[i] for i in idxs]
        r_fail = run_eval(agent, test_pos_rand, nt_neg)
        print(f"  No-target negatives: {len(nt_neg)}")

        # ── 4. Spatial holdout — south train, north test ──────────────────
        df_full = normalise_sites_df(pd.read_csv(cfg["sites_csv"]))
        mine_df = df_full[df_full[cfg["site_col"]].fillna("").str.contains("Mine|Deposit")]
        north_pos_all = [
            GeochemSample(site_code=str(r["SITE_CODE"]), x=float(r["X"]), y=float(r["Y"]))
            for _, r in mine_df[mine_df.Y >= SPATIAL_BOUNDARY].iterrows()
        ]
        south_pos_all = [
            GeochemSample(site_code=str(r["SITE_CODE"]), x=float(r["X"]), y=float(r["Y"]))
            for _, r in mine_df[mine_df.Y < SPATIAL_BOUNDARY].iterrows()
        ]
        n_north_test = min(MAX_TEST, len(north_pos_all))
        rng4 = np.random.default_rng(SEED + 4)
        if len(north_pos_all) > n_north_test:
            pick = rng4.choice(len(north_pos_all), n_north_test, replace=False)
            test_pos_north = [north_pos_all[i] for i in pick]
        else:
            test_pos_north = list(north_pos_all)
        north_test_ids = {s.id for s in test_pos_north}
        all_north_ids  = {s.id for s in north_pos_all}

        r_spatial = None
        if len(south_pos_all) >= 10 and len(test_pos_north) >= 5:
            agent_sp = GeochemAgent(catalog=catalog, target_config=tc,
                                    n_bg=300, bbox=WA_BBOX, seed=SEED + 10)
            agent_sp.plan(check_bbox=WA_BBOX)
            agent_sp.setup(exclude_ids=all_north_ids)  # train on south only

            # negatives: non-mine sites in the north region (with 5 km buffer)
            north_mask = all_sites.Y >= SPATIAL_BOUNDARY
            sp_neg = gen_region_negatives(north_mask, all_sites, all_mine_ids, N_NEG,
                                          seed=SEED + 5,
                                          forbid_xy_km=_forbid_xy_km(test_pos_north))
            r_spatial = run_eval(agent_sp, test_pos_north, sp_neg)
            print(f"  Spatial split — south train: {len(south_pos_all)}"
                  f"  north test: {len(test_pos_north)}"
                  f"  north neg: {len(sp_neg)}")
        else:
            print(f"  Spatial split — skipped (south={len(south_pos_all)}, "
                  f"north={len(north_pos_all)})")

        # ── Results ───────────────────────────────────────────────────────
        print()
        print_result("1. random",     r_random)
        print_result("2. far_random",  r_far)
        print_result("3. true_nonmine", r_fail)
        print_result("4. spatial",     r_spatial)

        summary[metal] = {
            "random":    r_random,
            "far_random": r_far,
            "true_nonmine": r_fail,
            "spatial":   r_spatial,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    strats = ["random", "far_random", "true_nonmine", "spatial"]
    labels = ["Random", "Far(>50km)", "TrueNonMine", "Spatial"]
    print(f"\n{'='*72}")
    print("SUMMARY — AUC by metal × negative strategy")
    print(f"{'='*72}")
    print(f"  {'Metal':<6}  " + "  ".join(f"{l:>12}" for l in labels))
    print(f"  {'-'*6}  " + "  ".join(f"{'-'*12}" for _ in labels))
    for metal in ALL_METALS:
        row = []
        for st in strats:
            r = summary[metal][st]
            row.append(f"{r['auc']:>12.3f}" if r else f"{'—':>12}")
        print(f"  {metal:<6}  " + "  ".join(row))

    # ── Dump JSON summary for HTML reporter ───────────────────────────────────
    suffix = "" if ABLATE_TAG == "full" else f"_{ABLATE_TAG}"
    out_json = ROOT / "reports" / f"eval_comprehensive_summary{suffix}.json"
    out_json.parent.mkdir(exist_ok=True)
    with out_json.open("w") as fh:
        json.dump({
            "metals": ALL_METALS,
            "strategies": strats,
            "results": {
                m: {s: (None if r is None else {
                    "auc":   r["auc"],
                    "n_pos": r["n_pos"],
                    "n_neg": r["n_neg"],
                    "prec":  {str(k): v for k, v in r["prec"].items()},
                }) for s, r in d.items()}
                for m, d in summary.items()
            },
            "config": {"MAX_TEST": MAX_TEST, "N_NEG": N_NEG,
                       "BUFFER_KM": BUFFER_KM, "FAR_KM": FAR_KM, "SEED": SEED},
        }, fh, indent=2)
    print(f"\n  JSON summary: {out_json}")

    # ── Dump raw per-sample scores for ROC / score-dist downstream plots ──────
    out_scores = ROOT / "reports" / f"eval_scores{suffix}.json"
    with out_scores.open("w") as fh:
        json.dump({
            m: {s: (None if r is None else {
                "y_pred": r["y_pred"],
                "y_true": r["y_true"],
                "auc":    r["auc"],
            }) for s, r in d.items()}
            for m, d in summary.items()
        }, fh)
    print(f"  Scores: {out_scores}")


if __name__ == "__main__":
    main()
