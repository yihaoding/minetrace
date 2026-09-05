"""Held-out evaluation of GeochemAgent for Cu prospectivity.

Splits the KB positives into train (80%) and test (20%), fits the agent
on the train set, then scores held-out positives + random negatives and
reports per-expert AUC and composite AUC.

Usage::

    cd <project root>
    /group/pmc050/yding/miniconda3/envs/geochem/bin/python scripts/eval_geochem_agent.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("eval_geochem_agent")

from core.catalog import DataCatalog, SourceSpec
from domains.geochem.geochem_agent import GeochemAgent, GeochemTargetConfig, WA_BBOX
from domains.geochem.samples import GeochemInstance, GeochemSample

DATA_ROOT   = ROOT / "datasets"
GEOCHEM_DIR = DATA_ROOT / "geochemical" / "states" / "state_geochemical"
SITES_DIR   = DATA_ROOT / "geochemical" / "states" / "sites"
RASTER_DIR  = DATA_ROOT / "selected_layers" / "rasters"
VECTOR_DIR  = DATA_ROOT / "selected_layers" / "vectors"
SITES_CSV   = str(SITES_DIR / "gswa_cu_sites.csv")
ALL_SITES_CSVS = {
    "Cu": str(SITES_DIR / "gswa_cu_sites.csv"),
    "Au": str(SITES_DIR / "gswa_au_site.csv"),
    "Ni": str(SITES_DIR / "gswa_ni_site.csv"),
    "W":  str(SITES_DIR / "gswa_w_site.csv"),
}

N_TEST_POS = 100   # held-out positives (random split)
N_TEST_NEG = 300   # negatives
SEED       = 42

# Spatial validation: north WA (Pilbara + Murchison + Paterson) as test region
SPATIAL_TEST_Y_MIN = -25.0   # Y > -25 → test region (northern WA)

# Strict negative: other-metal Mine/Deposit sites with no Cu Mine/Deposit/Prospect
STRICT_NEG_METALS = ["Au_SITES", "Ni_SITES", "Zn_SITES", "Pb_SITES",
                     "W_SITES", "Ag_SITES", "PGE_SITES"]


def build_catalog() -> DataCatalog:
    catalog = DataCatalog()
    assay_specs = [
        ("sediment",    "sediment",    "gswa_all_sediment.csv",          "STREA"),
        ("rockchips",   "rockchip",    "gswa_all_rockchips.csv",         "ROCKC"),
        ("drillhole",   "drillhole",   "gswa_all_drillhole_maxgrade.csv","DRILL"),
        ("shallowdrill","shallowdrill","gswa_all_shallowdrill.csv",       "SHALL"),
        ("soil",        "soil",        "gswa_all_surfsoilgeochem.csv",    "SOIL"),
    ]
    for name, subtype, fname, st_filter in assay_specs:
        catalog.register(SourceSpec(
            name=name, source_type="assay_spatial", modality="geochemistry",
            subtype=subtype, path=str(GEOCHEM_DIR / fname),
            loader_kwargs={"sample_type_filter": st_filter},
        ))
    raster_specs = [
        ("mag","magnetics","WA_80m_Mag_Merge_1VD_v1_2020.ers",0.08),
        ("grav","gravity","WA_400m_Grav_Merge_v1_2020.ers",0.40),
        ("K","radiometrics","WA_80m_K_Merge_v1_2018.ers",0.08),
        ("Th","radiometrics","WA_80m_Th_Merge_v1_2018.ers",0.08),
        ("U","radiometrics","WA_80m_U_Merge_v1_2018.ers",0.08),
        ("LuHf","geochronology","WA_LuHf_TDM2_masked.bil",0.10),
        ("SmNd","geochronology","WA_SmNd_TDM2_masked.bil",0.10),
    ]
    for rkey, subdir, fname, res in raster_specs:
        catalog.register(SourceSpec(
            name=f"raster_{rkey}", source_type="raster", modality="geophysics",
            subtype=rkey, path=str(RASTER_DIR / subdir / fname),
            resolution_km=res, loader_kwargs={"raster_key": rkey},
        ))
    vector_specs = [
        ("fault","structures","500k_interpstrucl20.shp"),
        ("worm_mag","geophysics_worms","worm_mag.shp"),
        ("worm_grav","geophysics_worms","worm_grav.shp"),
        ("geology","geology_context","GeologyMERGED.shp"),
        ("cenozoic","geology_context","500k_cenozoicp20.shp"),
    ]
    for layer_key, subdir, fname in vector_specs:
        catalog.register(SourceSpec(
            name=f"vector_{layer_key}", source_type="vector", modality="geology",
            subtype=layer_key, path=str(VECTOR_DIR / subdir / fname),
            loader_kwargs={"layer_key": layer_key},
        ))
    return catalog


def load_all_positives() -> list[GeochemSample]:
    df = pd.read_csv(SITES_CSV)
    col = "Cu_SITES"
    conf_map = {"Mine": 1.0, "Deposit": 0.9}
    samples = []
    for _, row in df.iterrows():
        val = str(row.get(col, ""))
        for kw in conf_map:
            if kw in val:
                sid = str(row.get("SITE_CODE", row.get("site_code", "")))
                x = float(row.get("X", row.get("x", 0)))
                y = float(row.get("Y", row.get("y", 0)))
                samples.append(GeochemSample(site_code=sid, x=x, y=y))
                break
    return samples


def load_all_sites_combined() -> pd.DataFrame:
    """Merge Cu/Au/Ni/W site CSVs into a unified X,Y,SITE_CODE frame."""
    frames = []
    for metal, path in ALL_SITES_CSVS.items():
        df = pd.read_csv(path)
        # normalise SITE_CODE column (Au file uses 'fid')
        if "SITE_CODE" not in df.columns and "fid" in df.columns:
            df = df.rename(columns={"fid": "SITE_CODE"})
        df["SITE_CODE"] = df["SITE_CODE"].astype(str).str.strip()
        df["_metal"] = metal
        frames.append(df[["X", "Y", "SITE_CODE", "_metal"]])
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["X", "Y"]
    ).dropna(subset=["X", "Y"])
    logger.info("Combined sites: %d unique locations from Cu/Au/Ni/W", len(combined))
    return combined


def load_strict_negatives(sites_csv: str, pos_ids: set, n: int, seed: int) -> list[GeochemSample]:
    """Strict negatives: Cu Prospect/Occurrence sites that were drilled but never developed.

    These are genuine Cu exploration failures — someone targeted Cu, drilled,
    and found nothing economic. Same geological context as positives, confirmed
    to lack economic Cu grades.
    """
    df = pd.read_csv(sites_csv)
    cu_explored   = df["Cu_SITES"].fillna("").str.contains("Prospect|Occurrence")
    was_drilled   = df["SITE_SUB_T"] == "Drillhole"
    not_developed = df["SITE_STAGE"] == "Undeveloped"
    no_cu_econ    = ~df["Cu_SITES"].fillna("").str.contains("Mine|Deposit")
    cands = df[cu_explored & was_drilled & not_developed & no_cu_econ].copy()
    cands = cands[~cands["SITE_CODE"].isin(pos_ids)]
    logger.info("Strict negative candidates (Cu drillhole failures): %d", len(cands))
    rng = np.random.default_rng(seed + 99)
    if len(cands) > n:
        cands = cands.sample(n, random_state=int(rng.integers(0, 9999)))
    samples = [
        GeochemSample(site_code=str(row["SITE_CODE"]), x=float(row["X"]), y=float(row["Y"]))
        for _, row in cands.iterrows()
    ]
    logger.info("Using %d strict negatives (Cu drillhole, Undeveloped)", len(samples))
    return samples


def load_spatially_matched_negatives(
    sites_csv: str,
    test_pos: list,
    all_pos_ids: set,
    radius_km: float = 50.0,
    seed: int = 42,
    all_sites_df: pd.DataFrame = None,
) -> list[GeochemSample]:
    """For each test positive, sample one non-mine site within radius_km.

    Uses combined all-metal sites if provided, otherwise falls back to Cu CSV.
    Falls back to nearest available non-mine site if none within radius.
    """
    from scipy.spatial import cKDTree
    if all_sites_df is not None:
        # use combined all-metal sites; exclude known Cu Mine/Deposit
        cu_df = pd.read_csv(sites_csv)
        cu_mine_codes = set(
            cu_df[cu_df["Cu_SITES"].fillna("").str.contains("Mine|Deposit")].SITE_CODE.astype(str)
        )
        non_pos = all_sites_df[~all_sites_df["SITE_CODE"].isin(cu_mine_codes)].copy()
    else:
        df = pd.read_csv(sites_csv)
        non_pos = df[~df["Cu_SITES"].fillna("").str.contains("Mine|Deposit")].dropna(
            subset=["X", "Y"]
        ).copy()
    non_pos = non_pos[~non_pos["SITE_CODE"].isin(all_pos_ids)].reset_index(drop=True)

    KM_PER_DEG = 100.0
    neg_xy = np.column_stack([non_pos.X.values * KM_PER_DEG,
                               non_pos.Y.values * KM_PER_DEG])
    tree = cKDTree(neg_xy)

    rng = np.random.default_rng(seed + 7)
    matched: list[GeochemSample] = []
    used_idx: set = set()
    fallback_count = 0

    for pos_sample in test_pos:
        px = pos_sample.x * KM_PER_DEG
        py = pos_sample.y * KM_PER_DEG
        candidates = tree.query_ball_point([px, py], r=radius_km)
        candidates = [i for i in candidates if i not in used_idx]

        if candidates:
            pick = int(rng.choice(candidates))
        else:
            # fallback: nearest not yet used
            fallback_count += 1
            dists, idxs = tree.query([px, py], k=min(20, len(non_pos)))
            available = [i for i in (idxs if np.ndim(idxs) > 0 else [idxs])
                         if i not in used_idx]
            if not available:
                continue
            pick = available[0]

        used_idx.add(pick)
        row = non_pos.iloc[pick]
        matched.append(GeochemSample(
            site_code=str(row["SITE_CODE"]),
            x=float(row["X"]), y=float(row["Y"]),
        ))

    logger.info(
        "Spatially matched negatives: %d  (radius=%.0f km, %d fallbacks)",
        len(matched), radius_km, fallback_count,
    )
    return matched


def sample_random_background(agent: GeochemAgent, n: int, seed: int) -> list[GeochemSample]:
    """Random WA background points with assay coverage."""
    rng = np.random.default_rng(seed + 1)
    x_min, x_max, y_min, y_max = WA_BBOX
    bg: list[GeochemSample] = []
    attempts = 0
    while len(bg) < n and attempts < n * 500:
        attempts += 1
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        covered = False
        for src_name in agent.active_sources():
            try:
                src = agent._registry.get(src_name)
                if hasattr(src, "has_coverage") and src.has_coverage(x, y):
                    covered = True
                    break
            except KeyError:
                pass
        if covered:
            bg.append(GeochemSample(site_code=f"BG_{len(bg):04d}", x=x, y=y))
    logger.info("Sampled %d random background points in %d attempts", len(bg), attempts)
    return bg


def evaluate(agent, test_pos, neg_samples, label: str) -> None:
    """Score test_pos + neg_samples, print AUC and Precision@K."""
    all_test = test_pos + neg_samples
    labels   = [1] * len(test_pos) + [0] * len(neg_samples)
    logger.info("[%s] Scoring %d samples (%d pos, %d neg) …",
                label, len(all_test), len(test_pos), len(neg_samples))
    node_scores = agent.score_batch(all_test)

    scored_pairs = [(ns, lbl) for ns, lbl in zip(node_scores, labels) if ns is not None]
    if len(scored_pairs) < 10:
        print(f"[{label}] Too few scored samples ({len(scored_pairs)}). Skipping.")
        return

    y_true = [lbl for _, lbl in scored_pairs]
    y_pred = [ns.score for ns, _ in scored_pairs]
    auc = float(roc_auc_score(y_true, y_pred))

    print(f"\n=== {label} ===")
    print(f"  Composite AUC : {auc:.3f}  (n={len(scored_pairs)}, "
          f"{sum(y_true)} pos / {len(y_true)-sum(y_true)} neg)")

    # per-expert AUC (only experts that score both classes)
    exp_scores: dict[str, list] = {}
    exp_labels: dict[str, list] = {}
    for ns, lbl in scored_pairs:
        for exp_name, child_ns in ns.children.items():
            exp_scores.setdefault(exp_name, []).append(child_ns.score)
            exp_labels.setdefault(exp_name, []).append(lbl)

    rows = []
    for name in sorted(exp_scores):
        ys, yl = exp_scores[name], exp_labels[name]
        if len(set(yl)) < 2 or len(ys) < 5:
            continue
        rows.append((float(roc_auc_score(yl, ys)), name, len(ys)))

    if rows:
        print(f"\n  {'Expert':<40s}  {'AUC':>6s}  {'n':>5s}")
        print(f"  {'-'*40}  {'-'*6}  {'-'*5}")
        for expert_auc, name, n in sorted(rows, reverse=True):
            print(f"  {name:<40s}  {expert_auc:.3f}  {n:>5d}")

    # precision@K
    print()
    paired = sorted(zip(y_pred, y_true), reverse=True)
    for k in (10, 20, 50):
        if k > len(paired):
            continue
        top_k = [lbl for _, lbl in paired[:k]]
        print(f"  Precision@{k:<3d}: {sum(top_k)/k:.2f}  ({sum(top_k)}/{k})")


def main() -> None:
    rng = np.random.default_rng(SEED)

    # 1. Load all positives and split train/test
    all_pos = load_all_positives()
    all_pos_ids = {s.id for s in all_pos}
    logger.info("Total KB positives: %d", len(all_pos))
    idx      = rng.permutation(len(all_pos))
    test_pos = [all_pos[i] for i in idx[:N_TEST_POS]]
    test_ids = {s.id for s in test_pos}
    logger.info("Held-out test positives: %d", len(test_pos))

    # 2. Build catalog and agent
    catalog = build_catalog()
    tc = GeochemTargetConfig(
        target="Cu",
        pathfinders={"Au": 1.5, "Mo": 1.5, "Ag": 1.0, "Co": 0.8, "Bi": 0.8, "Pb": 0.5},
        ratio_features=["ratio_Cu_Mo", "ratio_Au_Cu", "ratio_Co_Ni"],
        confidence_map={"Mine": 1.0, "Deposit": 0.9},
        sites_csv=SITES_CSV,
    )
    agent = GeochemAgent(catalog=catalog, target_config=tc, n_bg=300, bbox=WA_BBOX, seed=SEED)

    # 3. Plan + setup (excluding test positives from training)
    logger.info("Planning …")
    agent.plan(check_bbox=WA_BBOX)
    logger.info("Setting up (excluding %d test sites from training) …", len(test_ids))
    agent.setup(exclude_ids=test_ids)
    logger.info("Active sources : %s", agent.active_sources())

    # 4. Build negative sets
    all_sites   = load_all_sites_combined()
    strict_neg  = load_strict_negatives(SITES_CSV, all_pos_ids, N_TEST_NEG, seed=SEED)
    random_neg  = sample_random_background(agent, N_TEST_NEG, seed=SEED)
    matched_neg = load_spatially_matched_negatives(
        SITES_CSV, test_pos, all_pos_ids, radius_km=50.0, seed=SEED,
        all_sites_df=all_sites,
    )

    # 5. Random-split evaluations (3 negative types)
    evaluate(agent, test_pos, random_neg,  "Random background negatives (baseline)")
    evaluate(agent, test_pos, strict_neg,  "Strict negatives (Cu drillhole failures)")
    evaluate(agent, test_pos, matched_neg, "Spatially matched negatives (within 50 km)")

    # 6. Spatial validation: train south, test north
    print("\n" + "="*60)
    print("SPATIAL VALIDATION")
    print(f"  Train region: Y < {SPATIAL_TEST_Y_MIN}  (Yilgarn + SW WA)")
    print(f"  Test  region: Y > {SPATIAL_TEST_Y_MIN}  (Pilbara + Murchison + Paterson)")
    print("="*60)

    all_pos_df = pd.read_csv(SITES_CSV)
    all_pos_df = all_pos_df[all_pos_df["Cu_SITES"].fillna("").str.contains("Mine|Deposit")]

    south_pos = [
        GeochemSample(site_code=str(r.SITE_CODE), x=float(r.X), y=float(r.Y))
        for _, r in all_pos_df[all_pos_df.Y < SPATIAL_TEST_Y_MIN].iterrows()
    ]
    north_pos = [
        GeochemSample(site_code=str(r.SITE_CODE), x=float(r.X), y=float(r.Y))
        for _, r in all_pos_df[all_pos_df.Y >= SPATIAL_TEST_Y_MIN].iterrows()
    ]
    north_ids = {s.id for s in north_pos}
    logger.info("Spatial split — train: %d south  |  test: %d north",
                len(south_pos), len(north_pos))

    # re-fit agent on south-only positives
    agent_spatial = GeochemAgent(
        catalog=catalog, target_config=tc, n_bg=300, bbox=WA_BBOX, seed=SEED + 1
    )
    agent_spatial.plan(check_bbox=WA_BBOX)
    agent_spatial.setup(exclude_ids=north_ids)

    # negatives for spatial test: all-sites within north region, no Cu Mine/Deposit
    north_all = all_sites[all_sites.Y >= SPATIAL_TEST_Y_MIN]
    north_non_cu = north_all[~north_all["SITE_CODE"].isin(
        set(all_pos_df.SITE_CODE.astype(str))
    )]
    rng2 = np.random.default_rng(SEED + 5)
    if len(north_non_cu) > N_TEST_NEG:
        north_non_cu = north_non_cu.sample(N_TEST_NEG,
                                            random_state=int(rng2.integers(0, 9999)))
    spatial_neg = [
        GeochemSample(site_code=str(r.SITE_CODE), x=float(r.X), y=float(r.Y))
        for _, r in north_non_cu.iterrows()
    ]
    logger.info("Spatial negatives (north region non-Cu-mine sites): %d", len(spatial_neg))

    evaluate(agent_spatial, north_pos, spatial_neg,
             f"Spatial validation — north WA test (Y > {SPATIAL_TEST_Y_MIN})")


if __name__ == "__main__":
    main()
