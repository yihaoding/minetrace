"""Queensland re-run of the WA `eval_comprehensive.py` protocol (geochem-only expert layer).

Same code path as WA (ProspectivityModel + 4 negative strategies, AUROC / P@k), with:
  * GSQ data converted to the gswa schema (scripts/qld/convert_qld_assays.py, build_qld_sites.py)
  * region loop: R0 = statewide (WA-identical protocol, incl. south-train/north-test spatial split)
                 R1..R5 = scenario regions; strategies 1-3 are in-region (WA random split),
                 strategy 4 'spatial' = REGION HOLDOUT (train on positives outside the region,
                 test on in-region positives, negatives = in-region non-mine sites)
  * bbox / centre latitude / spatial boundary set per region; raster & vector specs NOT registered
  * MAX_TRAIN_POS cap (WA had ≤1057 Cu positives; QLD Au has 8343 — cap keeps runtime comparable)
Env: SMOKE=1 for a quick test (fewer test/neg/bg, one region, one metal).
"""
from __future__ import annotations
import os, sys, json, time, logging, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ABLATE_LAYERS", "geochem")
_spec = importlib.util.spec_from_file_location("eval_comprehensive", ROOT / "scripts" / "eval_comprehensive.py")
ec = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ec)
from core.catalog import DataCatalog, SourceSpec
from domains.geochem.prospectivity_model import ProspectivityModel, TargetConfig
from domains.geochem.samples import GeochemSample

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
log = logging.getLogger("eval_qld")

SMOKE = os.environ.get("SMOKE") == "1"
QDATA   = ROOT / "datasets" / "queensland"
GEO_DIR = QDATA / "geochemical"
LAB_DIR = QDATA / "labels"
OUT_DIR = ROOT / "reports" / "qld"; OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = ec.SEED
MAX_TEST = 10 if SMOKE else ec.MAX_TEST          # 30
N_NEG    = 40 if SMOKE else ec.N_NEG             # 200
N_BG     = 100 if SMOKE else 300
MAX_TRAIN_POS = 300 if SMOKE else 1500
FAR_KM, BUFFER_KM, KM_PER_DEG = ec.FAR_KM, ec.BUFFER_KM, ec.KM_PER_DEG
QLD_BBOX = (138.0, 153.6, -29.2, -10.0)
LAYERS = frozenset({"geochem"})

# ── scenario regions (lon_min, lon_max, lat_min, lat_max) ─────────────────────
REGIONS = {
    "R0_QLD_statewide":      dict(bbox=QLD_BBOX, spatial_boundary=-20.5, scenario="statewide, WA-identical protocol (spatial = south train / north test)"),
    "R1_MtIsa_Cu":           dict(bbox=(138.5, 141.5, -22.6, -18.8), scenario="Proterozoic Mt Isa Inlier: IOCG / breccia / sediment-hosted Cu, dense exploration data"),
    "R2_CharterTowers_Au":   dict(bbox=(145.5, 147.8, -21.8, -19.3), scenario="Charters Towers–Ravenswood–Drummond: intrusion-related & epithermal Au, dense historic mining"),
    "R3_Georgetown_poly":    dict(bbox=(142.3, 145.0, -19.8, -17.3), scenario="Georgetown–Etheridge–Croydon inlier: polymetallic Au/Cu/W, moderate data density"),
    "R4_Herberton_WSn":      dict(bbox=(144.6, 146.0, -18.2, -16.3), scenario="Herberton–Hodgkinson: granite-related W/Sn (+Au), steep terrain"),
    "R5_SEQ_NewEngland_Au":  dict(bbox=(150.5, 153.5, -28.9, -24.5), scenario="Phanerozoic New England Orogen / Gympie: epithermal–VMS Au(-Cu), geologically unlike WA"),
}
METALS = ["Cu", "Au", "W", "Sn", "Ni"]
if SMOKE:
    REGIONS = {"R1_MtIsa_Cu": REGIONS["R1_MtIsa_Cu"]}; METALS = ["Cu"]

ASSAY_FILES = [("sediment","sediment","qld_all_sediment.csv","STREA"),("rockchips","rockchip","qld_all_rockchips.csv","ROCKC"),
               ("drillhole","drillhole","qld_all_drillhole_maxgrade.csv","DRILL"),("shallowdrill","shallowdrill","qld_all_shallowdrill.csv","SHALL"),
               ("soil","soil","qld_all_surfsoilgeochem.csv","SOIL")]

def in_bbox(df, bbox, xcol="X", ycol="Y"):
    x0,x1,y0,y1 = bbox
    return (df[xcol]>=x0)&(df[xcol]<=x1)&(df[ycol]>=y0)&(df[ycol]<=y1)

def region_assay_dir(rname, bbox, margin=1.0):
    """Crop statewide assay CSVs to bbox+margin (cached)."""
    if rname.startswith("R0"): return GEO_DIR
    d = GEO_DIR / "regions" / rname; d.mkdir(parents=True, exist_ok=True)
    x0,x1,y0,y1 = bbox; eb=(x0-margin,x1+margin,y0-margin,y1+margin)
    for _,_,f,_ in ASSAY_FILES:
        if (d/f).exists(): continue
        parts=[]
        for ch in pd.read_csv(GEO_DIR/f, chunksize=300_000, low_memory=False):
            parts.append(ch[in_bbox(ch,eb)])
        pd.concat(parts).to_csv(d/f, index=False)
    return d

def build_catalog(assay_dir, center_lat):
    cat = DataCatalog()
    for name, sub, f, st in ASSAY_FILES:
        cat.register(SourceSpec(name=name, source_type="assay_spatial", modality="geochemistry", subtype=sub,
                                path=str(assay_dir/f), loader_kwargs={"sample_type_filter": st, "center_lat": center_lat}))
    return cat

def make_agent(catalog, tc, bbox, seed):
    try:    return ProspectivityModel(catalog=catalog, target_config=tc, n_bg=N_BG, bbox=bbox, seed=seed)
    except TypeError: return ProspectivityModel(catalog=catalog, config=tc, n_bg=N_BG, bbox=bbox, seed=seed)

def plan_with_fallback(agent, bbox, tag):
    plan = agent.plan(check_bbox=bbox)
    print(f"    plan[{tag}]: " + "; ".join(f"{k}={v}" for k,v in plan.rationale.items()))
    if not any(a for a in plan.active_source_names):
        print(f"    plan[{tag}]: no assay source passed the coverage probe -> fallback trust_catalog")
        agent.plan(check_bbox=None)

def sample_ids(samples): return {s.id for s in samples}

def cap_train(all_pos, test_ids, rng):
    train = [p for p in all_pos if p.id not in test_ids]
    if len(train) > MAX_TRAIN_POS:
        keep = set(rng.choice(len(train), MAX_TRAIN_POS, replace=False).tolist())
        dropped = {p.id for i,p in enumerate(train) if i not in keep}
        return dropped
    return set()

def gen_random_negatives(agent, n, seed, bbox, forbid_xy_km=None, mine_tree=None, min_km=None, tag="RBG"):
    rng = np.random.default_rng(seed); x0,x1,y0,y1 = bbox
    ftree = cKDTree(forbid_xy_km) if forbid_xy_km is not None and len(forbid_xy_km) else None
    bg=[]; attempts=0; max_att = n*(2000 if mine_tree is not None else 500)
    while len(bg)<n and attempts<max_att:
        attempts+=1; x,y=float(rng.uniform(x0,x1)),float(rng.uniform(y0,y1))
        if mine_tree is not None and mine_tree.query([x*KM_PER_DEG,y*KM_PER_DEG],k=1)[0] < min_km: continue
        if ftree is not None and ftree.query([x*KM_PER_DEG,y*KM_PER_DEG],k=1)[0] < BUFFER_KM: continue
        for src_name in agent.active_sources():
            try:
                src = agent._registry.get(src_name)
                if hasattr(src,"has_coverage") and src.has_coverage(x,y):
                    bg.append(GeochemSample(site_code=f"{tag}_{len(bg):04d}", x=x, y=y)); break
            except KeyError: pass
    return bg

# ── labels ────────────────────────────────────────────────────────────────────
ENR = LAB_DIR/"enriched"; NOD = LAB_DIR/"no_sites"
all_sites = pd.read_csv(LAB_DIR/"qld_all_sites.csv", low_memory=False)
all_sites["SITE_CODE"]=all_sites["SITE_CODE"].astype(str).str.strip()
all_mine_ids = set(all_sites.loc[all_sites.SITE_TYPE_.isin(["Mine","Deposit"]),"SITE_CODE"])
mine_xy = all_sites.loc[all_sites.SITE_TYPE_.isin(["Mine","Deposit"]),["X","Y"]].values*KM_PER_DEG
mine_tree = cKDTree(mine_xy)
all_sites_xy = all_sites[["X","Y","SITE_CODE"]].dropna().drop_duplicates(subset=["X","Y"]).reset_index(drop=True)

def metal_cfg(m):
    c = dict(ec.METAL_CONFIGS[m]); c["sites_csv"]=str(ENR/f"enriched_{m}_sites.csv"); c["site_col"]=f"{m}_SITES"
    c["no_target_csv"]=str(NOD/f"no_{m.lower()}_sites.csv"); return c

def write_region_labels(rname, m, bbox):
    """Region-filtered enriched + no_ CSVs so the in-region agent's KB is in-region only."""
    d = LAB_DIR/"regions"/rname; d.mkdir(parents=True, exist_ok=True)
    e = pd.read_csv(ENR/f"enriched_{m}_sites.csv", low_memory=False); e = e[in_bbox(e,bbox)]
    e.to_csv(d/f"enriched_{m}_sites.csv", index=False)
    n = pd.read_csv(NOD/f"no_{m.lower()}_sites.csv", low_memory=False); n = n[in_bbox(n,bbox)]
    n.to_csv(d/f"no_{m.lower()}_sites.csv", index=False)
    return str(d/f"enriched_{m}_sites.csv"), str(d/f"no_{m.lower()}_sites.csv")

# ── main loop ─────────────────────────────────────────────────────────────────
summary = {}; scores = {}
t_start = time.time()
for rname, R in REGIONS.items():
    bbox = R["bbox"]; center_lat = (bbox[2]+bbox[3])/2.0
    statewide = rname.startswith("R0")
    print(f"\n{'#'*78}\n# {rname}  bbox={bbox}  centre_lat={center_lat:.1f}\n# {R['scenario']}\n{'#'*78}")
    assay_dir = region_assay_dir(rname, bbox)
    catalog_r = build_catalog(assay_dir, center_lat)
    catalog_sw = build_catalog(GEO_DIR, (QLD_BBOX[2]+QLD_BBOX[3])/2.0)
    summary[rname]={"bbox":bbox,"scenario":R["scenario"],"metals":{}}; scores[rname]={}
    for m in METALS:
        cfg = metal_cfg(m); t0=time.time()
        print(f"\n{'='*72}\n  {rname} / {m}\n{'='*72}")
        enr_all = ec.normalise_sites_df(pd.read_csv(cfg["sites_csv"], low_memory=False))
        md_all = enr_all[enr_all[cfg["site_col"]].fillna("").str.contains("Mine|Deposit")]
        md_in = md_all[in_bbox(md_all,bbox)]
        if len(md_in) < 20:
            print(f"  skipped: only {len(md_in)} Mine|Deposit positives in region"); summary[rname]["metals"][m]=None; continue
        if statewide:
            sites_csv, no_csv = cfg["sites_csv"], cfg["no_target_csv"]
        else:
            sites_csv, no_csv = write_region_labels(rname, m, bbox)
        all_pos = ec.load_mine_deposit_samples(sites_csv, cfg["site_col"], cfg["confidence_map"])
        all_pos_ids = sample_ids(all_pos)
        rng = np.random.default_rng(SEED); perm = rng.permutation(len(all_pos))
        n_test = min(MAX_TEST, max(0, len(all_pos)-10))
        test_pos = [all_pos[i] for i in perm[:n_test]]; test_ids = sample_ids(test_pos)
        forbid = ec._forbid_xy_km(test_pos)
        dropped = cap_train(all_pos, test_ids, np.random.default_rng(SEED+7))
        print(f"  positives in region: {len(all_pos)}  test={len(test_pos)}  train={len(all_pos)-len(test_pos)-len(dropped)} (capped from {len(all_pos)-len(test_pos)})")
        tc = TargetConfig(target=m, pathfinders=cfg["pathfinders"], ratio_features=cfg["ratio_features"],
                          confidence_map=cfg["confidence_map"], sites_csv=sites_csv, layers=LAYERS)
        agent = make_agent(catalog_r, tc, bbox, SEED)
        plan_with_fallback(agent, bbox, "in-region")
        agent.setup(exclude_ids=test_ids|dropped)
        print(f"  Active experts: {len(agent.active_experts())}  sources: {agent.active_sources()}  ({time.time()-t0:.0f}s)")
        # 1 random
        neg = gen_random_negatives(agent, N_NEG, SEED+1, bbox, forbid)
        r_random = ec.run_eval(agent, test_pos, neg)
        # 2 far random (>=50 km from ANY Mine|Deposit of any commodity, statewide tree)
        neg = gen_random_negatives(agent, N_NEG, SEED+2, bbox, forbid, mine_tree, FAR_KM, tag="FAR")
        r_far = ec.run_eval(agent, test_pos, neg); print(f"  far_random negatives: {len(neg)}")
        # 3 true non-mine
        nt = ec.load_no_target_sites(no_csv, all_pos_ids, forbid_xy_km=forbid)
        rng3 = np.random.default_rng(SEED+3)
        if len(nt) > N_NEG: nt = [nt[i] for i in rng3.choice(len(nt), N_NEG, replace=False)]
        r_nt = ec.run_eval(agent, test_pos, nt); print(f"  true_nonmine negatives: {len(nt)}")
        # 4 spatial
        r_sp = None; sp_info = ""
        if statewide:
            SB = R["spatial_boundary"]
            north = [GeochemSample(site_code=str(r.SITE_CODE), x=float(r.X), y=float(r.Y)) for _,r in md_all[md_all.Y>=SB].iterrows()]
            south = [GeochemSample(site_code=str(r.SITE_CODE), x=float(r.X), y=float(r.Y)) for _,r in md_all[md_all.Y< SB].iterrows()]
            rng4 = np.random.default_rng(SEED+4); n_nt = min(MAX_TEST, len(north))
            test_n = [north[i] for i in rng4.choice(len(north), n_nt, replace=False)] if len(north)>n_nt else list(north)
            if len(south)>=10 and len(test_n)>=5:
                drop_s = cap_train(south, set(), np.random.default_rng(SEED+8))
                ag = make_agent(catalog_sw, tc, bbox, SEED+10); plan_with_fallback(ag, bbox, "south")
                ag.setup(exclude_ids=sample_ids(north)|drop_s)
                sp_neg = ec.gen_region_negatives(all_sites_xy.Y>=SB, all_sites_xy, all_mine_ids, N_NEG, seed=SEED+5, forbid_xy_km=ec._forbid_xy_km(test_n))
                r_sp = ec.run_eval(ag, test_n, sp_neg); sp_info=f"south train {len(south)-len(drop_s)} / north test {len(test_n)} / north neg {len(sp_neg)}"
        else:
            # region holdout: train on positives OUTSIDE region (statewide files), test on the same in-region test set
            outside = md_all[~in_bbox(md_all,bbox)]
            out_pos = [GeochemSample(site_code=str(r.SITE_CODE), x=float(r.X), y=float(r.Y)) for _,r in outside.iterrows()]
            if len(out_pos)>=10:
                tc_sw = TargetConfig(target=m, pathfinders=cfg["pathfinders"], ratio_features=cfg["ratio_features"],
                                     confidence_map=cfg["confidence_map"], sites_csv=cfg["sites_csv"], layers=LAYERS)
                drop_o = cap_train(out_pos, set(), np.random.default_rng(SEED+9))
                ag = make_agent(catalog_sw, tc_sw, QLD_BBOX, SEED+10); ag.plan(check_bbox=None)
                ag.setup(exclude_ids=set(md_in.SITE_CODE.astype(str))|drop_o)   # exclude ALL in-region positives
                sp_neg = ec.gen_region_negatives(in_bbox(all_sites_xy,bbox), all_sites_xy, all_mine_ids, N_NEG, seed=SEED+5, forbid_xy_km=forbid)
                r_sp = ec.run_eval(ag, test_pos, sp_neg); sp_info=f"outside train {len(out_pos)-len(drop_o)} / in-region test {len(test_pos)} / in-region non-mine neg {len(sp_neg)}"
        print(f"  spatial: {sp_info or 'skipped'}")
        print(); ec.print_result("1. random", r_random); ec.print_result("2. far_random", r_far); ec.print_result("3. true_nonmine", r_nt); ec.print_result("4. spatial/holdout", r_sp)
        print(f"  [{m} done in {time.time()-t0:.0f}s]")
        res = {"random":r_random,"far_random":r_far,"true_nonmine":r_nt,"spatial":r_sp}
        summary[rname]["metals"][m] = {"n_pos_region":len(all_pos),"n_train":len(all_pos)-len(test_pos)-len(dropped),"spatial_info":sp_info,
            **{s:(None if r is None else {"auc":r["auc"],"n_pos":r["n_pos"],"n_neg":r["n_neg"],"prec":{str(k):v for k,v in r["prec"].items()}}) for s,r in res.items()}}
        scores[rname][m] = {s:(None if r is None else {"y_pred":r["y_pred"],"y_true":r["y_true"],"auc":r["auc"]}) for s,r in res.items()}
        tag = "_smoke" if SMOKE else ""
        json.dump({"regions":summary,"config":{"MAX_TEST":MAX_TEST,"N_NEG":N_NEG,"N_BG":N_BG,"MAX_TRAIN_POS":MAX_TRAIN_POS,"BUFFER_KM":BUFFER_KM,"FAR_KM":FAR_KM,"SEED":SEED,"layers":"geochem"}},
                  open(OUT_DIR/f"eval_qld_summary{tag}.json","w"), indent=2)
        json.dump(scores, open(OUT_DIR/f"eval_qld_scores{tag}.json","w"))

# ── summary table ─────────────────────────────────────────────────────────────
strats=["random","far_random","true_nonmine","spatial"]
print(f"\n{'='*78}\nSUMMARY — AUC by region × metal × negative strategy (geochem-only)\n{'='*78}")
print(f"  {'region':<24s}{'metal':<6s}{'n_pos':>6s}  "+"  ".join(f"{s:>12s}" for s in ["Random","Far(>50km)","TrueNonMine","Spatial/HO"]))
for rname,R in summary.items():
    for m,d in R["metals"].items():
        if d is None: print(f"  {rname:<24s}{m:<6s}{'—':>6s}  (skipped)"); continue
        print(f"  {rname:<24s}{m:<6s}{d['n_pos_region']:>6d}  "+"  ".join(f"{d[s]['auc']:>12.3f}" if d[s] else f"{'—':>12s}" for s in strats))
print(f"\ntotal {time.time()-t_start:.0f}s")
