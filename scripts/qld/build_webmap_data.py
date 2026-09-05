"""Build data for the self-contained Queensland prospectivity web map.
Per metal: fit geochem-only agent on ALL Mine|Deposit positives, score a 0.05° grid (heatmap PNG),
and precompute point explanations (narrative.describe) for probe points (major sites + 0.2° grid).
Outputs reports/qld/webmap/<metal>.json (+ png) and shared basemap.json.
"""
from __future__ import annotations
import os, sys, json, time, base64, io, math, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
os.environ.setdefault("ABLATE_LAYERS", "geochem")
_spec = importlib.util.spec_from_file_location("eval_qld", ROOT/"scripts"/"qld"/"eval_comprehensive_qld.py")
# we only need helpers from the eval module namespace without running its loop -> re-implement minimal pieces
from core.catalog import DataCatalog, SourceSpec
from domains.geochem.prospectivity_model import ProspectivityModel, TargetConfig
from domains.geochem.samples import GeochemSample
from domains.geochem.narrative import describe
import importlib; ec_spec = importlib.util.spec_from_file_location("eval_comprehensive", ROOT/"scripts"/"eval_comprehensive.py")
ec = importlib.util.module_from_spec(ec_spec); ec_spec.loader.exec_module(ec)

METALS = sys.argv[1:] or ["Cu","Au"]
QDATA = ROOT/"datasets"/"queensland"; GEO_DIR = QDATA/"geochemical"; LAB = QDATA/"labels"
OUT = ROOT/"reports"/"qld"/"webmap"; OUT.mkdir(parents=True, exist_ok=True)
QLD_BBOX = (138.0, 153.6, -29.2, -10.0); CENTER_LAT = -19.6
GRID_DEG = 0.05; PROBE_DEG = 0.2; NPROC = int(os.environ.get("NPROC", "8"))
RAMP = ["#fde6cf","#fbb27a","#ef7a3a","#c4431c","#6e1a0a"]
ASSAY = [("sediment","sediment","qld_all_sediment.csv","STREA"),("rockchips","rockchip","qld_all_rockchips.csv","ROCKC"),
         ("drillhole","drillhole","qld_all_drillhole_maxgrade.csv","DRILL"),("shallowdrill","shallowdrill","qld_all_shallowdrill.csv","SHALL"),
         ("soil","soil","qld_all_surfsoilgeochem.csv","SOIL")]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ── shared: basemap + regions ─────────────────────────────────────────────────
if not (OUT/"basemap.json").exists():
    import geopandas as gpd
    ne = gpd.read_file(QDATA/"basemap"/"ne_10m_admin_1_states_provinces.shp")
    au = ne[ne["admin"]=="Australia"][["name","geometry"]].copy()
    au["geometry"] = au.geometry.simplify(0.03, preserve_topology=True)
    feats=[]
    for _,r in au.iterrows():
        polys = [r.geometry] if r.geometry.geom_type=="Polygon" else list(r.geometry.geoms)
        rings=[]
        for p in polys:
            if p.area < 0.01: continue
            rings.append([[round(x,3),round(y,3)] for x,y in p.exterior.coords])
        feats.append({"name":r["name"],"rings":rings})
    summ = json.load(open(ROOT/"reports"/"qld"/"eval_qld_summary.json"))
    regions=[]
    for rn,R in summ["regions"].items():
        if rn.startswith("R0"): continue
        regions.append({"id":rn.split("_")[0],"name":rn,"bbox":R["bbox"],"scenario":R["scenario"],
                        "metals":{m:(None if d is None else {s:(d[s]["auc"] if d[s] else None) for s in ["random","far_random","true_nonmine","spatial"]}) for m,d in R["metals"].items()}})
    json.dump({"states":feats,"regions":regions,"bbox":QLD_BBOX}, open(OUT/"basemap.json","w"))
    log("basemap written:", len(feats), "states,", len(regions), "regions")

# ── assay coordinate trees for 'nearby samples' counts ────────────────────────
KM_LON = 111.0*math.cos(math.radians(CENTER_LAT)); KM_LAT = 111.0
trees = {}
for name,_,f,_ in ASSAY:
    xy = pd.read_csv(GEO_DIR/f, usecols=["X","Y"]).values
    trees[name] = cKDTree(np.column_stack([xy[:,0]*KM_LON, xy[:,1]*KM_LAT]))
log("assay trees built")
def nearby_counts(x,y,r=10.0):
    p=[x*KM_LON,y*KM_LAT]; return [int(trees[n].query_ball_point(p,r,return_length=True)) for n,_,_,_ in ASSAY]

catalog = DataCatalog()
for name, sub, f, st in ASSAY:
    catalog.register(SourceSpec(name=name, source_type="assay_spatial", modality="geochemistry", subtype=sub,
                                path=str(GEO_DIR/f), loader_kwargs={"sample_type_filter": st, "center_lat": CENTER_LAT}))

def hex2rgb(h): return tuple(int(h[i:i+2],16) for i in (1,3,5))
STOPS=[hex2rgb(h) for h in RAMP]
def colour(v):
    t=max(0,min(1,v))*(len(STOPS)-1); i=min(int(t),len(STOPS)-2); f=t-i
    return tuple(int(round(STOPS[i][k]*(1-f)+STOPS[i+1][k]*f)) for k in range(3))

AGENT=None
def _score_chunk(args):
    idxs, xs, ys = args
    samples=[GeochemSample(site_code=f"G{i}", x=float(x), y=float(y)) for i,x,y in zip(idxs,xs,ys)]
    out=[]
    for i,ns in zip(idxs, AGENT.score_batch(samples)):
        out.append((int(i), None if ns is None else float(ns.score)))
    return out
def _explain_chunk(args):
    idxs, xs, ys, metal = args
    res=[]
    for i,x,y in zip(idxs,xs,ys):
        s=GeochemSample(site_code=f"P{i}", x=float(x), y=float(y))
        ns=AGENT.score_batch([s])[0]
        if ns is None: res.append((int(i),None)); continue
        pn=describe(ns, metal, float(x), float(y), top_n=6)
        experts=[[e.name, round(e.score,3), round(e.tree_weight,3)] for e in sorted(pn.expert_contribs, key=lambda e:-e.effective_contrib)]
        sig=[[fs.human_label(), fs.source, round(fs.z_score,2), round(fs.contribution,3)] for fs in pn.top_signals[:6]]
        res.append((int(i), {"s":round(pn.g_score,4),"tier":pn.tier,"why":pn.tier_reason,"ex":experts,"sig":sig}))
    return res

for metal in METALS:
    cfg=dict(ec.METAL_CONFIGS[metal]); sites_csv=str(LAB/"enriched"/f"enriched_{metal}_sites.csv")
    tc=TargetConfig(target=metal, pathfinders=cfg["pathfinders"], ratio_features=cfg["ratio_features"],
                    confidence_map=cfg["confidence_map"], sites_csv=sites_csv, layers=frozenset({"geochem"}))
    AGENT=ProspectivityModel(catalog=catalog, config=tc, n_bg=400, bbox=QLD_BBOX, seed=42)
    AGENT.plan(check_bbox=None); AGENT.setup(exclude_ids=None)
    log(metal, "agent ready; experts:", AGENT.active_experts())
    # coverage mask for grid
    srcs=[AGENT._registry.get(n) for n in AGENT.active_sources()]
    xs=np.arange(QLD_BBOX[0], QLD_BBOX[1]+1e-9, GRID_DEG); ys=np.arange(QLD_BBOX[2], QLD_BBOX[3]+1e-9, GRID_DEG)
    XX,YY=np.meshgrid(xs,ys); fx,fy=XX.ravel(),YY.ravel()
    cov=np.array([any(s.has_coverage(float(x),float(y)) for s in srcs) for x,y in zip(fx,fy)])
    idx=np.where(cov)[0]; log(metal, f"grid {len(xs)}x{len(ys)}={len(fx):,} cells, covered {len(idx):,}")
    import multiprocessing as mp
    ctx=mp.get_context("fork"); chunks=[(idx[i:i+500], fx[idx[i:i+500]], fy[idx[i:i+500]]) for i in range(0,len(idx),500)]
    scores=np.full(len(fx), np.nan)
    with ctx.Pool(NPROC) as pool:
        for k,out in enumerate(pool.imap_unordered(_score_chunk, chunks)):
            for i,s in out:
                if s is not None: scores[i]=s
            if k%20==0: log(metal, f"scored chunk {k+1}/{len(chunks)}")
    grid=scores.reshape(len(ys),len(xs)); np.savez_compressed(OUT/f"grid_{metal}.npz", grid=grid, xs=xs, ys=ys)
    valid=grid[np.isfinite(grid)]; lo,hi=np.nanpercentile(valid,[2,98]); log(metal, f"grid scored: {np.isfinite(grid).sum():,} valid, p2={lo:.3f} p98={hi:.3f}")
    # PNG (row 0 = north)
    from PIL import Image
    H,W=grid.shape; img=np.zeros((H,W,4),dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            v=grid[H-1-r,c]
            if np.isfinite(v):
                t=(v-lo)/max(1e-9,hi-lo); rgb=colour(t); img[r,c,:3]=rgb; img[r,c,3]=int(120+100*max(0,min(1,t)))
    buf=io.BytesIO(); Image.fromarray(img,"RGBA").save(buf,"PNG",optimize=True); png=base64.b64encode(buf.getvalue()).decode()
    # sites
    e=pd.read_csv(sites_csv, low_memory=False); e["SITE_CODE"]=e["SITE_CODE"].astype(str)
    md=e[e[f"{metal}_SITES"].fillna("").str.contains("Mine|Deposit")].copy()
    sites=[[round(float(r.X),4),round(float(r.Y),4),str(r.SITE_TITLE)[:40],int(r.SIZE_ORDER) if pd.notna(r.SIZE_ORDER) else 0,str(r.SITE_STAGE),str(r.SITE_TYPE_)] for _,r in md.iterrows()]
    site_tree=cKDTree(np.column_stack([md.X.values*KM_LON, md.Y.values*KM_LAT]))
    # probes: major sites + coarse grid (covered only)
    major=md[md.SIZE_ORDER>=2]
    px=np.arange(QLD_BBOX[0], QLD_BBOX[1]+1e-9, PROBE_DEG); py=np.arange(QLD_BBOX[2], QLD_BBOX[3]+1e-9, PROBE_DEG)
    PX,PY=np.meshgrid(px,py); pfx,pfy=PX.ravel(),PY.ravel()
    pcov=np.array([any(s.has_coverage(float(x),float(y)) for s in srcs) for x,y in zip(pfx,pfy)])
    probe_xy=np.vstack([np.column_stack([major.X.values,major.Y.values]), np.column_stack([pfx[pcov],pfy[pcov]])])
    kind=["site"]*len(major)+["grid"]*int(pcov.sum())
    log(metal, f"probes: {len(major)} major sites + {int(pcov.sum())} grid = {len(probe_xy)}")
    pidx=np.arange(len(probe_xy)); pchunks=[(pidx[i:i+100], probe_xy[i:i+100,0], probe_xy[i:i+100,1], metal) for i in range(0,len(pidx),100)]
    expl=[None]*len(probe_xy)
    with ctx.Pool(NPROC) as pool:
        for k,out in enumerate(pool.imap_unordered(_explain_chunk, pchunks)):
            for i,d in out: expl[i]=d
            if k%10==0: log(metal, f"explained chunk {k+1}/{len(pchunks)}")
    probes=[]
    for i,(x,y) in enumerate(probe_xy):
        d=expl[i]
        if d is None: continue
        dist,j=site_tree.query([x*KM_LON,y*KM_LAT],k=1)
        probes.append({"x":round(float(x),4),"y":round(float(y),4),"k":kind[i],"s":d["s"],"tier":d["tier"],"why":d["why"],"ex":d["ex"],"sig":d["sig"],
                       "near":[int(j),round(float(dist),1)],"nb":nearby_counts(float(x),float(y))})
    meta={"metal":metal,"n_pos":int(len(md)),"experts":AGENT.active_experts(),"sources":AGENT.active_sources(),
          "grid":{"x0":float(xs[0]),"x1":float(xs[-1]),"y0":float(ys[0]),"y1":float(ys[-1]),"nx":int(W),"ny":int(H),"lo":float(lo),"hi":float(hi)},"ramp":RAMP,"nb_sources":[a[0] for a in ASSAY]}
    json.dump({"meta":meta,"png":png,"sites":sites,"probes":probes}, open(OUT/f"{metal}.json","w"))
    log(metal, f"written {OUT/f'{metal}.json'} ({(OUT/f'{metal}.json').stat().st_size/1e6:.1f} MB), probes {len(probes)}")
log("done")
