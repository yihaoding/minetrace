"""Build WA-schema label files from MINOCC (qld_all_sites.csv):
  labels/enriched/enriched_<El>_sites.csv  (33-col gswa/enriched schema, <El>_SITES = "<El> <SITE_TYPE_>")
  labels/no_sites/no_<el>_sites.csv         (Mine|Deposit for OTHER metals, target absent; lowercase schema)
Membership rule (analogue of MINEDEX_<el> membership): element symbol appears in SITE_COMMO
(MINOCC 'All Commodities') or TARGET_COM equals the element's main-commodity name.
"""
import csv, re
from pathlib import Path
import os
import pandas as pd
ROOT=Path(os.environ.get("PROJECT_ROOT", "/mmfs1/data/group/pmc050/yding/gad_reasoning") + "/datasets/queensland/labels")
(ROOT/"enriched").mkdir(exist_ok=True); (ROOT/"no_sites").mkdir(exist_ok=True)
d=pd.read_csv(ROOT/"qld_all_sites.csv",low_memory=False)
d["SITE_CODE"]=d["SITE_CODE"].astype(str).str.strip()
d["SITE_COMMO"]=d["SITE_COMMO"].fillna(""); d["TARGET_COM"]=d["TARGET_COM"].fillna("")
GSWA_SITES_COLS=["Ag","Au","Bi","Co","Cu","Dmd","Fe","Gr","Li","Mn","Nb","Ni","Pb","PGE","Ptsh","REE","Sn","Ta","V","W","Zn"]
MAIN_NAME={"Ag":"SILVER","Au":"GOLD","Bi":"BISMUTH","Co":"COBALT","Cu":"COPPER","Dmd":"DIAMOND","Fe":"IRON","Gr":"GRAPHITE","Li":"LITHIUM","Mn":"MANGANESE",
           "Nb":"NIOBIUM","Ni":"NICKEL","Pb":"LEAD","PGE":"PLATINUM","Ptsh":"POTASH","REE":"RARE EARTHS","Sn":"TIN","Ta":"TANTALUM","V":"VANADIUM","W":"TUNGSTEN","Zn":"ZINC","U":"URANIUM"}
ALIASES={"PGE":{"Pge","Pt","Pd","Pgm"},"REE":{"Ree","La","Ce","Nd","Y","Mnz"},"Gr":{"Gr","Gph"},"Dmd":{"Dmd","Dia"},"Ptsh":{"K","Ptsh"}}
def toks(s): return [t for t in re.split(r"[ ,;/]+",str(s)) if t]
commo_tokens=d["SITE_COMMO"].apply(lambda s:set(toks(s)))
def member(el):
    al=ALIASES.get(el,{el})
    return commo_tokens.apply(lambda ts: bool(ts&al)) | d["TARGET_COM"].str.upper().eq(MAIN_NAME.get(el,el.upper()))
mem={el:member(el) for el in GSWA_SITES_COLS+["U"]}
stype=d["SITE_TYPE_"].fillna("")
sites_val={el:[f"{el} {t}" if m else "" for m,t in zip(mem[el],stype)] for el in GSWA_SITES_COLS}
TIER={"Mine":"mine","Deposit":"deposit","Prospect":"prospect","Occurrence":"occurrence"}
base=pd.DataFrame({"SITE_CODE":d.SITE_CODE,"SITE_TITLE":d.SITE_TITLE,"SHORT_NAME":d.SHORT_NAME,"X":d.X,"Y":d.Y,"SITE_COMMO":d.SITE_COMMO,
                   "SITE_TYPE_":stype,"SITE_SUB_T":d["SITE_SUB_T"].fillna(""),"SITE_STAGE":d.SITE_STAGE.fillna("")})
for el in GSWA_SITES_COLS: base[f"{el}_SITES"]=sites_val[el]
base["tier"]=stype.map(TIER).fillna("other")
base["DEPOSIT_SIZE"]=d["DEPOSIT_SIZE"].fillna(""); base["SIZE_ORDER"]=d["SIZE_ORDER"]
summary=[]
for el in ["Cu","Au","W","Sn","Ni","Pb","Zn","Co","U"]:
    m=mem[el]; out=base[m].copy()
    main=d.loc[m,"TARGET_COM"].str.upper().eq(MAIN_NAME[el])
    out["is_primary"]=main.astype(int).values; out["target"]=el
    if el=="U": out["U_SITES"]=[f"U {t}" for t in stype[m]]
    rank={"mine":0,"deposit":1,"prospect":2,"occurrence":3,"other":4}
    out=out.assign(_r=out.tier.map(rank)).sort_values(["_r","SITE_CODE"]).drop(columns="_r")
    out.to_csv(ROOT/"enriched"/f"enriched_{el}_sites.csv",index=False)
    # negatives: Mine|Deposit for other metals, target absent
    neg=d[(~m)&stype.isin(["Mine","Deposit"])]
    pd.DataFrame({"site_code":neg.SITE_CODE,"site_title":neg.SITE_TITLE,"short_name":neg.SHORT_NAME,"site_commo":neg.SITE_COMMO,"site_type":neg.SITE_TYPE_,
                  "site_stage":neg.SITE_STAGE,"X":neg.X,"Y":neg.Y,"source_files":neg.TARGET_COM}).to_csv(ROOT/"no_sites"/f"no_{el.lower()}_sites.csv",index=False)
    md=out.tier.isin(["mine","deposit"])
    summary.append((el,len(out),int(md.sum()),int((md&(out.SIZE_ORDER>=2)).sum()),int(out.is_primary.sum()),len(neg)))
print(f"{'el':4s} {'rows':>6s} {'Mine|Dep':>9s} {'M|D size>=SMALL':>16s} {'primary':>8s} {'no_sites':>9s}")
for s in summary: print(f"{s[0]:4s} {s[1]:6d} {s[2]:9d} {s[3]:16d} {s[4]:8d} {s[5]:9d}")
print("\nenriched_Cu head:"); print(pd.read_csv(ROOT/"enriched"/"enriched_Cu_sites.csv").head(2).T.to_string())
