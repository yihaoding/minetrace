"""Build per-element enriched positive-sample CSVs in gswa-compatible schema.

For each of the 15 MINEDEX target elements, write a single CSV whose rows are
every site flagged for that element (GSWA's <el>_SITES flag === MINEDEX_<el>
membership). The schema matches `gswa_<el>_sites.csv` so the existing
`ProspectivityModel._load_kb` and `SiteCatalogSource` work as drop-in:

    SITE_CODE, SITE_TITLE, SHORT_NAME, X, Y,
    SITE_COMMO, SITE_TYPE_, SITE_SUB_T, SITE_STAGE,
    Ag_SITES, Au_SITES, Bi_SITES, Co_SITES, Cu_SITES, Dmd_SITES, Fe_SITES,
    Gr_SITES, Li_SITES, Mn_SITES, Nb_SITES, Ni_SITES, Pb_SITES, PGE_SITES,
    Ptsh_SITES, REE_SITES, Sn_SITES, Ta_SITES, V_SITES, W_SITES, Zn_SITES,
    tier, is_primary, target

The KEY enrichment: <X>_SITES for a site reflects the SITE-LEVEL site_type
(`<el> Mine` / `<el> Deposit` / `<el> Prospect` / `<el> Occurrence`), rather
than GSWA's per-element fine-grained classification. This catches polymetallic
sites where the target metal is a byproduct of a higher-tier metal (e.g. Co
at a Ni Mine), boosting M+D positives substantially:
    Cu  752 -> 1057,  Au 156 -> 287,  Ni 547 -> 581,  W 86 -> 99

The other 11 elements (Sn/Co/Ta/Mn/REE/Nb/Li/V/PGE/Ptsh/Gr) gain a baseline.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path("/group/pmc050/yding/gad_reasoning/datasets/geochemical/states")
SRC  = ROOT / "sites"
DST  = ROOT / "sites_enriched"
DST.mkdir(exist_ok=True)

ELEMS = ["Au","Co","Cu","Gr","Li","Mn","Nb","Ni","PGE","Ptsh","REE","Sn","Ta","V","W"]

# Full gswa-style *_SITES column list (21 cols). 6 of them — Ag, Bi, Dmd, Fe,
# Pb, Zn — have no MINEDEX file in this dataset and are emitted as empty.
GSWA_SITES_COLS = ["Ag","Au","Bi","Co","Cu","Dmd","Fe","Gr","Li","Mn",
                   "Nb","Ni","Pb","PGE","Ptsh","REE","Sn","Ta","V","W","Zn"]

# Tokens accepted as a match for target element in site_commo
# (used only for the informational `is_primary` flag)
ALIASES: dict[str, set[str]] = {
    "Au":   {"Au"},
    "Co":   {"Co"},
    "Cu":   {"Cu"},
    "Gr":   {"Gr", "Graphite", "TGC"},
    "Li":   {"Li", "Li2O", "Lpd", "Mica", "Lithium"},
    "Mn":   {"Mn"},
    "Nb":   {"Nb"},
    "Ni":   {"Ni"},
    "PGE":  {"PGE", "Pt", "Pd", "PGM", "Os", "Ir", "Rh", "Ru"},
    "Ptsh": {"Ptsh", "K", "K2SO4", "K2O", "SOP", "MOP", "Potash", "Glt", "Salt"},
    "REE":  {"REE", "HREE", "LREE",
             "La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Y",
             "Mnz","Xen","Bsn","HM"},
    "Sn":   {"Sn", "SnO2"},
    "Ta":   {"Ta", "Ta2O5"},
    "V":    {"V"},
    "W":    {"W", "WO3"},
}

TOKEN_SPLIT = re.compile(r"[,/]\s*|\s+")


def tokenize_commo(s: str) -> list[str]:
    s = (s or "").strip().strip('"').strip("'")
    return [t for t in TOKEN_SPLIT.split(s) if t]


def site_type_tier(stype: str) -> str:
    s = (stype or "").strip()
    if s == "Mine":       return "mine"
    if s == "Deposit":    return "deposit"
    if s == "Prospect":   return "prospect"
    if s == "Occurrence": return "occurrence"
    return "other"


def pool_all_sites() -> tuple[dict[str, dict], dict[str, set[str]]]:
    """Return (site_code -> canonical row, element -> set of site_codes)."""
    pool: dict[str, dict] = {}
    in_file: dict[str, set[str]] = {el: set() for el in ELEMS}
    for el in ELEMS:
        with (SRC / f"MINEDEX_{el}_XY_GDA2020.csv").open() as fh:
            for r in csv.DictReader(fh):
                sc = r["site_code"].strip()
                if not sc:
                    continue
                in_file[el].add(sc)
                if sc not in pool or len(r.get("site_commo","")) > len(pool[sc].get("site_commo","")):
                    pool[sc] = dict(r)
    return pool, in_file


def build_per_element(pool: dict[str, dict],
                      in_file: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    """Write enriched_<el>_sites.csv in gswa-compat schema. Return tier counts."""
    base_cols = ["SITE_CODE","SITE_TITLE","SHORT_NAME","X","Y",
                 "SITE_COMMO","SITE_TYPE_","SITE_SUB_T","SITE_STAGE"]
    sites_cols = [f"{el}_SITES" for el in GSWA_SITES_COLS]
    extra_cols = ["tier","is_primary","target"]
    out_cols = base_cols + sites_cols + extra_cols
    summary: dict[str, dict[str, int]] = {}

    for el in ELEMS:
        anchor_ids = in_file[el]
        counts = {"mine":0, "deposit":0, "prospect":0, "occurrence":0, "other":0}
        rows_out: list[dict] = []
        aliases = ALIASES[el]

        for sc in anchor_ids:
            r = pool[sc]
            stype = (r.get("site_type","") or "").strip()
            tier  = site_type_tier(stype)
            counts[tier] += 1
            toks = tokenize_commo(r.get("site_commo",""))
            is_primary = bool(toks) and (toks[0] in aliases)

            row_out = {c: "" for c in out_cols}
            row_out["SITE_CODE"]  = sc
            row_out["SITE_TITLE"] = r.get("site_title","")
            row_out["SHORT_NAME"] = r.get("short_name","")
            row_out["X"]          = r.get("X","")
            row_out["Y"]          = r.get("Y","")
            row_out["SITE_COMMO"] = r.get("site_commo","")
            row_out["SITE_TYPE_"] = stype
            row_out["SITE_SUB_T"] = ""              # not available in MINEDEX simple schema
            row_out["SITE_STAGE"] = r.get("site_stage","")
            # Populate <el>_SITES for every element this site belongs to
            for el2 in ELEMS:
                if sc in in_file[el2]:
                    row_out[f"{el2}_SITES"] = f"{el2} {stype}" if stype else el2
            row_out["tier"]       = tier
            row_out["is_primary"] = "1" if is_primary else "0"
            row_out["target"]     = el
            rows_out.append(row_out)

        rank = {"mine":0,"deposit":1,"prospect":2,"occurrence":3,"other":4}
        rows_out.sort(key=lambda x: (rank[x["tier"]], x["SITE_CODE"]))
        with (DST / f"enriched_{el}_sites.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=out_cols)
            w.writeheader()
            w.writerows(rows_out)
        summary[el] = counts
    return summary


def _current_pipeline_counts() -> dict[str, tuple[int, int]]:
    """(rows, unique_sites) of Mine|Deposit positives in the CURRENT pipeline
    using gswa_<el>_sites.csv + <El>_SITES filter. For comparison."""
    cur: dict[str, tuple[int, int]] = {el: (0, 0) for el in ELEMS}
    for el, gf, col in [("Cu","gswa_cu_sites.csv","Cu_SITES"),
                        ("Au","gswa_au_site.csv","Au_SITES"),
                        ("Ni","gswa_ni_site.csv","Ni_SITES"),
                        ("W", "gswa_w_site.csv", "W_SITES")]:
        rows = 0
        uniq: set[str] = set()
        with (SRC / gf).open() as fh:
            for r in csv.DictReader(fh):
                v = r.get(col,"") or ""
                if "Mine" in v or "Deposit" in v:
                    rows += 1
                    uniq.add(r["SITE_CODE"].strip())
        cur[el] = (rows, len(uniq))
    return cur


def write_summary(summary: dict[str, dict[str, int]],
                  current: dict[str, tuple[int, int]]) -> None:
    with (DST / "enriched_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["element","Mine","Deposit","Prospect","Occurrence","other",
                    "enriched_M+D_unique","total_in_MINEDEX",
                    "current_M+D_rows","current_M+D_unique","delta_vs_rows"])
        for el in ELEMS:
            c = summary[el]
            md_new = c["mine"] + c["deposit"]
            total  = sum(c.values())
            md_rows, md_uniq = current[el]
            if md_rows > 0:
                delta = f"{md_new - md_rows:+d}"
                rows_disp = str(md_rows); uniq_disp = str(md_uniq)
            else:
                delta = "new"; rows_disp = "—"; uniq_disp = "—"
            w.writerow([el, c["mine"], c["deposit"], c["prospect"], c["occurrence"],
                        c["other"], md_new, total, rows_disp, uniq_disp, delta])


def main() -> None:
    pool, in_file = pool_all_sites()
    print(f"Pooled unique sites: {len(pool)}")
    summary  = build_per_element(pool, in_file)
    current  = _current_pipeline_counts()
    write_summary(summary, current)

    print()
    hdr = (f"{'Element':<6}  {'Mine':>5}  {'Depo':>5}  {'Prosp':>5}  {'Occ':>5}  "
           f"{'M+D new':>7}  {'old rows':>8}  {'old uniq':>8}  {'Δ rows':>6}")
    print(hdr); print("-" * len(hdr))
    for el in ELEMS:
        c = summary[el]
        md_new = c["mine"] + c["deposit"]
        md_rows, md_uniq = current[el]
        if md_rows > 0:
            delta = f"{md_new-md_rows:+d}"
            rows_disp = str(md_rows); uniq_disp = str(md_uniq)
        else:
            delta = "new"; rows_disp = "—"; uniq_disp = "—"
        print(f"{el:<6}  {c['mine']:>5}  {c['deposit']:>5}  {c['prospect']:>5}  "
              f"{c['occurrence']:>5}  {md_new:>7}  {rows_disp:>8}  {uniq_disp:>8}  {delta:>6}")
    print()
    print(f"Output dir: {DST}")
    print(f"  - enriched_<el>_sites.csv  (15 files, gswa-compat schema)")
    print(f"  - enriched_summary.csv     (counts table)")


if __name__ == "__main__":
    main()
