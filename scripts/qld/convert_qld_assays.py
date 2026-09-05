import os
"""Convert GSQ Explorer CSV export -> GSWA-style assay CSVs (X,Y,SAMPLEID,SAMPLETYPE,WAMEX_A_NO,COMPSAMPID,<El>_ppm...).
Conventions copied from gswa_all_*.csv: missing = -9999, below-detection = -DL (negative), units ppm.
Units resolved per row via s_*.Job_No -> lib_Job.Job_no -> Alias_Job -> lib_Job_An(Column_Name).Units; unresolved -> nominal ppm.
"""
import zipfile, csv, io, sys, time, collections
Z=os.environ.get("PROJECT_ROOT", "/mmfs1/data/group/pmc050/yding/gad_reasoning") + "/datasets/queensland/geochemistry/whole-of-queensland-2024-csv.zip"
OUT=os.environ.get("PROJECT_ROOT", "/mmfs1/data/group/pmc050/yding/gad_reasoning") + "/datasets/queensland/geochemical"
ELS=['Au','Cu','Pb','Zn','Ag','As','Bi','Mo','Mn','Fe','Ni','Co','Cr','V','Ba','Cd','Sn','Sb','Hg','Te','P','W','Zr','Ti','Mg','Th','U','Pt','Pd','S','F']
MISSING="-9999"
zf=zipfile.ZipFile(Z); csv.field_size_limit(10**9)
def reader(n): return csv.reader(io.TextIOWrapper(zf.open(n),encoding='latin-1',newline=''))
t0=time.time()
# --- unit maps
r=reader("lib_Job.csv"); h=next(r); ix={c:i for i,c in enumerate(h)}
job2alias={row[ix['Job_no']].strip():row[ix['Alias_Job']].strip() for row in r}
r=reader("lib_Job_An.csv"); h=next(r); ix={c:i for i,c in enumerate(h)}
alias_units={}
for row in r:
    alias_units[(row[ix['Alias_Job']].strip(),row[ix['Column_Name']].strip())]=row[ix['Units']].strip().lower()
FACTOR={'ppm':1.0,'ppb':1e-3,'%':1e4,'pct':1e4,'g/t':1.0,'ppt':1e-6,'oz/t':34.2857}
stats=collections.Counter()
def conv(raw,job,el):
    s=raw.strip().replace('$','').replace(',','')
    if s=='' : return MISSING
    try: v=float(s)
    except: return MISSING
    u=alias_units.get((job2alias.get(job,''),el))
    f=FACTOR.get(u)
    if f is None:
        stats[f'{el}:unresolved' if u is None else f'{el}:unit?{u}']+=1; f=1.0
    else: stats[f'{el}:{u}']+=1
    return f"{v*f:.6g}"
HEAD=["X","Y","SAMPLEID","SAMPLETYPE","WAMEX_A_NO","COMPSAMPID"]+[f"{e}_ppm" for e in ELS]
def surface(table,stype,fname,dtype_filter=None):
    r=reader(table); h=next(r); ix={c:i for i,c in enumerate(h)}
    els=[e for e in ELS if e in ix]
    n=w=0
    with open(f"{OUT}/{fname}","w",newline="") as fo:
        wr=csv.writer(fo); wr.writerow(HEAD)
        for row in r:
            n+=1
            if dtype_filter and not dtype_filter(row[ix['Data_Type']].strip()): continue
            lon=row[ix['Longitude_GDA2020']].strip(); lat=row[ix['Latitude_GDA2020']].strip()
            if not lon or not lat: continue
            job=row[ix['Job_No']].strip()
            vals={e:conv(row[ix[e]],job,e) for e in els}
            wr.writerow([lon,lat,row[ix['Sample_ID']].strip(),stype,row[ix['Report']].strip(),row[ix['Sample']].strip()]+[vals.get(e,MISSING) for e in ELS]); w+=1
    print(f"{table} -> {fname}: read {n:,} wrote {w:,} ({time.time()-t0:.0f}s)",flush=True)
surface("s_Seds.csv","STREA","qld_all_sediment.csv")
surface("s_Soil.csv","SOIL","qld_all_surfsoilgeochem.csv")
surface("s_RC.csv","ROCKC","qld_all_rockchips.csv")
# --- drilling: per-hole max grade (like gswa_all_drillhole_maxgrade), split shallow vs all by Drilling_Type
r=reader("h_Loc.csv"); h=next(r); ix={c:i for i,c in enumerate(h)}
collar={}
for row in r:
    lon=row[ix['Longitude_GDA2020']].strip(); lat=row[ix['Latitude_GDA2020']].strip()
    if lon and lat: collar[row[ix['Collar_ID']].strip()]=(lon,lat,row[ix['Drilling_Type']].strip().upper(),row[ix['Report']].strip(),row[ix['Hole_ID']].strip(),row[ix['Final_Depth']].strip())
print(f"collars with coords: {len(collar):,}",flush=True)
r=reader("h_Sample.csv"); h=next(r); ix={c:i for i,c in enumerate(h)}
els=[e for e in ELS if e in ix]
best={}  # collar -> {el: max numeric ppm}; negatives (below DL) kept only if nothing positive
n=0
for row in r:
    n+=1; cid=row[ix['Collar_ID']].strip()
    if cid not in collar: continue
    job=row[ix['Job_No']].strip(); d=best.setdefault(cid,{})
    for e in els:
        s=conv(row[ix[e]],job,e)
        if s==MISSING: continue
        v=float(s); cur=d.get(e)
        if cur is None or v>cur: d[e]=v
print(f"h_Sample rows read: {n:,}; holes with assays: {len(best):,} ({time.time()-t0:.0f}s)",flush=True)
SHALLOW={'RAB','AC','VAC','AUG','AUGER','AIRCORE','RABAC','VACUUM','HAND','PERC'}
with open(f"{OUT}/qld_all_drillhole_maxgrade.csv","w",newline="") as fa, open(f"{OUT}/qld_all_shallowdrill.csv","w",newline="") as fs:
    wa=csv.writer(fa); ws=csv.writer(fs); wa.writerow(HEAD+["DRILL_TYPE","FINAL_DEPTH"]); ws.writerow(HEAD+["DRILL_TYPE","FINAL_DEPTH"])
    na=ns=0; dtypes=collections.Counter()
    for cid,d in best.items():
        lon,lat,dt,rep,hid,dep=collar[cid]; dtypes[dt]+=1
        rowv=[lon,lat,cid,"DRILL",rep,hid]+[f"{d[e]:.6g}" if e in d else MISSING for e in ELS]+[dt,dep]
        wa.writerow(rowv); na+=1
        try: shallow=(dt in SHALLOW) or (dep and float(dep)<=30)
        except: shallow=dt in SHALLOW
        if shallow: rowv[3]="SHALL"; ws.writerow(rowv); ns+=1
print(f"drill maxgrade: {na:,} holes; shallow subset: {ns:,}; drilling types: {dtypes.most_common(10)}")
print("\nunit resolution stats:"); 
for k,v in sorted(stats.items()): print(f"  {k}: {v:,}")
print(f"done in {time.time()-t0:.0f}s")
