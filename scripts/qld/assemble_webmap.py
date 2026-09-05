"""Inject basemap + per-metal JSON into webmap_template.html -> self-contained page."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; W=ROOT/"reports"/"qld"/"webmap"
metals=[m for m in ["Cu","Au","W","Sn"] if (W/f"{m}.json").exists()]
data={"basemap":json.load(open(W/"basemap.json")),"metals":{m:json.load(open(W/f"{m}.json")) for m in metals}}
tpl=(ROOT/"scripts"/"qld"/"webmap_template.html").read_text()
blob="const DATA="+json.dumps(data,separators=(",",":"),ensure_ascii=False)+";"
out=Path(sys.argv[1]) if len(sys.argv)>1 else W/"qld_webmap.html"
out.write_text(tpl.replace("/*__DATA__*/",blob))
print("wrote",out,f"{out.stat().st_size/1e6:.1f} MB; metals={metals}; probes="+", ".join(f"{m}:{len(data['metals'][m]['probes'])}" for m in metals))
