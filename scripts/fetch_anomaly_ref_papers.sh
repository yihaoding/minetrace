#!/bin/bash
set -u
cd ${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/papers/anomaly_ref
IDS=$(cut -d'|' -f1 ids5.txt | paste -sd,)
curl -sS "http://export.arxiv.org/api/query?id_list=${IDS}&max_results=60" -o meta5.xml
echo "meta5.xml bytes: $(wc -c < meta5.xml)"
# print id | published | title for verification
python3 - <<'PY'
import xml.etree.ElementTree as ET, re
ns={'a':'http://www.w3.org/2005/Atom'}
root=ET.parse('meta5.xml').getroot()
for e in root.findall('a:entry',ns):
    aid=e.find('a:id',ns).text.split('/abs/')[-1]
    aid=re.sub(r'v\d+$','',aid)
    t=' '.join(e.find('a:title',ns).text.split())
    p=e.find('a:published',ns).text[:10]
    print(f"{aid}\t{p}\t{t}")
PY
echo "--- downloading pdfs"
while IFS='|' read -r id name; do
  out="pdf/${name}_${id}.pdf"
  if [ -s "$out" ]; then echo "exists $out"; continue; fi
  curl -sSL -A "Mozilla/5.0 (research paper fetch)" "https://arxiv.org/pdf/${id}" -o "$out"
  if head -c 4 "$out" | grep -q '%PDF'; then echo "ok   $out $(du -k "$out" | cut -f1)K"; else echo "FAIL $out"; rm -f "$out"; fi
  sleep 1
done < ids5.txt
