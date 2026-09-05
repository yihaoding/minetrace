#!/bin/bash
cd ${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/papers/anomaly_ref
IDS=$(cut -d'|' -f1 ids5.txt | paste -sd,)
curl -sSL "https://export.arxiv.org/api/query?id_list=${IDS}&max_results=60" -o meta5.xml
echo "meta5.xml bytes: $(wc -c < meta5.xml)"
python3 - <<'PY'
import xml.etree.ElementTree as ET, re
ns={'a':'http://www.w3.org/2005/Atom'}
root=ET.parse('meta5.xml').getroot()
for e in root.findall('a:entry',ns):
    aid=re.sub(r'v\d+$','',e.find('a:id',ns).text.split('/abs/')[-1])
    t=' '.join(e.find('a:title',ns).text.split())
    p=e.find('a:published',ns).text[:10]
    au=[a.find('a:name',ns).text for a in e.findall('a:author',ns)]
    first=au[0].split()[-1] if au else '?'
    print(f"{aid}\t{p}\t{first}{'+' if len(au)>1 else ''}\t{t}")
PY
