"""Append reviewed supplemental audit files, without replacing any frozen entry."""
import json
import os
from pathlib import Path
import time
from prepare_dataset_upload import sha

root=Path('E:/datasets');path=root/'_management/UPLOAD_MANIFEST_20260921.json'
m=json.loads(path.read_text(encoding='utf-8'))
assert m['pending']==['ZeroWaste-f'], 'Wait until manual dataset registration is complete'
known={j['remote']:j for j in m['files']}
for p in sorted((root/'HRSID').glob('SUPPLEMENTAL_SPLIT_CONSISTENCY.*')):
    key='HRSID/'+p.name;fingerprint=sha(p)
    if key in known:
        assert known[key]['sha256']==fingerprint
        continue
    m['files'].append(dict(dataset='HRSID',local=str(p),remote=key,bytes=p.stat().st_size,sha256=fingerprint,extract=False))
m['updated']=time.time();tmp=path.with_suffix('.supplemental.tmp');tmp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
print('Supplemental audit files registered',flush=True)
