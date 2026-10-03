"""Create a read-only-use guide in the owned remote data root; never clobber others."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from dataset_upload_worker import SSH,REMOTE

p=Path('E:/datasets/_management/SERVER_DATASET_INDEX_20260921.md')
data=p.read_bytes();encoded=base64.b64encode(data).decode();fingerprint=hashlib.sha256(data).hexdigest()
code=f"""
import base64,hashlib,sys
sys.path.insert(0,'{REMOTE}/_management')
from dataset_remote_ingest import safe,ROOT,OWNER
assert (ROOT/'_management/OWNER.txt').read_text().strip()==OWNER
p=safe('README.md');data=base64.b64decode('{encoded}')
if p.exists():
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()=='{fingerprint}', 'Existing README differs; preserve and inspect'
else:
    with p.open('xb') as f:f.write(data)
assert hashlib.sha256(p.read_bytes()).hexdigest()=='{fingerprint}'
print('README hash verified: {fingerprint}')
"""
payload=base64.b64encode(code.encode()).decode()
subprocess.run(SSH+[f'''python3 -c "import base64;exec(base64.b64decode('{payload}'))"'''],check=True,timeout=45)
print(json.dumps(dict(local=str(p),remote=REMOTE+'/README.md',sha256=fingerprint)))
