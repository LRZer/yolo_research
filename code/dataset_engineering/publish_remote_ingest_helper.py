"""Atomically deploy only this task's helper, preserving active imports and data."""
import base64
import hashlib
from pathlib import Path
import subprocess
from dataset_upload_worker import SSH,REMOTE

p=Path(__file__).with_name('dataset_remote_ingest.py')
h=hashlib.sha256(p.read_bytes()).hexdigest()
tmp=REMOTE+'/_management/helper_deploy_'+h+'.py'
subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(p),'gpu8-team:'+tmp],check=True,timeout=60)
code=f"""
import hashlib,os
from pathlib import Path
r=Path('{REMOTE}');p=Path('{tmp}');t=r/'_management/dataset_remote_ingest.py'
assert (r/'_management/OWNER.txt').read_text().strip()=='rz-dataset-ingest-20260921'
assert all(not x.is_symlink() for x in [p,t,*p.parents])
assert hashlib.sha256(p.read_bytes()).hexdigest()=='{h}'
os.replace(p,t)
print('Atomic helper deployment SHA256 {h}')
"""
payload=base64.b64encode(code.encode()).decode()
subprocess.run(SSH+[f'''python3 -c "import base64;exec(base64.b64decode('{payload}'))"'''],check=True,timeout=60)
