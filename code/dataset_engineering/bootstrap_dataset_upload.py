import base64
from pathlib import Path
import subprocess

root='/home/ubuntu/rz/object_detection_dataset_test'
code="""
from pathlib import Path
p=Path('/home/ubuntu/rz/object_detection_dataset_test')
assert all(not q.is_symlink() for q in [p,*p.parents])
owner=p/'_management/OWNER.txt'
assert not p.exists() or (owner.is_file() and owner.read_text().strip()=='rz-dataset-ingest-20260921'), 'existing unowned target'
(p/'_management').mkdir(parents=True,exist_ok=True)
if not owner.exists():
    with owner.open('x') as f: f.write('rz-dataset-ingest-20260921')
print(p)
"""
encoded=base64.b64encode(code.encode()).decode()
subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu8-team',f'''python3 -c "import base64;exec(base64.b64decode('{encoded}'))"'''],check=True)
for source in [Path(__file__).with_name('dataset_remote_ingest.py'),Path('E:/datasets/_management/vendor/rarfile.py')]:
    subprocess.run(['scp','-o','BatchMode=yes',str(source),'gpu8-team:'+root+'/_management/'+source.name],check=True)
print('bootstrap ready')
