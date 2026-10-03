"""Add publisher split documents and verify Drive content lengths in the saved manifest."""
import json
import re
import requests
from dataset_acquire import MGT, save, fetch

path=MGT/'download_manifest.json'
jobs=json.loads(path.read_text())
extra=[
    ('ExDark','protocol/imageclasslist.txt','https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image-Dataset/810e4a0c5d6eddebd7d2280d371a090decc719bb/Groundtruth/imageclasslist.txt'),
    ('ExDark','protocol/README.md','https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image-Dataset/810e4a0c5d6eddebd7d2280d371a090decc719bb/Groundtruth/README.md'),
    ('LLVIP','protocol/README.md','https://raw.githubusercontent.com/bupt-ai-cz/LLVIP/c1a655cce437ebfd990a97b04fc48fbb99f4c47b/README.md'),
    ('LLVIP','protocol/download_dataset.md','https://raw.githubusercontent.com/bupt-ai-cz/LLVIP/c1a655cce437ebfd990a97b04fc48fbb99f4c47b/download_dataset.md'),
]
for ds,name,url in extra:
    if not any(j['dataset']==ds and j['name']==name for j in jobs):
        jobs.append(dict(dataset=ds,name=name,url=url,size=None,checksum=None))
for j in jobs:
    if 'drive.usercontent.google.com' in j['url']:
        try:
            with requests.get(j['url'],headers={'Range':'bytes=0-0'},stream=True,timeout=(15,30)) as r:
                r.raise_for_status()
                cr=r.headers.get('Content-Range','')
                m=re.fullmatch(r'bytes 0-0/(\d+)',cr)
                if r.status_code!=206 or not m:
                    raise ValueError(f'unexpected range response {r.status_code}, {cr}')
                j['size']=int(m[1])
                j['last_modified']=r.headers.get('Last-Modified')
                print(j['dataset'],j['size'],flush=True)
        except Exception as e:
            print(j['dataset'],'metadata error',str(e),flush=True)
save(path,jobs)
for job in jobs:
    if job['name'].startswith('protocol/'):
        fetch(job)
