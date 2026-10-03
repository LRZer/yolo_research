"""Local-only download runner. Never transfers files to the shared server."""
import os
os.environ['DATASET_ROOT'] = 'E:/datasets'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from collections import defaultdict
import concurrent.futures as cf
import json
import time
from pathlib import Path
from dataset_acquire import ROOT, MGT, save, fetch

if __name__ == '__main__':
    jobs = json.loads((MGT/'download_manifest.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for j in jobs:
        groups[j['dataset']].append(j)
    jobs = [group[i] for i in range(max(map(len, groups.values()))) for group in groups.values() if i < len(group)]
    for ds in ['JODD','DUO','ORCA','AP-10K','LLVIP','HIT-UAV','ExDark','SeaClear']:
        (ROOT/ds).mkdir(parents=True,exist_ok=True)
    save(MGT/'local_runner.json',dict(pid=os.getpid(),status='running',started=time.time(),root=str(ROOT),workers=4,server_upload=False))
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,jobs))
    save(MGT/'download_results.json',results)
    save(MGT/'local_runner.json',dict(pid=os.getpid(),status='downloads_finished',finished=time.time(),root=str(ROOT),server_upload=False))
