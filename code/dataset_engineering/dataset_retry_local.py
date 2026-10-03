"""Retry only already-failed files of an explicitly selected local dataset."""
import os
os.environ['DATASET_ROOT']='E:/datasets'
os.environ['HTTPS_PROXY']='http://127.0.0.1:7897'
os.environ['HTTP_PROXY']='http://127.0.0.1:7897'
import json
import sys
import time
from dataset_acquire import ROOT,MGT,fetch,save

if __name__=='__main__':
    ds=sys.argv[1]
    jobs=json.loads((MGT/'download_manifest.json').read_text(encoding='utf-8'))
    pending=[]
    for j in jobs:
        if j['dataset']!=ds:
            continue
        p=ROOT/ds/'checks'/(j['name'].replace('/','_')+'.json')
        allowed={'failed','running'} if '--resume-interrupted' in sys.argv else {'failed'}
        if p.exists() and json.loads(p.read_text())['status'] in allowed:
            if ds=='LLVIP':
                j['range_chunked']=True
            pending.append(j)
    save(MGT/(ds+'.retry_runner.json'),dict(pid=os.getpid(),started=time.time(),files=[j['name'] for j in pending],status='running'))
    results=[fetch(j) for j in pending]
    save(MGT/(ds+'.retry_results.json'),results)
    save(MGT/(ds+'.retry_runner.json'),dict(pid=os.getpid(),finished=time.time(),status='finished'))
