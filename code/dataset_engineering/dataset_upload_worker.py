"""Upload only manifest-listed validated dataset files; resumable SHA-checked SSH."""
import base64
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

MGT=Path('E:/datasets/_management')
MANIFEST=MGT/'UPLOAD_MANIFEST_20260921.json'
STATE=MGT/'UPLOAD_STATE_20260921.json'
REMOTE='/home/ubuntu/rz/object_detection_dataset_test'
PY='/home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','gpu8-team']

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)

def sha(path,limit=None):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while limit is None or limit>0:
            b=f.read(4*1024*1024 if limit is None else min(limit,4*1024*1024))
            if not b: break
            h.update(b)
            if limit is not None: limit-=len(b)
    return h.hexdigest()

def cmd(job,action,**extra):
    payload=base64.b64encode(json.dumps(dict(job,action=action,**extra)).encode()).decode()
    return SSH+[PY+' '+REMOTE+'/_management/dataset_remote_ingest.py '+payload]

def remote(job,action):
    r=subprocess.run(cmd(job,action),capture_output=True,text=True,encoding='utf-8',timeout=7200)
    if r.returncode: raise RuntimeError(r.stderr[-4000:])
    return json.loads(r.stdout)

def run():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    state=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {'results':{}}
    state.update(status='running',pid=os.getpid(),started=time.time())
    save(STATE,state)
    for job in manifest['files']:
        key=job['remote']
        prior=state['results'].get(key,{})
        if prior.get('status')=='complete' and prior.get('sha256')==job['sha256']: continue
        state.update(current=key,phase='checking_local_hash',updated=time.time());save(STATE,state)
        src=Path(job['local'])
        try:
            if src.stat().st_size!=job['bytes'] or sha(src)!=job['sha256']: raise ValueError('local source changed')
            for attempt in range(3):
                try:
                    probe=remote(job,'probe')
                    if probe['final']:
                        if probe['bytes']!=job['bytes'] or probe['sha256']!=job['sha256']: raise ValueError('existing remote final differs; not overwritten')
                        break
                    offset=probe['bytes']
                    if offset>job['bytes'] or sha(src,offset)!=probe['sha256']: raise ValueError('remote partial prefix differs; not overwritten')
                    state.update(phase='uploading',file_bytes=job['bytes'],sent_bytes=offset,updated=time.time());save(STATE,state)
                    print('UPLOAD',key,offset,'/',job['bytes'],flush=True)
                    started=time.monotonic();last=started;sent=offset
                    with (MGT/'upload_transport.stderr.log').open('ab') as err:
                        proc=subprocess.Popen(cmd(job,'receive',offset=offset),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
                        try:
                            with src.open('rb') as f:
                                f.seek(offset)
                                while True:
                                    b=f.read(1024*1024)
                                    if not b: break
                                    proc.stdin.write(b);sent+=len(b)
                                    now=time.monotonic()
                                    if now-last>=5:
                                        state.update(sent_bytes=sent,bytes_per_second=(sent-offset)/(now-started),updated=time.time());save(STATE,state);last=now
                            proc.stdin.close()
                            out=proc.stdout.read()
                            rc=proc.wait(timeout=600)
                            if rc: raise RuntimeError('SSH receive failed, code '+str(rc))
                            json.loads(out)
                        finally:
                            if proc.poll() is None:
                                proc.terminate();proc.wait(timeout=20)
                    break
                except (RuntimeError,OSError,subprocess.SubprocessError) as e:
                    if attempt==2: raise
                    print('RETRY',key,str(e),flush=True);time.sleep(10)
            result={'sha256':job['sha256'],'bytes':job['bytes'],'status':'complete','finished':time.time()}
            if job.get('extract'):
                state.update(phase='remote_extraction_and_decode',updated=time.time());save(STATE,state)
                result['extraction']=remote(job,'extract')
            state['results'][key]=result
            save(STATE,state)
            print('COMPLETE',key,flush=True)
        except Exception as e:
            state['results'][key]=dict(status='failed',error=str(e),time=time.time())
            save(STATE,state)
            print('FAILED',key,str(e),flush=True)
            if any(token in str(e) for token in ('Connection refused','Connection timed out','No route to host','stdio forwarding failed')):
                state.update(status='transport_unavailable_requires_check',updated=time.time());save(STATE,state)
                return
    state.update(status='batch_finished',updated=time.time());save(STATE,state)

if __name__=='__main__':
    with (MGT/'upload_worker.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        run()
