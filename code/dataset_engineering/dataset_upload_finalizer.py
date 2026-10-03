"""After every queued file completes, audit local originals and current remote contents."""
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
from dataset_upload_worker import SSH, PY, REMOTE, remote, sha, save

MGT=Path('E:/datasets/_management')
STATE=MGT/'FINALIZER_STATE_20260921.json'

def main():
    state=dict(status='waiting_for_all_transfers',pid=os.getpid(),started=time.time())
    save(STATE,state);deadline=time.monotonic()+6*3600
    while True:
        manifest=json.loads((MGT/'UPLOAD_MANIFEST_20260921.json').read_text(encoding='utf-8'))
        transfer=json.loads((MGT/'UPLOAD_STATE_20260921.json').read_text(encoding='utf-8'))
        ready=not manifest.get('pending') and all(transfer['results'].get(j['remote'],{}).get('status')=='complete' and transfer['results'][j['remote']].get('sha256')==j['sha256'] for j in manifest['files'])
        if ready:break
        if time.monotonic()>deadline:raise TimeoutError('Six-hour finalizer wait ended; final audit not run')
        time.sleep(45)
    state.update(status='verifying_local_originals',updated=time.time());save(STATE,state)
    originals=[]
    for j in manifest['files']:
        p=Path(j['local'])
        if not p.is_file() or p.stat().st_size!=j['bytes'] or sha(p)!=j['sha256']:raise ValueError('Local original/report changed '+str(p))
        originals.append(dict(path=str(p),bytes=j['bytes'],sha256=j['sha256']))
    save(MGT/'FINAL_LOCAL_ORIGINALS_AUDIT.json',dict(status='all_local_originals_preserved_hash_verified',files=originals,finished=time.time()))
    state.update(status='deploying_final_audit',updated=time.time());save(STATE,state)
    for p in [MGT/'UPLOAD_MANIFEST_20260921.json',Path(__file__).with_name('dataset_final_remote_audit.py')]:
        target='_management/'+p.name
        subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(p),'gpu8-team:'+REMOTE+'/'+target],check=True,timeout=120)
        probe=remote(dict(remote=target),'probe')
        if not probe['final'] or probe['bytes']!=p.stat().st_size or probe['sha256']!=sha(p):raise ValueError('Final audit deployment hash mismatch')
    state.update(status='remote_independent_final_audit',updated=time.time());save(STATE,state)
    with (MGT/'FINAL_REMOTE_AUDIT.log').open('wb') as out,(MGT/'FINAL_REMOTE_AUDIT.err.log').open('wb') as err:
        subprocess.run(SSH+[PY+' '+REMOTE+'/_management/dataset_final_remote_audit.py'],stdout=out,stderr=err,check=True,timeout=7200)
    payload=subprocess.run(SSH+['cat '+REMOTE+'/_management/FINAL_TRANSFER_AUDIT.json'],capture_output=True,check=True,timeout=120)
    result=json.loads(payload.stdout)
    save(MGT/'FINAL_TRANSFER_AUDIT.json',result)
    if result['status']!='all_manifest_files_and_extracted_content_verified' or result['errors'] or len(result['files'])!=len(manifest['files']):raise ValueError('Remote final audit did not pass')
    expected={j['remote']:(j['bytes'],j['sha256']) for j in manifest['files']}
    actual={j['path']:(j['bytes'],j['sha256']) for j in result['files']}
    if actual!=expected:raise ValueError('Remote final audit coverage mismatch')
    state.update(status='final_audit_passed_agent_review_and_readme_pending',datasets=len(result['datasets']),files=len(result['files']),finished=time.time());save(STATE,state)
    print('INDEPENDENT FINAL AUDIT PASSED; agent must review and deliver final index before marking goal complete',flush=True)

if __name__=='__main__':
    with (MGT/'upload_finalizer.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        except Exception as e:
            state=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
            state.update(status='failed_requires_review',error=str(e),finished=time.time());save(STATE,state);raise
