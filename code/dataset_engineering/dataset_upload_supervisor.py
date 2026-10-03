"""Continue batches after the current worker exits; stop on failures for review."""
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
from register_manual_uploads import main as register

MGT=Path('E:/datasets/_management')

def worker_busy():
    with (MGT/'upload_worker.lock').open('a+b') as f:
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:return True
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        return False

def main():
    started=time.time()
    while time.time()-started<6*3600:
        manifest=register()
        state=json.loads((MGT/'UPLOAD_STATE_20260921.json').read_text(encoding='utf-8'))
        if not worker_busy():
            failed={k:v for k,v in state['results'].items() if v.get('status')=='failed'}
            if failed:raise RuntimeError('Upload/extraction requires review: '+json.dumps(failed,ensure_ascii=True))
            todo=[j for j in manifest['files'] if state['results'].get(j['remote'],{}).get('status')!='complete']
            if todo:
                print('START NEXT BATCH',len(todo),flush=True)
                with (MGT/'UPLOAD_20260921.log').open('ab') as out,(MGT/'UPLOAD_20260921.err.log').open('ab') as err:
                    subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('dataset_upload_worker.py'))],stdout=out,stderr=err,check=True)
            elif not manifest.get('pending'):
                print('ALL MANIFEST FILES FINISHED; final human/agent audit required',flush=True);return
        time.sleep(30)
    raise TimeoutError('Six-hour bounded supervisor window ended; inspect before resuming')

if __name__=='__main__':
    with (MGT/'upload_supervisor.lock').open('a+b') as f:
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        print('SUPERVISOR',os.getpid(),flush=True);main()
