"""Append successfully audited user archives to the frozen upload manifest."""
import json
import os
from pathlib import Path
import time
from prepare_dataset_upload import sha

ROOT=Path('E:/datasets')
MANIFEST=ROOT/'_management/UPLOAD_MANIFEST_20260921.json'

def main():
    m=json.loads(MANIFEST.read_text(encoding='utf-8'));known={j['remote']:j for j in m['files']};added=[]
    for ds in ['HRSID','VisDrone2019-DET','LLVIP','ZeroWaste-f']:
        base=ROOT/ds;p=base/'MANUAL_VALIDATION.json'
        if not p.exists():continue
        v=json.loads(p.read_text(encoding='utf-8'))
        if v.get('errors') or not v.get('archives'):continue
        if ds=='VisDrone2019-DET' and 'test-dev' not in v.get('splits',{}):continue
        note=v['status']+'; '+('; '.join(v.get('warnings',[])))
        jobs=[]
        for a in v['archives']:
            job=dict(dataset=ds,local=a['local'],remote=ds+'/archives/'+a['name'],bytes=a['bytes'],sha256=a['sha256'],extract=True,validation_note=note)
            if a.get('zip_password'):job['zip_password']=a['zip_password']
            jobs.append(job)
        docs=[base/'MANUAL_VALIDATION.json',base/'MANUAL_VALIDATION.md']
        if ds=='ZeroWaste-f':docs += [base/'EXTRACTION_VALIDATION.json',base/'source_metadata.json']
        if (base/'archives/protocol').exists(): docs+=sorted((base/'archives/protocol').rglob('*'))
        for doc in docs:
            if doc.is_file():jobs.append(dict(dataset=ds,local=str(doc),remote=ds+'/'+doc.relative_to(base).as_posix(),bytes=doc.stat().st_size,sha256=sha(doc),extract=False))
        for job in jobs:
            key=job['remote']
            if key in known:
                if known[key]['sha256']!=job['sha256']:raise ValueError('Registered source/report changed; inspect before overwriting '+key)
            else:m['files'].append(job);known[key]=job;added.append(key)
        if ds in m.get('pending',[]):m['pending'].remove(ds)
    if added:
        m['updated']=time.time();tmp=MANIFEST.with_suffix('.tmp');tmp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,MANIFEST)
        print('REGISTERED',len(added),'files:',','.join(sorted({key.split('/')[0] for key in added})),flush=True)
    return m

if __name__=='__main__':main()
