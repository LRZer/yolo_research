"""Freeze a manifest for the six previously audited datasets, preserving source files."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('E:/datasets')
OUT=ROOT/'_management/UPLOAD_MANIFEST_20260921.json'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    manifest=json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else dict(files=[],created=time.time())
    known={j['remote'] for j in manifest['files']}
    for ds in ['HIT-UAV','DUO','AP-10K','SeaClear','ExDark','ORCA']:
        base=ROOT/ds
        v=json.loads((base/'VALIDATION.json').read_text(encoding='utf-8'))
        assert not v['errors'],(ds,v['errors'])
        assert v['status']=='verified_usable_original_format' or ds=='ORCA'
        files=sorted((base/'archives').rglob('*'))
        files += [p for p in base.iterdir() if p.is_file() and p.name in {'VALIDATION.json','VALIDATION.md','REVIEW.md','READING_NOTES.md','source_metadata.json'}]
        for p in files:
            if not p.is_file() or p.name.endswith(('.part','.lock')): continue
            rel=ds+'/'+p.relative_to(base).as_posix()
            if rel in known: continue
            fingerprint=sha(p)
            if p.is_relative_to(base/'archives'):
                name=p.relative_to(base/'archives').as_posix()
                chk=base/'checks'/(name.replace('/','_')+'.json')
                if not chk.exists(): raise ValueError('missing file audit: '+str(chk))
                c=json.loads(chk.read_text(encoding='utf-8'))
                assert c['status']=='download_verified' and c['sha256']==fingerprint and c['bytes']==p.stat().st_size,str(p)
            manifest['files'].append(dict(dataset=ds,local=str(p),remote=rel,bytes=p.stat().st_size,sha256=fingerprint,extract=p.name.endswith(('.zip','.rar','.tar.gz','.tgz')),validation_note=v['status']+'; '+('; '.join(v.get('warnings',[])))))
            known.add(rel)
        print(ds,'manifest ready',flush=True)
    manifest.update(updated=time.time(),destination='gpu8-team:/home/ubuntu/rz/object_detection_dataset_test',pending=['LLVIP','HRSID','VisDrone2019-DET','ZeroWaste-f'],not_downloaded=['JODD'])
    tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,OUT)
    print('FILES',len(manifest['files']),'BYTES',sum(j['bytes'] for j in manifest['files']),flush=True)

if __name__=='__main__': main()
