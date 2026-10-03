"""Task-scoped receive/extract helper. No training, installs, or shared config writes."""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
import zlib

ROOT = Path('/home/ubuntu/rz/object_detection_dataset_test')
OWNER = 'rz-dataset-ingest-20260921'

def safe(rel):
    p = PurePosixPath(rel)
    if p.is_absolute() or not p.parts or '..' in p.parts or '\\' in rel:
        raise ValueError('unsafe path')
    out = ROOT.joinpath(*p.parts)
    for part in [ROOT, *out.parents, out]:
        if part.is_symlink():
            raise ValueError('symlink in task path')
    if not out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('outside task root')
    return out

def sha(path, limit=None):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while limit is None or limit > 0:
            b = f.read(4*1024*1024 if limit is None else min(limit,4*1024*1024))
            if not b: break
            h.update(b)
            if limit is not None: limit -= len(b)
    return h.hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.ingest_tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
    os.replace(tmp,path)

def apple_double_metadata(path):
    if not path.name.startswith('._'):
        return False
    with path.open('rb') as f:
        return f.read(8) == bytes.fromhex('0005160700020000')

def extract(job, src):
    dest = safe(job['dataset']+'/raw/'+src.name)
    marker = dest/'_INGEST_COMPLETE.json'
    if marker.is_file():
        result=json.loads(marker.read_text())
        if result['sha256'] != job['sha256']: raise ValueError('extraction marker mismatch')
        return result
    # A prior decode failure can leave a complete extraction. Reuse it only after
    # every member below matches the archive; never overwrite a partial directory.
    existing = dest.exists() and any(dest.iterdir())
    dest.mkdir(parents=True, exist_ok=True)
    kind = 'zip' if zipfile.is_zipfile(src) else ('rar' if src.suffix.lower()=='.rar' else 'tar')
    if kind == 'zip':
        ar = zipfile.ZipFile(src)
        if job.get('zip_password'): ar.setpassword(job['zip_password'].encode())
        rows=[(i.filename,i.is_dir(),stat.S_ISLNK(i.external_attr>>16),i.file_size,i.CRC,i) for i in ar.infolist()]
    elif kind == 'tar':
        ar = tarfile.open(src)
        rows=[(i.name,i.isdir(),not(i.isdir() or i.isfile()),i.size,None,i) for i in ar.getmembers()]
    else:
        sys.path.insert(0,str(ROOT/'_management'))
        import rarfile
        ar=rarfile.RarFile(src)
        rows=[(i.filename,i.isdir(),i.is_symlink(),i.file_size,i.CRC,i) for i in ar.infolist()]
    total=0
    seen=set()
    for name,isdir,link,size,crc,info in rows:
        p=PurePosixPath(name.replace('\\','/'))
        if p.is_absolute() or '..' in p.parts or (p.parts and ':' in p.parts[0]) or link:
            raise ValueError('unsafe member: '+name)
        if not isdir and (not p.parts or str(p) in seen): raise ValueError('duplicate/empty member')
        if not isdir: seen.add(str(p))
        total += size
    if total>60*1024**3 or shutil.disk_usage(dest).free<total+20*1024**3:
        raise ValueError('extraction disk budget exceeded')
    if existing:
        pass  # Verification-only recovery, with no extraction writes.
    elif kind=='zip' and job.get('zip_password'):
        # Members were safety-audited and destination is empty. Native unzip verifies
        # compressed CRC; below independently checks every extracted file and image.
        subprocess.run(['unzip','-qq','-n','-P',job['zip_password'],str(src),'-d',str(dest)],check=True,stdout=subprocess.DEVNULL)
    elif kind=='rar':
        subprocess.run(['unrar','x','-idq','-o-',str(src),str(dest)+'/'],check=True,stdout=subprocess.DEVNULL)
    else:
        for name,isdir,link,size,crc,info in rows:
            p=PurePosixPath(name.replace('\\','/'))
            if not p.parts: continue
            out=dest.joinpath(*p.parts)
            if isdir: out.mkdir(parents=True,exist_ok=True); continue
            out.parent.mkdir(parents=True,exist_ok=True)
            inp=ar.extractfile(info) if kind=='tar' else ar.open(info)
            with inp, out.open('xb') as f: shutil.copyfileobj(inp,f,1024*1024)
    from PIL import Image
    count=0
    images=0
    metadata=0
    state=safe('_management/extraction_current.json')
    for name,isdir,link,size,crc,info in rows:
        if isdir: continue
        out=safe(job['dataset']+'/raw/'+src.name+'/'+str(PurePosixPath(name.replace('\\','/'))))
        if out.is_symlink() or not out.is_file() or out.stat().st_size!=size:
            raise ValueError('extracted size/type mismatch: '+name)
        if crc is not None:
            actual=0
            with out.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''): actual=zlib.crc32(b,actual)
            if actual & 0xffffffff != crc: raise ValueError('extracted CRC mismatch: '+name)
        else:
            expected=hashlib.sha256()
            reader=ar.extractfile(info) if kind=='tar' else ar.open(info)
            with reader as f:
                for b in iter(lambda:f.read(1024*1024),b''): expected.update(b)
            if sha(out)!=expected.hexdigest():raise ValueError('extracted content mismatch: '+name)
        auxiliary=apple_double_metadata(out)
        if auxiliary:metadata+=1
        if not auxiliary and out.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}:
            with Image.open(out) as im: im.verify()
            with Image.open(out) as im: im.load()
            images+=1
        count+=1
        if count%500==0:
            save(state,dict(dataset=job['dataset'],archive=src.name,files_checked=count,images_decoded=images,time=time.time()))
    ar.close()
    result=dict(status='archive_hash_extraction_and_image_decode_passed',sha256=job['sha256'],files=count,images=images,apple_double_metadata_files=metadata,bytes=total,verification_only_recovery=existing,finished=time.time(),annotation_validation=job.get('validation_note'))
    save(marker,result)
    return result

def main(job):
    if ROOT.is_symlink() or not (ROOT/'_management/OWNER.txt').is_file() or (ROOT/'_management/OWNER.txt').read_text().strip()!=OWNER:
        raise ValueError('task root ownership not established')
    target=safe(job['remote'])
    target.parent.mkdir(parents=True,exist_ok=True)
    part=safe(job['remote']+'.upload-part')
    mode=job['action']
    if mode=='probe':
        p=target if target.exists() else part
        if p.exists():
            if not p.is_file(): raise ValueError('not regular file')
            return dict(final=p==target,bytes=p.stat().st_size,sha256=sha(p))
        return dict(final=False,bytes=0,sha256=hashlib.sha256(b'').hexdigest())
    if mode=='receive':
        with safe('_management/receive.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if target.exists(): raise ValueError('final already exists')
            if (part.stat().st_size if part.exists() else 0)!=job['offset']: raise ValueError('partial changed')
            if shutil.disk_usage(target.parent).free < job['bytes']-job['offset']+20*1024**3: raise ValueError('disk reserve')
            with part.open('ab') as f:
                while True:
                    b=sys.stdin.buffer.read(1024*1024)
                    if not b: break
                    f.write(b)
                f.flush(); os.fsync(f.fileno())
            if part.stat().st_size!=job['bytes'] or sha(part)!=job['sha256']: raise ValueError('uploaded hash/size mismatch')
            os.rename(part,target)
        return dict(status='uploaded_hash_verified',bytes=job['bytes'],sha256=job['sha256'])
    if mode=='extract':
        if target.stat().st_size!=job['bytes'] or sha(target)!=job['sha256']: raise ValueError('source hash mismatch before extraction')
        return extract(job,target)
    raise ValueError('unknown action')

if __name__=='__main__':
    print(json.dumps(main(json.loads(base64.b64decode(sys.argv[1]))),ensure_ascii=True),flush=True)
