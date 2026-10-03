"""Await complete official archive, then native safe extraction and full CRC/decode."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import time
import zipfile
import zlib
from PIL import Image

BASE=Path('E:/datasets/ZeroWaste-f')
SRC=BASE/'archives/zerowaste-f-final.zip'
DEST=BASE/'raw/zerowaste-f-final.zip'
PASS='UP#1VuX409z4'

def digest(p,alg):
    h=hashlib.new(alg)
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(r):
    p=BASE/'EXTRACTION_VALIDATION.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)

def main():
    r=dict(status='awaiting_complete_archive',pid=os.getpid(),started=time.time(),errors=[])
    save(r);deadline=time.monotonic()+7200
    while not SRC.exists():
        if time.monotonic()>deadline:raise TimeoutError('Complete archive not available within two hours')
        time.sleep(15)
    if SRC.stat().st_size!=7518242799:raise ValueError('wrong archive size')
    md5=digest(SRC,'md5');fingerprint=digest(SRC,'sha256')
    if md5!='e26e31a58080bca6782dca0e56074c5d':raise ValueError('publisher MD5 mismatch')
    r.update(status='extracting',md5=md5,sha256=fingerprint,bytes=SRC.stat().st_size);save(r)
    with zipfile.ZipFile(SRC) as z:
        rows=z.infolist();seen=set();total=0
        for i in rows:
            p=PurePosixPath(i.filename.replace('\\','/'))
            if p.is_absolute() or '..' in p.parts or (p.parts and ':' in p.parts[0]) or (i.external_attr>>16)&0o170000==0o120000:raise ValueError('unsafe ZIP member')
            if not i.is_dir():
                if str(p) in seen:raise ValueError('duplicate ZIP member')
                seen.add(str(p))
            total+=i.file_size
        if total>60*1024**3 or shutil.disk_usage(BASE).free<total+20*1024**3:raise ValueError('disk budget exceeded')
        if DEST.exists() and any(DEST.iterdir()):raise ValueError('existing extraction, inspect before retry')
        if any(p.is_symlink() for p in [DEST,*DEST.parents]):raise ValueError('symlink destination')
        DEST.mkdir(parents=True,exist_ok=True)
        subprocess.run(['tar','-xf',str(SRC),'--passphrase',PASS,'-C',str(DEST)],check=True)
        count=0;images=0;jsonfiles=[];dimensions={}
        for i in rows:
            if i.is_dir():continue
            out=DEST.joinpath(*PurePosixPath(i.filename.replace('\\','/')).parts)
            if out.is_symlink() or not out.is_file() or out.stat().st_size!=i.file_size:raise ValueError('size/type mismatch '+i.filename)
            crc=0
            with out.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):crc=zlib.crc32(b,crc)
            if crc&0xffffffff!=i.CRC:raise ValueError('CRC mismatch '+i.filename)
            if out.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}:
                with Image.open(out) as im:im.verify()
                with Image.open(out) as im:im.load();dimensions[out.relative_to(DEST).as_posix()]=list(im.size)
                images+=1
            if out.suffix.lower()=='.json':
                doc=json.loads(out.read_text(encoding='utf-8-sig'))
                jsonfiles.append(dict(path=out.relative_to(DEST).as_posix(),keys=list(doc)[:12] if isinstance(doc,dict) else None))
            count+=1
            if count%1000==0:
                r.update(status='crc_and_decode',files_checked=count,images_decoded=images,updated=time.time());save(r);print('ZeroWaste checked',count,'images',images,flush=True)
    (BASE/'IMAGE_DIMENSIONS.json').write_text(json.dumps(dimensions),encoding='utf-8')
    r.update(status='archive_crc_and_image_decode_passed_annotation_audit_pending',files_checked=count,images_decoded=images,json_files=jsonfiles,finished=time.time());save(r)
    print('EXTRACTION DONE',count,images,'JSON',len(jsonfiles),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        r=json.loads((BASE/'EXTRACTION_VALIDATION.json').read_text(encoding='utf-8'));r.update(status='failed',errors=[str(e)],finished=time.time());save(r);raise
