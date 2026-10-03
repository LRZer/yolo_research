"""Independent end-state file/hash/extracted-content audit within the task root."""
import json
from pathlib import PurePosixPath
import sys
import tarfile
import time
import zipfile
import zlib
import hashlib
from dataset_remote_ingest import ROOT, OWNER, safe, sha, save, apple_double_metadata

def crc(path):
    value=0
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):value=zlib.crc32(b,value)
    return value&0xffffffff

def archive_audit(job,archive):
    dest=safe(job['dataset']+'/raw/'+archive.name)
    marker=json.loads((dest/'_INGEST_COMPLETE.json').read_text())
    assert marker['sha256']==job['sha256'] and marker['status']=='archive_hash_extraction_and_image_decode_passed'
    if zipfile.is_zipfile(archive):
        ar=zipfile.ZipFile(archive)
        rows=[(i.filename,i.file_size,i.CRC,i) for i in ar.infolist() if not i.is_dir()]
    elif archive.suffix=='.rar':
        sys.path.insert(0,str(ROOT/'_management'))
        import rarfile
        ar=rarfile.RarFile(archive)
        rows=[(i.filename,i.file_size,i.CRC,i) for i in ar.infolist() if not i.isdir()]
    else:
        ar=tarfile.open(archive)
        rows=[(i.name,i.size,None,i) for i in ar.getmembers() if i.isfile()]
    try:
        total=0
        images=0
        metadata=0
        expected=set()
        for name,size,wanted_crc,info in rows:
            p=PurePosixPath(name.replace('\\','/'))
            expected.add(str(p))
            out=safe(job['dataset']+'/raw/'+archive.name+'/'+str(p))
            if not out.is_file() or out.stat().st_size!=size:raise ValueError('missing/size mismatch '+str(out))
            if wanted_crc is not None:
                if crc(out)!=wanted_crc:raise ValueError('extracted CRC mismatch '+str(out))
            else:
                h=hashlib.sha256()
                with ar.extractfile(info) as f:
                    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
                if h.hexdigest()!=sha(out):raise ValueError('extracted tar content mismatch '+str(out))
            auxiliary=apple_double_metadata(out)
            if auxiliary:
                metadata+=1
            elif out.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}:
                from PIL import Image
                with Image.open(out) as im:im.verify()
                with Image.open(out) as im:im.load()
                images+=1
            total+=size
        actual=set()
        for out in dest.rglob('*'):
            if out.is_symlink():raise ValueError('symlink found in extracted tree '+str(out))
            if out.is_file() and out.name!='_INGEST_COMPLETE.json':
                actual.add(out.relative_to(dest).as_posix())
        if actual!=expected:
            raise ValueError('extracted member set mismatch: missing='+repr(sorted(expected-actual)[:10])+' extra='+repr(sorted(actual-expected)[:10]))
        if marker['files']!=len(rows) or marker['bytes']!=total:raise ValueError('marker counts differ from archive')
        if marker['images']!=images:raise ValueError('marker image count differs from independent decode')
        return dict(files=len(rows),bytes=total,images_independently_decoded=images,apple_double_metadata_files=metadata,current_extracted_content_verified=True,annotation_validation=job.get('validation_note'))
    finally:ar.close()

def main():
    assert not ROOT.is_symlink() and (ROOT/'_management/OWNER.txt').read_text().strip()==OWNER
    manifest=json.loads(safe('_management/UPLOAD_MANIFEST_20260921.json').read_text())
    if manifest.get('pending'):raise ValueError('manifest still has pending datasets')
    r=dict(started=time.time(),status='running',files=[],errors=[],datasets={})
    output=safe('_management/FINAL_TRANSFER_AUDIT.json')
    for job in manifest['files']:
        path=safe(job['remote'])
        try:
            if not path.is_file() or path.stat().st_size!=job['bytes'] or sha(path)!=job['sha256']:raise ValueError('final archive/file hash mismatch '+job['remote'])
            row=dict(path=job['remote'],bytes=job['bytes'],sha256=job['sha256'],status='verified')
            if job.get('extract'):row['extraction']=archive_audit(job,path)
            elif path.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}:
                from PIL import Image
                with Image.open(path) as im:im.verify()
                with Image.open(path) as im:im.load();row['standalone_image_size']=list(im.size)
                row['standalone_image_decoded']=True
            r['files'].append(row)
            r['datasets'].setdefault(job['dataset'],{'files':0,'bytes':0})
            r['datasets'][job['dataset']]['files']+=1;r['datasets'][job['dataset']]['bytes']+=job['bytes']
            print('FINAL VERIFIED',job['remote'],flush=True)
        except Exception as e:r['errors'].append(str(e));print('FINAL FAILED',str(e),flush=True)
        save(output,r)
    r.update(status='all_manifest_files_and_extracted_content_verified' if not r['errors'] and len(r['files'])==len(manifest['files']) else 'failed',finished=time.time())
    save(output,r)
    return r

if __name__=='__main__':
    result=main();print(result['status'],flush=True)
