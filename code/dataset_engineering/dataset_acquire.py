"""Scoped, resumable public dataset acquisition; no training or system changes."""
import concurrent.futures as cf
from collections import defaultdict
import hashlib
try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import tempfile
import zipfile
import requests

def download_in_ranges(job, part):
    """Append only fully received, correctly ranged chunks to a task-owned partial file."""
    total=job.get('size')
    if not isinstance(total,int) or total <= 0:
        raise ValueError('segmented transfer requires verified expected size')
    segment=part.with_name(part.name+'.segment')
    deadline=time.monotonic()+6*3600
    while (part.stat().st_size if part.exists() else 0) < total:
        if time.monotonic()>deadline:
            raise TimeoutError('segmented transfer time budget reached; prefix retained')
        start=part.stat().st_size if part.exists() else 0
        end=min(total-1,start+8*1024*1024-1)
        wanted=f'bytes {start}-{end}/{total}'
        last=None
        for attempt in range(5):
            try:
                with requests.get(job['url'],headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,45)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=wanted:
                        raise ValueError(f'range rejected: {r.status_code} {r.headers.get("Content-Range")} != {wanted}')
                    with segment.open('wb') as out:
                        for block in r.iter_content(256*1024):
                            out.write(block)
                if segment.stat().st_size!=end-start+1:
                    raise ValueError('incomplete range payload')
                if (part.stat().st_size if part.exists() else 0)!=start:
                    raise RuntimeError('partial prefix changed unexpectedly')
                with part.open('ab') as out, segment.open('rb') as inp:
                    for block in iter(lambda:inp.read(1024*1024),b''):
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
                # Completed segment is disposable; the validated bytes are now in the retained prefix.
                segment.unlink()
                print(job['dataset'],'range_progress',end+1,'/',total,flush=True)
                break
            except Exception as e:
                last=e
                if attempt==4:
                    raise RuntimeError(f'range transfer failed; prefix retained: {last}') from e
                time.sleep(5*(attempt+1))

ROOT = Path(os.environ.get('DATASET_ROOT', '/home/ubuntu/rz/datasets'))
MGT = ROOT / '_management'
MGT.mkdir(exist_ok=True)

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def get(url):
    r = requests.get(url, timeout=(15, 45))
    r.raise_for_status()
    return r.json()

def digest(path, alg='sha256'):
    h = hashlib.new(alg)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def rar_file_only(path, destination=None):
    """Work around Windows bsdtar attempting to decompress a zero-byte RAR directory.

    Read every regular member in one pass. Extraction callers must first audit paths,
    links, disk space, and absence of existing destination files.
    """
    sys.path.insert(0, str(MGT/'vendor'))
    import rarfile
    with rarfile.RarFile(path) as archive:
        names = [i.filename for i in archive.infolist() if not i.isdir()]
    if any('\n' in n or '\r' in n for n in names):
        raise ValueError('newline in archive member name')
    with tempfile.TemporaryDirectory(prefix='rar-check-', dir=MGT) as temp:
        selection = Path(temp)/'members.txt'
        selection.write_text('\n'.join(names)+'\n', encoding='utf-8')
        cmd = ['tar', '-xf' if destination else '-xOf', str(path), '-T', str(selection)]
        if destination:
            cmd += ['-C', str(destination)]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode:
        raise ValueError(f'RAR regular-member check failed: {proc.stderr[-1500:]}')
    return len(names)

def fetch(job):
    lockdir = ROOT/job['dataset']/'checks'
    lockdir.mkdir(parents=True, exist_ok=True)
    with (lockdir/(job['name'].replace('/', '_')+'.lock')).open('a') as lock:
        try:
            if fcntl:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except (BlockingIOError, OSError):
            return dict(job, status='already_running')
        return fetch_locked(job)

def fetch_locked(job):
    ds, name, url = job['dataset'], job['name'], job['url']
    target = ROOT / ds / 'archives' / name
    if not target.resolve().is_relative_to((ROOT/ds/'archives').resolve()):
        raise ValueError('download path escapes dataset archive directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    result = dict(job, status='running', started=time.time())
    statusfile = ROOT / ds / 'checks' / (name.replace('/', '_') + '.json')
    save(statusfile, result)
    try:
        if not target.exists():
            if job.get('range_chunked'):
                download_in_ranges(job,part)
            cmd = ['curl', '--http1.1', '-fL', '--connect-timeout', '20',
                   '--max-time', '300', '--speed-limit', '1024', '--speed-time', '60',
                   '--limit-rate', '12M', '-C', '-', '-o', str(part), url]
            with (ROOT / ds / 'checks' / (name.replace('/', '_')+'.download.log')).open('a') as log:
                for attempt in range(48):
                    if job.get('size') is not None and part.exists() and part.stat().st_size == job['size']:
                        proc=subprocess.CompletedProcess(cmd,0)
                        break
                    log.write(f'\nHTTP/1.1 resume attempt {attempt+1}; retained bytes={part.stat().st_size if part.exists() else 0}\n')
                    log.flush()
                    proc = subprocess.run(cmd, stdout=log, stderr=log)
                    # Even a final interrupted transfer may already have all expected bytes.
                    if proc.returncode == 0 or (job.get('size') is not None and part.exists() and part.stat().st_size == job['size']):
                        break
                    # Curl refuses to append on code 33; a later valid Range response can safely resume.
                    if proc.returncode not in {5, 6, 7, 18, 28, 33, 35, 52, 55, 56, 92}:
                        break
                    time.sleep(min(30, 5+attempt))
            if proc.returncode:
                if not (job.get('size') is not None and part.exists() and part.stat().st_size == job['size']):
                    raise RuntimeError(f'curl exit {proc.returncode}; partial file retained')
            if job.get('size') is not None and part.stat().st_size != job['size']:
                raise ValueError(f'size mismatch {part.stat().st_size} != {job["size"]}')
            with part.open('rb') as f:
                prefix = f.read(256).lstrip().lower()
            if prefix.startswith((b'<!doctype html', b'<html')) and not name.endswith('.html'):
                raise ValueError('HTML returned instead of data')
            if job.get('checksum'):
                alg, value = job['checksum'].split(':', 1)
                if digest(part, alg) != value:
                    raise ValueError('publisher checksum mismatch')
            part.rename(target)
        if job.get('size') is not None and target.stat().st_size != job['size']:
            raise ValueError('existing file size mismatch; retained unchanged')
        if job.get('checksum'):
            alg, value = job['checksum'].split(':', 1)
            if digest(target, alg) != value:
                raise ValueError('existing file checksum mismatch; retained unchanged')
        result.update(bytes=target.stat().st_size, sha256=digest(target))
        if name.endswith('.zip') and not zipfile.is_zipfile(target):
            raise ValueError('ZIP filename does not contain a valid ZIP archive')
        if name.endswith('.json'):
            with target.open(encoding='utf-8-sig') as f:
                json.load(f)
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as z:
                if job.get('zip_password'):
                    z.setpassword(job['zip_password'].encode('utf-8'))
                bad = z.testzip()
                if bad:
                    raise ValueError(f'ZIP CRC failure {bad}')
                result['archive_entries'] = len(z.infolist())
            result['archive_integrity'] = 'ZIP CRC passed'
        elif name.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(target) as t:
                result['archive_entries'] = len(t.getmembers())
            result['archive_integrity'] = 'tar listing passed; publisher checksum verified'
        elif name.endswith('.rar'):
            if os.name == 'nt':
                result['rar_regular_members_tested'] = rar_file_only(target)
                p = subprocess.CompletedProcess([], 0)
            else:
                p = subprocess.run(['unrar', 't', '-idq', str(target)], capture_output=True, text=True)
            if p.returncode:
                raise ValueError(f'RAR test failed: {p.stderr[-1000:]}')
            result['archive_integrity'] = 'RAR test passed'
        result.update(status='download_verified', finished=time.time())
    except Exception as e:
        result.update(status='failed', error=str(e), finished=time.time())
    save(statusfile, result)
    print(ds, name, result['status'], result.get('error', ''), flush=True)
    return result

def catalog():
    jobs = []
    def add(ds, name, url, size=None, checksum=None):
        jobs.append(dict(dataset=ds, name=name, url=url, size=size, checksum=checksum))
    for ds in ['JODD', 'DUO', 'ORCA', 'AP-10K', 'LLVIP', 'HIT-UAV', 'ExDark', 'SeaClear']:
        (ROOT/ds).mkdir(exist_ok=True)
    try:
        meta = get('https://zenodo.org/api/records/7633134')
        save(ROOT/'HIT-UAV'/'source_metadata.json', meta)
        for f in meta['files']:
            add('HIT-UAV', Path(f['key']).name, f['links']['self'], f['size'], f['checksum'])
    except Exception as e:
        print('HIT-UAV metadata', repr(e), flush=True)
    try:
        meta = get('https://data.4tu.nl/v2/articles/4f1dff25-e157-4399-a5d4-478055461689')
        save(ROOT/'SeaClear'/'source_metadata.json', meta)
        for f in meta['files']:
            add('SeaClear', f['name'], f['download_url'], f['size'], 'md5:'+f['computed_md5'])
    except Exception as e:
        print('SeaClear metadata', repr(e), flush=True)
    try:
        meta = get('https://researchdata.um.edu.my/api/datasets/:persistentId/?persistentId=doi:10.22452/RD/JUSQEK')
        save(ROOT/'ExDark'/'source_metadata.json', meta)
        for f in meta['data']['latestVersion']['files']:
            if not f['restricted']:
                v = f['dataFile']
                add('ExDark', f['label'], 'https://researchdata.um.edu.my/api/access/datafile/'+str(v['id']),
                    v['filesize'], v['checksum']['type'].lower()+':'+v['checksum']['value'])
    except Exception as e:
        print('ExDark metadata', repr(e), flush=True)
    try:
        meta = get('https://huggingface.co/api/datasets/WongYukKwan/ORCA')
        revision = meta['sha']
        tree = get(f'https://huggingface.co/api/datasets/WongYukKwan/ORCA/tree/{revision}?recursive=true&limit=1000')
        save(ROOT/'ORCA'/'source_metadata.json', dict(repository=meta, tree=tree))
        for f in tree:
            if f['type'] == 'file':
                checksum = 'sha256:'+f['lfs']['oid'] if f.get('lfs') else None
                add('ORCA', f['path'], f'https://huggingface.co/datasets/WongYukKwan/ORCA/resolve/{revision}/{f["path"]}', f['size'], checksum)
    except Exception as e:
        print('ORCA metadata', repr(e), flush=True)
    drive = [('DUO','DUO.zip','1w-bWevH7jFs7A1bIBlAOvXOxe2OFSHHs'),
             ('AP-10K','ap-10k.zip','1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad'),
             ('LLVIP','LLVIP.zip','1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv')]
    for ds, name, ident in drive:
        add(ds, name, f'https://drive.usercontent.google.com/download?id={ident}&export=download&confirm=t')
    save(MGT/'download_manifest.json', jobs)
    return jobs

if __name__ == '__main__':
    if len(sys.argv) > 1:
        jobs = json.loads(Path(sys.argv[1]).read_text())
    else:
        jobs = catalog()
    groups = defaultdict(list)
    for job in jobs:
        groups[job['dataset']].append(job)
    jobs = [group[i] for i in range(max(map(len, groups.values()))) for group in groups.values() if i < len(group)]
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fetch, jobs))
    save(MGT/'download_results.json', results)
