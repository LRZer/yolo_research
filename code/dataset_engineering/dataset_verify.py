"""Validate downloaded archives and decoded images without changing original labels."""
import collections
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
import zipfile
import xml.etree.ElementTree as ET
from dataset_acquire import ROOT, MGT, digest, save, rar_file_only
sys.path.insert(0, str(MGT/'vendor'))
import rarfile
if os.name == 'nt':
    rarfile.BSDTAR_TOOL = 'tar'
from PIL import Image

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def unpack(path, dest):
    marker = dest/'_EXTRACTION_COMPLETE.json'
    source_sha = digest(path)
    if marker.exists():
        assert json.loads(marker.read_text())['sha256'] == source_sha
        return
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        archive = zipfile.ZipFile(path)
        members = [(x.filename, x.is_dir(), (x.external_attr >> 16) & 0o170000 == 0o120000, x) for x in archive.infolist()]
    elif path.suffix == '.rar':
        archive = rarfile.RarFile(path)
        members = [(x.filename, x.isdir(), x.is_symlink(), x) for x in archive.infolist()]
    else:
        archive = tarfile.open(path)
        members = [(x.name, x.isdir(), not(x.isdir() or x.isfile()), x) for x in archive.getmembers()]
    try:
        total = 0
        for name, isdir, link, info in members:
            p = PurePosixPath(name.replace('\\', '/'))
            if p.is_absolute() or '..' in p.parts or (p.parts and ':' in p.parts[0]) or link:
                raise ValueError(f'unsafe archive member: {name}')
            total += getattr(info, 'file_size', getattr(info, 'size', 0))
        if total > 50*1024**3:
            raise ValueError('uncompressed archive exceeds safety budget')
        if shutil.disk_usage(dest).free < total + 20*1024**3:
            raise ValueError('insufficient free disk with 20 GiB reserve')
        if os.name == 'nt' and isinstance(archive, rarfile.RarFile):
            # All members have passed the safety audit above; destination must be empty.
            if any(dest.iterdir()):
                raise ValueError('partial RAR extraction exists; inspect before retry')
            rar_file_only(path, dest)
            for name, isdir, link, info in members:
                if not isdir:
                    out = dest.joinpath(*PurePosixPath(name.replace('\\','/')).parts)
                    if not out.is_file() or out.stat().st_size != info.file_size:
                        raise ValueError(f'RAR extracted member missing/size mismatch: {name}')
            save(marker, dict(sha256=source_sha, archive=str(path), entries=len(members), finished=time.time()))
            return
        for name, isdir, link, info in members:
            if not PurePosixPath(name.replace('\\', '/')).parts:
                if isdir:
                    continue
                raise ValueError('empty archive filename')
            if '__MACOSX' in name or Path(name).name.startswith('._'):
                continue
            out = dest.joinpath(*PurePosixPath(name.replace('\\', '/')).parts)
            if isdir:
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                raise ValueError(f'partial extraction exists; inspect before retry: {out}')
            reader = archive.extractfile(info) if isinstance(archive, tarfile.TarFile) else archive.open(info)
            with reader as inp, out.open('xb') as dst:
                shutil.copyfileobj(inp, dst, 1024*1024)
        save(marker, dict(sha256=source_sha, archive=str(path), entries=len(members), finished=time.time()))
    finally:
        archive.close()

def verify(ds, jobs):
    base = ROOT/ds
    result = dict(dataset=ds, started=time.time(), errors=[], warnings=[], annotations=[])
    try:
        checks = []
        for j in jobs:
            f = base/'checks'/(j['name'].replace('/', '_')+'.json')
            c = json.loads(f.read_text())
            if c['status'] != 'download_verified':
                raise ValueError(f'not downloaded/verified: {j["name"]}')
            checks.append(c)
            path = base/'archives'/j['name']
            if path.name.endswith(('.zip', '.rar', '.tar.gz', '.tgz')):
                unpack(path, base/'raw'/path.name)
        result['download_bytes'] = sum(c['bytes'] for c in checks)
        images = sorted(p for p in (base/'raw').rglob('*') if p.is_file() and p.suffix.lower() in EXT)
        index = collections.defaultdict(list)
        sizes = {}
        modes = collections.Counter()
        for k, p in enumerate(images):
            index[p.name].append(p)
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im.load()
                    sizes[p] = im.size
                    modes[im.mode] += 1
            except Exception as e:
                result['errors'].append(f'image decode {p.relative_to(base)}: {e}')
            if k and k % 2000 == 0:
                print(ds, 'decoded', k, '/', len(images), flush=True)
        result.update(image_files=len(images), image_unique_filenames=len(index), image_modes=dict(modes))
        referenced = set()
        jsonfiles = list((base/'raw').rglob('*.json')) + list((base/'archives').rglob('*.json'))
        for p in jsonfiles:
            try:
                data = json.loads(p.read_text(encoding='utf-8-sig'))
            except Exception as e:
                result['errors'].append(f'JSON parse {p}: {e}')
                continue
            if ds == 'HIT-UAV' and isinstance(data, dict) and 'annotation' in data and 'annotations' not in data:
                # The publisher's horizontal JSON uses singular annotation and filename.
                data = dict(data, annotations=data['annotation'])
            if not isinstance(data, dict) or not isinstance(data.get('images'), list) or not isinstance(data.get('annotations'), list):
                continue
            ims = {x['id']: x for x in data['images']}
            cats = {x['id'] for x in data.get('categories', [])}
            if len(ims) != len(data['images']) or len(cats) != len(data.get('categories', [])) or len({a['id'] for a in data['annotations']}) != len(data['annotations']):
                result['errors'].append(f'duplicate image/category/annotation IDs: {p.name}')
            missing, wrongsize, invalid, outside = [], [], [], []
            for row in data['images']:
                name = Path((row.get('file_name') or row['filename']).replace('\\','/')).name
                if ds == 'DUO':
                    split = p.stem.removeprefix('instances_')
                    if split not in {'train', 'test'}:
                        raise ValueError(f'unknown DUO split: {p.name}')
                    expected_path = p.parent.parent/'images'/split/name
                    hits = [expected_path] if expected_path in sizes else []
                    referenced.add(f'{split}/{name}')
                elif ds == 'HIT-UAV':
                    expected_path = p.parent.parent/p.stem/name
                    hits = [expected_path] if expected_path in sizes else []
                    referenced.add(name)
                else:
                    referenced.add(name)
                    hits = index.get(name, [])
                if not hits:
                    missing.append(name)
                elif row.get('width') and row.get('height') and not any(sizes.get(h)==(row['width'],row['height']) for h in hits):
                    wrongsize.append(name)
            for a in data['annotations']:
                if a['image_id'] not in ims or (cats and a.get('category_id') not in cats):
                    invalid.append(a.get('id'))
                    continue
                b = a.get('bbox')
                if b is None:
                    invalid.append(a.get('id'))
                    continue
                rotated = ds == 'HIT-UAV' and 'rotate_json' in p.parts
                if len(b) != (5 if rotated else 4) or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in b) or b[2]<=0 or b[3]<=0:
                    invalid.append(a.get('id'))
                elif not rotated:
                    im = ims[a['image_id']]
                    if b[0]<-1 or b[1]<-1 or b[0]+b[2]>im.get('width',float('inf'))+1 or b[1]+b[3]>im.get('height',float('inf'))+1:
                        outside.append(a.get('id'))
            r = dict(file=str(p.relative_to(base)), images=len(ims), annotations=len(data['annotations']), categories=len(cats), missing_images=len(missing), dimension_mismatches=len(wrongsize), invalid_annotations=len(invalid), outside_bounds=len(outside), examples=dict(missing=missing[:10],wrongsize=wrongsize[:10],invalid=invalid[:10],outside=outside[:10]))
            result['annotations'].append(r)
            if missing or wrongsize or invalid:
                result['errors'].append(f'annotation consistency failure: {r}')
            if outside:
                result['warnings'].append(f'{p.name}: {len(outside)} boxes extend outside image; originals retained')
        result['coco_referenced_unique_images'] = len(referenced)
        if ds == 'ORCA':
            canonical = json.loads((base/'archives'/'data.json').read_text(encoding='utf-8'))
            canonical_names = {Path(r['file_name']).name for r in canonical['images']}
            labels = collections.Counter(str(a.get('label')) for a in canonical['annotations'])
            extra = sorted(set(index)-canonical_names)
            absent = sorted(canonical_names-set(index))
            unknown = sorted(set(labels)-{'0','1','2'})
            unlabeled_caption = [a for a in canonical['annotations'] if a.get('label') == -1]
            result['orca_metadata_audit'] = dict(
                canonical_images=len(canonical_names), actual_image_files=len(images),
                canonical_annotation_rows=len(canonical['annotations']), categories=len(canonical['categories']),
                caption_label_counts=dict(labels), unreferenced_images=extra, missing_images=absent,
                undocumented_caption_labels=unknown, minus_one_empty_captions=sum(not a.get('caption') for a in unlabeled_caption),
                notes='Label 1 is a negative caption, not a new positive detection target. Caption labels and repeated boxes must not be naively converted into independent positive targets. Originals retained unchanged.')
            if extra:
                result['warnings'].append(f'ORCA canonical data.json does not reference {len(extra)} image(s): {extra[:10]}')
            if unknown:
                result['warnings'].append(f'ORCA caption label values undocumented in publisher README: {unknown}; interpretation requires review')
            if absent:
                result['errors'].append('ORCA canonical image references missing from archive')
        if ds == 'HIT-UAV':
            hit_roots = list((base/'raw').glob('*/suojiashun-*'))
            if len(hit_roots) != 1:
                raise ValueError('unexpected HIT-UAV archive root')
            hit = hit_roots[0]
            reports = []
            valid_classes = {'Person','Car','Bicycle','OtherVehicle','DontCare'}
            splitsets = {}
            for variant in ('normal_json','rotate_json'):
                splitsets[variant] = {}
                for split in ('train','val','test'):
                    payload = json.loads((hit/variant/'annotations'/f'{split}.json').read_text())
                    names = {r['filename'] for r in payload['images']}
                    splitsets[variant][split] = names
                    actual = {p.name for p in (hit/variant/split).glob('*.jpg')}
                    if actual != names:
                        result['errors'].append(f'HIT-UAV {variant}/{split} actual images differ from label index')
                sets = list(splitsets[variant].values())
                if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
                    result['errors'].append(f'HIT-UAV split overlap in {variant}')
            if splitsets['normal_json'] != splitsets['rotate_json']:
                result['errors'].append('HIT-UAV horizontal/rotated split mismatch')
            for variant in ('normal_xml','rotate_xml'):
                xmls = sorted((hit/variant/'Annotations').glob('*.xml'))
                labelnames = set()
                counts = collections.Counter()
                invalid_count = 0
                for xp in xmls:
                    tree = ET.parse(xp).getroot()
                    name = xp.stem+'.jpg'
                    labelnames.add(name)
                    img = hit/variant/'JPEGImages'/name
                    dim = (int(tree.findtext('size/width')), int(tree.findtext('size/height')))
                    if sizes.get(img) != dim or Path(tree.findtext('filename')).stem != xp.stem:
                        result['errors'].append(f'HIT-UAV XML image mismatch {variant}/{xp.name}')
                    for obj in tree.findall('object'):
                        category = obj.findtext('name')
                        counts[category] += 1
                        if category not in valid_classes:
                            invalid_count += 1
                        box = obj.find('bndbox')
                        if box is not None:
                            vals = [float(box.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
                            valid = all(map(math.isfinite,vals)) and vals[2]>vals[0] and vals[3]>vals[1]
                        else:
                            box = obj.find('robndbox')
                            if box is None:
                                invalid_count += 1
                                continue
                            vals = [float(box.findtext(k)) for k in ('cx','cy','w','h','angle')]
                            valid = all(map(math.isfinite,vals)) and vals[2]>0 and vals[3]>0
                        if not valid:
                            invalid_count += 1
                actual = {p.name for p in (hit/variant/'JPEGImages').glob('*.jpg')}
                if len(labelnames)!=2898 or labelnames!=actual or labelnames!=referenced:
                    result['errors'].append(f'HIT-UAV {variant} image/XML set mismatch')
                if invalid_count:
                    result['errors'].append(f'HIT-UAV {variant}: {invalid_count} invalid XML objects')
                reports.append(dict(variant=variant,xml_files=len(xmls),classes=dict(counts),invalid_objects=invalid_count))
            result['hit_uav_original_formats'] = reports
            result['hit_uav_splits'] = {k:len(v) for k,v in splitsets['normal_json'].items()}
            result['hit_uav_notes'] = 'Four image copies per source image across JSON/XML horizontal/rotated formats, plus README illustrations. DontCare is an ignore label, not an ordinary target class. Rotated boxes checked for finite coordinates/positive sides, not horizontal bounds.'
        if ds == 'LLVIP':
            vi = {p.stem for p in images if 'visible' in p.parts}
            ir = {p.stem for p in images if 'infrared' in p.parts}
            xmls = list((base/'raw').rglob('*.xml'))
            labels = set()
            objects = 0
            for p in xmls:
                tree = ET.parse(p).getroot()
                labels.add(p.stem)
                objects += len(tree.findall('object'))
                for o in tree.findall('object'):
                    b=o.find('bndbox')
                    if b is None or float(b.findtext('xmax')) <= float(b.findtext('xmin')) or float(b.findtext('ymax')) <= float(b.findtext('ymin')):
                        result['errors'].append(f'invalid XML box {p.name}')
            result['paired_modalities'] = dict(visible=len(vi),infrared=len(ir),xml=len(labels),objects=objects)
            if len(vi)!=15488 or vi!=ir or vi!=labels:
                result['errors'].append('LLVIP paired images/labels differ from expected 15488 pairs')
        elif ds == 'ExDark':
            # Publisher README embeds this illustration; decode it above, but do not
            # count it as a dataset sample. Do not silently exclude any other extras.
            illustration = base/'raw'/'groundtruth.tar.gz'/'ExDark_Annno'/'annotations.png'
            sample_images = [p for p in images if p != illustration]
            sample_names = {p.name for p in sample_images}
            result['exdark_dataset_images'] = len(sample_images)
            result['auxiliary_images'] = [str(p.relative_to(base)) for p in images if p == illustration]
            txts = [p for p in (base/'raw').rglob('*.txt') if p.name.lower().endswith(('.jpg.txt','.png.txt','.jpeg.txt','.bmp.txt'))]
            labelnames = {p.name[:-4] for p in txts}
            box_classes = collections.Counter()
            allowed_classes = {'Bicycle','Boat','Bottle','Bus','Car','Cat','Chair','Cup','Dog','Motorbike','People','Table'}
            for p in txts:
                for line in p.read_text(errors='strict').splitlines():
                    if not line.strip() or line.lstrip().startswith('%'):
                        continue
                    v=line.split()
                    if len(v)<5:
                        result['errors'].append(f'invalid ExDark label line {p.name}')
                        continue
                    try:
                        x,y,w,h=map(float,v[1:5])
                        if w<=0 or h<=0 or not all(map(math.isfinite,[x,y,w,h])):
                            raise ValueError('invalid bbox')
                        if v[0] not in allowed_classes:
                            result['errors'].append(f'unknown ExDark object class {v[0]} in {p.name}')
                        box_classes[v[0]] += 1
                    except ValueError:
                        result['errors'].append(f'invalid ExDark bbox {p.name}')
            result['exdark_label_files'] = len(txts)
            result['exdark_box_classes'] = dict(box_classes)
            result['exdark_total_boxes'] = sum(box_classes.values())
            # One publisher pair uses .JPG for the image and .jpg in its label name.
            # Accept only a one-to-one case-folded mapping, record it explicitly,
            # and never rename the original files (Linux readers need this mapping).
            sample_case = {n.casefold():n for n in sample_names}
            label_case = {n.casefold():n for n in labelnames}
            case_map = {n:sample_case[n.casefold()] for n in labelnames if n not in sample_names and n.casefold() in sample_case}
            result['exdark_label_filename_case_map'] = case_map
            if len(sample_images)!=7363 or len(txts)!=7363 or len(labelnames)!=7363 or len(sample_case)!=7363 or len(label_case)!=7363 or set(label_case)!=set(sample_case):
                result['errors'].append('ExDark images/labels differ from expected 7363')
            splitfile=base/'archives'/'protocol'/'imageclasslist.txt'
            rows=[line.split() for line in splitfile.read_text(encoding='utf-8-sig').splitlines()[1:] if line.strip()]
            splitnames=[r[0] for r in rows]
            splitcounts=collections.Counter(r[4] for r in rows if len(r)==5)
            result['official_split_counts']={ {'1':'train','2':'val','3':'test'}.get(k,k):v for k,v in splitcounts.items() }
            if any(len(r)!=5 or r[4] not in {'1','2','3'} for r in rows):
                result['errors'].append('invalid author split-list row')
            if len(splitnames)!=7363 or len(set(splitnames))!=7363 or set(splitnames)!=sample_names:
                result['errors'].append('ExDark author split list does not match image files')
        else:
            # ORCA archive entries include a directory and an unreferenced image;
            # inspect canonical metadata explicitly above instead of treating entries as images.
            expected = {'DUO':7782,'AP-10K':10015,'HIT-UAV':2898,'SeaClear':8610}.get(ds)
            if not result['annotations']:
                result['errors'].append('no COCO annotations found')
            if expected and len(referenced)!=expected:
                result['warnings'].append(f'published image count {expected}; observed unique referenced image keys {len(referenced)}; requires review')
        if not images:
            result['errors'].append('no decoded image files found')
        result['status'] = 'needs_review' if result['errors'] or result['warnings'] else 'verified_usable_original_format'
    except Exception as e:
        result['errors'].append(str(e))
        result['status'] = 'incomplete_or_failed'
    result['finished']=time.time()
    save(base/'VALIDATION.json', result)
    lines = [f'# {ds} 数据完整性检查', '', f'状态：{result["status"]}', '', '原始图片/标注保留，不做标签修正、不重分集合。未转换为YOLO格式，未训练。', '', '```json', json.dumps(result,ensure_ascii=False,indent=2), '```']
    (base/'VALIDATION.md').write_text('\n'.join(lines),encoding='utf-8')
    print(ds,result['status'],flush=True)
    return result

if __name__=='__main__':
    jobs=json.loads((MGT/'download_manifest.json').read_text())
    datasets=list(dict.fromkeys(j['dataset'] for j in jobs))
    if len(sys.argv)>1:
        datasets=sys.argv[1:]
    results=[]
    for ds in datasets:
        results.append(verify(ds,[j for j in jobs if j['dataset']==ds]))
    save(MGT/'validation_results.json',results)
