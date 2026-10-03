"""Read-only archive content audits; writes reports, never changes source labels."""
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image

ROOT=Path('E:/datasets')
os.environ['DATASET_ROOT']=str(ROOT)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def report(ds,r):
    r.update(finished=time.time(),status='failed' if r['errors'] else ('file_integrity_passed_annotation_review_required' if r.get('annotation_issues') else ('verified_with_usage_notes' if r['warnings'] else 'verified_usable_original_format')))
    p=ROOT/ds/'MANUAL_VALIDATION.json'
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
    (ROOT/ds/'MANUAL_VALIDATION.md').write_text('# '+ds+' — 原始数据验收\n\n原始标签未改写，未转换 YOLO。\n\n```json\n'+json.dumps(r,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    print(ds,r['status'],r['errors'][:10],flush=True)

def zip_contents(path):
    sizes={};labels={};other={}
    with zipfile.ZipFile(path) as z:
        seen=set()
        for k,i in enumerate(z.infolist()):
            name=i.filename
            p=PurePosixPath(name.replace('\\','/'))
            if p.is_absolute() or '..' in p.parts or (p.parts and ':' in p.parts[0]) or (i.external_attr>>16)&0o170000==0o120000:
                raise ValueError('unsafe member '+name)
            if i.is_dir():continue
            if name in seen:raise ValueError('duplicate member '+name)
            seen.add(name)
            b=z.read(i)  # Whole-member read also checks the ZIP CRC.
            if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}:
                with Image.open(io.BytesIO(b)) as im: im.verify()
                with Image.open(io.BytesIO(b)) as im: im.load();sizes[name]=im.size
            elif p.suffix.lower() in {'.xml','.txt','.json'}:labels[name]=b
            else:other[name]=len(b)
            if k and k%2000==0: print(path.name,'CRC/read/decode',k,flush=True)
    return sizes,labels,other

def llvip():
    p=ROOT/'LLVIP/LLVIP/LLVIP.zip'
    r=dict(dataset='LLVIP',started=time.time(),errors=[],warnings=[],archives=[])
    try:
        sizes,labels,other=zip_contents(p)
        groups=collections.defaultdict(dict)
        for name,size in sizes.items():
            parts=PurePosixPath(name).parts
            if len(parts)!=4 or parts[1] not in {'visible','infrared'} or parts[2] not in {'train','test'}:
                raise ValueError('unknown LLVIP image path '+name)
            groups[(parts[1],parts[2])][PurePosixPath(name).stem]=size
        splitcounts={}
        names=set();objects=0;outside=0;classes=collections.Counter();geometry=[]
        for split in ['train','test']:
            vis=groups[('visible',split)];ir=groups[('infrared',split)]
            if vis!=ir:raise ValueError('modality pair names/dimensions differ in '+split)
            if names & set(vis):raise ValueError('train/test filename overlap')
            names.update(vis);splitcounts[split]=len(vis)
        xmls={PurePosixPath(n).stem:b for n,b in labels.items() if n.endswith('.xml')}
        if len(names)!=15488 or set(xmls)!=names or len(sizes)!=30976: raise ValueError('paired image/XML counts differ')
        dims={k:v for group in groups.values() for k,v in group.items()}
        for name,b in xmls.items():
            doc=ET.fromstring(b)
            w=int(doc.findtext('size/width'));h=int(doc.findtext('size/height'))
            if dims[name]!=(w,h):raise ValueError('XML dimensions differ '+name)
            for obj in doc.findall('object'):
                cls=obj.findtext('name');classes[cls]+=1
                box=obj.find('bndbox'); x1,y1,x2,y2=[float(box.findtext(key)) for key in ['xmin','ymin','xmax','ymax']]
                if not all(map(math.isfinite,[x1,y1,x2,y2])):raise ValueError('nonfinite XML box '+name)
                if x2<=x1 or y2<=y1:geometry.append(dict(image=name,box=[x1,y1,x2,y2],issue='nonpositive continuous xyxy extent; inclusive pixel convention may differ; original retained'))
                if x1<0 or y1<0 or x2>w or y2>h:outside+=1
                objects+=1
        if set(classes)!={'person'}:raise ValueError('unexpected LLVIP classes')
        r.update(image_files=len(sizes),paired_images=len(names),xml_files=len(xmls),splits=splitcounts,objects=objects,classes=dict(classes),out_of_bounds=outside)
        if geometry:
            r['annotation_issues']=geometry
            r['warnings'].append(f'{len(geometry)} original XML boxes have nonpositive continuous-coordinate extent; coordinate convention/reader policy requires review before training. File integrity is separate from annotation usability.')
        if outside:r['warnings'].append(f'{outside} original boxes extend outside image; preserved')
        r['version_note']='User-supplied archive. All member paths/sizes/CRCs match the other user-downloaded LLVIP.zip; this is not independent proof of the author 2023 label revision.'
        r['archives']=[dict(local=str(p),name='LLVIP.zip',bytes=p.stat().st_size,sha256=sha(p))]
    except Exception as e:r['errors'].append(str(e))
    report('LLVIP',r)

def visdrone():
    ds='VisDrone2019-DET';r=dict(dataset=ds,started=time.time(),errors=[],warnings=[],archives=[],splits={})
    try:
        requested=[('train',6471),('val',548),('test-dev',1610)]
        if (ROOT/ds/f'{ds}-test-challenge.zip').exists(): requested.append(('test-challenge',1580))
        for split,expected in requested:
            p=ROOT/ds/f'{ds}-{split}.zip';sizes,labels,other=zip_contents(p)
            ims={PurePosixPath(n).stem:sz for n,sz in sizes.items()}
            anns={PurePosixPath(n).stem:b for n,b in labels.items() if 'annotations' in PurePosixPath(n).parts and n.endswith('.txt')}
            if len(ims)!=expected or len(sizes)!=expected:raise ValueError('wrong count '+split)
            if split=='test-challenge':
                if anns:raise ValueError('unexpected challenge labels; inspect version')
                r['warnings'].append('test-challenge contains 1580 images and no annotations: prediction-only, not test-dev; labeled test-dev is missing')
            elif set(anns)!=set(ims):raise ValueError('image/annotation mismatch '+split)
            cats=collections.Counter();outside=0;ignored=0;geometry=[]
            for stem,b in anns.items():
                width,height=ims[stem]
                for row in b.decode('utf-8-sig').splitlines():
                    if not row.strip():continue
                    vals=[float(x) for x in row.strip().rstrip(',').split(',')]
                    if len(vals)!=8 or not all(map(math.isfinite,vals)):raise ValueError('invalid row '+stem)
                    x,y,w,h,score,cat,trunc,occ=vals
                    if cat!=int(cat) or not 0<=cat<=11 or score not in (0,1) or trunc not in (0,1,2) or occ not in (0,1,2):raise ValueError('invalid annotation '+stem)
                    if w<=0 or h<=0:geometry.append(dict(image=stem,row=vals,ignored_or_other=score==0 or cat in (0,11)))
                    if x<0 or y<0 or x+w>width or y+h>height:outside+=1
                    ignored+=int(score==0 or cat in (0,11));cats[int(cat)]+=1
            r['splits'][split]=dict(images=len(ims),annotations=len(anns),objects=sum(cats.values()),class_counts=dict(cats),ignored_or_other_objects=ignored,boxes_outside_bounds=outside)
            if geometry:
                r.setdefault('annotation_issues',[]).append(dict(split=split,nonpositive_boxes=geometry))
                r['warnings'].append(f'{split}: {len(geometry)} original zero/negative-extent rows, {sum(x["ignored_or_other"] for x in geometry)} marked ignored/other. Original rows retained; training reader must follow ignore protocol, not silently treat as positive boxes.')
            if outside:r['warnings'].append(f'{split}: {outside} original boxes outside bounds; retained, not clipped')
            r['archives'].append(dict(local=str(p),name=p.name,bytes=p.stat().st_size,sha256=sha(p)))
    except Exception as e:r['errors'].append(str(e))
    report(ds,r)

def hrsid():
    from dataset_verify import unpack
    p=Path('E:/BaiduNetdiskDownload/HRSID_JPG.rar');base=ROOT/'HRSID';dest=base/'raw/HRSID_JPG.rar'
    r=dict(dataset='HRSID',started=time.time(),errors=[],warnings=[],archives=[],splits={})
    try:
        unpack(p,dest)
        sizes={}
        for k,image in enumerate(dest.rglob('*.jpg')):
            if image.name in sizes:raise ValueError('duplicate image basename')
            with Image.open(image) as im:im.verify()
            with Image.open(image) as im:im.load();sizes[image.name]=im.size
            if k and k%2000==0:print('HRSID decoded',k,flush=True)
        if len(sizes)!=5604:raise ValueError('HRSID image count mismatch')
        splitsets={}
        for ann in (dest/'HRSID_JPG/annotations').glob('*.json'):
            d=json.loads(ann.read_text(encoding='utf-8'));ims={v['id']:v for v in d['images']};cats={c['id'] for c in d['categories']}
            if len(ims)!=len(d['images']) or len({a['id'] for a in d['annotations']})!=len(d['annotations']):raise ValueError('duplicate annotation/image IDs')
            splitsets[ann.stem]={v['file_name'] for v in ims.values()}
            for v in ims.values():
                if sizes.get(v['file_name'])!=(v['width'],v['height']):raise ValueError('missing image or wrong dimensions '+v['file_name'])
            outside=0
            for a in d['annotations']:
                if a['image_id'] not in ims or a['category_id'] not in cats:raise ValueError('invalid annotation reference')
                x,y,w,h=a['bbox'];image=ims[a['image_id']]
                if not all(map(math.isfinite,[x,y,w,h])) or w<=0 or h<=0:raise ValueError('invalid box')
                if x<0 or y<0 or x+w>image['width'] or y+h>image['height']:outside+=1
            r['splits'][ann.stem]=dict(images=len(ims),objects=len(d['annotations']),categories=d['categories'],boxes_outside_bounds=outside)
            if outside:r['warnings'].append(f'{ann.name}: {outside} original boxes outside image bounds')
        if splitsets['train2017'] & splitsets['test2017'] or splitsets['train2017']|splitsets['test2017']!=set(sizes) or splitsets['train_test2017']!=set(sizes):raise ValueError('HRSID splits inconsistent')
        r.update(image_files=len(sizes),archives=[dict(local=str(p),name=p.name,bytes=p.stat().st_size,sha256=sha(p))])
    except Exception as e:r['errors'].append(str(e))
    report('HRSID',r)

if __name__=='__main__':
    {'LLVIP':llvip,'HRSID':hrsid,'VisDrone2019-DET':visdrone}[sys.argv[1]]()
