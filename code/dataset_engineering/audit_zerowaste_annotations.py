"""Verify original ZeroWaste-f COCO and mask pairing after full archive decoding."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import time
from PIL import Image

BASE=Path('E:/datasets/ZeroWaste-f')
RAW=BASE/'raw/zerowaste-f-final.zip'

def save(result):
    p=BASE/'MANUAL_VALIDATION.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
    (BASE/'MANUAL_VALIDATION.md').write_text('# ZeroWaste-f 原始数据验收\n\n原始包、COCO标签、语义掩码原样保留；未转YOLO、未训练。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')

def main():
    deadline=time.monotonic()+7200
    while True:
        v=json.loads((BASE/'EXTRACTION_VALIDATION.json').read_text(encoding='utf-8'))
        if v['status']=='archive_crc_and_image_decode_passed_annotation_audit_pending':break
        if v['status']=='failed':raise RuntimeError('Extraction failed: '+str(v['errors']))
        if time.monotonic()>deadline:raise TimeoutError('Extraction audit not ready')
        time.sleep(15)
    sizes=json.loads((BASE/'IMAGE_DIMENSIONS.json').read_text())
    r=dict(dataset='ZeroWaste-f',started=time.time(),errors=[],warnings=[],splits={},archive_integrity=v,archives=[])
    try:
        catdefs=None;splitnames={};all_masks=collections.Counter()
        for split,expected in [('train',3002),('val',572),('test',929)]:
            folder=RAW/'splits_final_deblurred'/split
            d=json.loads((folder/'labels.json').read_text())
            ims={i['id']:i for i in d['images']};cats={c['id']:c['name'] for c in d['categories']}
            if len(ims)!=len(d['images']) or len(cats)!=len(d['categories']) or len({a['id'] for a in d['annotations']})!=len(d['annotations']):raise ValueError('duplicate COCO IDs '+split)
            if catdefs is None:catdefs=cats
            if catdefs!=cats:raise ValueError('category definitions differ')
            if len(ims)!=expected:raise ValueError('unexpected split count')
            actual={p.name for p in (folder/'data').iterdir() if p.is_file()}
            names={i['file_name'] for i in ims.values()};splitnames[split]=names
            if names!=actual or len(names)!=len(ims):raise ValueError('COCO/data filename mismatch '+split)
            masks={p.stem:p for p in (folder/'sem_seg').iterdir() if p.is_file()}
            if len(masks)!=expected or set(masks)!={Path(n).stem for n in names}:raise ValueError('mask/image mismatch '+split)
            mask_values=collections.Counter()
            for image in ims.values():
                p=folder/'data'/image['file_name'];mask=masks[Path(image['file_name']).stem]
                dim=[image['width'],image['height']]
                if sizes.get(p.relative_to(RAW).as_posix())!=dim or sizes.get(mask.relative_to(RAW).as_posix())!=dim:raise ValueError('COCO/mask/image dimensions differ')
                with Image.open(mask) as im:
                    palette=im.getcolors(maxcolors=256)
                    if palette is None:raise ValueError('mask has more than 256 values')
                    for count,value in palette:mask_values[str(value)]+=count
            counts=collections.Counter();outside=[];invalid_polygons=[]
            for a in d['annotations']:
                if a['image_id'] not in ims or a['category_id'] not in cats:raise ValueError('invalid COCO reference')
                b=a['bbox']
                if len(b)!=4 or not all(map(math.isfinite,b)) or b[2]<=0 or b[3]<=0:raise ValueError('invalid bbox '+str(a['id']))
                im=ims[a['image_id']];x,y,w,h=b
                if x<-1e-4 or y<-1e-4 or x+w>im['width']+1e-4 or y+h>im['height']+1e-4:outside.append(a['id'])
                polygons=a.get('segmentation')
                if not isinstance(polygons,list) or not polygons:invalid_polygons.append(a['id'])
                else:
                    for poly in polygons:
                        if len(poly)<6 or len(poly)%2 or not all(map(math.isfinite,poly)):invalid_polygons.append(a['id']);break
                if not math.isfinite(a['area']) or a['area']<=0:raise ValueError('invalid segmentation area')
                counts[cats[a['category_id']]]+=1
            if outside:r['warnings'].append(f'{split}: {len(outside)} original boxes outside bounds')
            if invalid_polygons:r['warnings'].append(f'{split}: {len(invalid_polygons)} nonstandard/invalid polygon records; originals retained')
            r['splits'][split]=dict(images=len(ims),masks=len(masks),annotations=len(d['annotations']),class_counts=dict(counts),outside_bbox_ids=outside,polygon_review_ids=invalid_polygons,mask_pixel_values=dict(mask_values))
            all_masks.update(mask_values)
            print('ZeroWaste annotation/mask audit',split,'images',len(ims),'objects',len(d['annotations']),flush=True)
        r['cross_split_filename_checks']=[]
        for a,b in [('train','val'),('train','test'),('val','test')]:
            for name in sorted(splitnames[a]&splitnames[b]):
                paths=[RAW/'splits_final_deblurred'/split/'data'/name for split in (a,b)]
                file_hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
                pixel_hashes=[]
                for p in paths:
                    with Image.open(p) as im:pixel_hashes.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
                r['cross_split_filename_checks'].append(dict(splits=[a,b],filename=name,file_sha256=file_hashes,rgb_pixel_sha256=pixel_hashes,pixels_identical=pixel_hashes[0]==pixel_hashes[1]))
                if pixel_hashes[0]==pixel_hashes[1]:r['warnings'].append(f'Exact decoded duplicate across {a}/{b}: {name}; original split preserved, leakage review needed')
        if r['cross_split_filename_checks']:r['warnings'].append('Three cross-split filename collisions were checked by encoded-file and decoded-RGB hashes; readers must use split-qualified paths. This check does not establish independence of nearby video frames.')
        r.update(image_files=4503,mask_files=4503,total_decoded_image_files=v['images_decoded'],categories=catdefs,total_annotations=sum(s['annotations'] for s in r['splits'].values()),mask_pixel_values=dict(all_masks))
        if v['images_decoded']!=9006:raise ValueError('full decode count differs from paired image/mask count')
        r['archives']=[dict(local=str(BASE/'archives/zerowaste-f-final.zip'),name='zerowaste-f-final.zip',bytes=v['bytes'],sha256=v['sha256'],zip_password='UP#1VuX409z4')]
        r['status']='verified_with_usage_notes' if r['warnings'] else 'verified_usable_original_format'
    except Exception as e:r['errors'].append(str(e));r['status']='failed'
    r['finished']=time.time();save(r)
    print(r['status'],r['errors'],r['warnings'],flush=True)

if __name__=='__main__':main()
