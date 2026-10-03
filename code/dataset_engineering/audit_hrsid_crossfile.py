"""Compare the three original annotation files; do not rewrite them."""
import collections
import hashlib
import json
import os
from pathlib import Path
import requests

BASE=Path('E:/datasets/HRSID')
ANN=BASE/'raw/HRSID_JPG.rar/HRSID_JPG/annotations'

def normalize(d,full=True):
    names={i['id']:i['file_name'] for i in d['images']}
    fields=None if full else {'bbox','category_id','iscrowd'}
    return collections.Counter(json.dumps(dict({k:v for k,v in a.items() if k not in {'id','image_id'} and (fields is None or k in fields)},file_name=names[a['image_id']]),sort_keys=True) for a in d['annotations'])

def main():
    docs={p.stem:json.loads(p.read_text()) for p in ANN.glob('*.json')}
    result={'scope':'Original JPG archive annotation consistency; no labels modified','files':{k:{'images':len(v['images']),'annotations':len(v['annotations'])} for k,v in docs.items()},'comparison':{}}
    for full in (True,False):
        c={k:normalize(v,full) for k,v in docs.items()}
        union=c['train2017']+c['test2017'];combined=c['train_test2017']
        result['comparison']['full_annotation' if full else 'detection_box']={}
        for key,diff in [('split_only',union-combined),('combined_only',combined-union)]:
            records=[dict(json.loads(k),multiplicity=n) for k,n in diff.items()]
            result['comparison']['full_annotation' if full else 'detection_box'][key]={'count':sum(diff.values()),'affected_images':len({r['file_name'] for r in records}),'records':records}
    os.environ['HTTPS_PROXY']='http://127.0.0.1:7897'
    s=requests.Session()
    try:
        r=s.get('https://api.github.com/repos/chaozhong2010/HRSID/commits/master',timeout=(15,30));r.raise_for_status();commit=r.json()['sha']
        result['official_repository_commit']=commit
        r=s.get(f'https://api.github.com/repos/chaozhong2010/HRSID/contents/annotations?ref={commit}',timeout=(15,30));r.raise_for_status()
        entries={a['name']:a for a in r.json()}
        result['official_comparison']={}
        for p in ANN.glob('*.json'):
            if p.name not in entries:
                result['official_comparison'][p.name]={'available':False};continue
            url=entries[p.name]['download_url'];r=s.get(url,timeout=(15,45));r.raise_for_status()
            remote=hashlib.sha256(r.content).hexdigest();local=hashlib.sha256(p.read_bytes()).hexdigest()
            result['official_comparison'][p.name]={'url':url,'local_sha256':local,'official_sha256':remote,'byte_identical':remote==local,'parsed_equal':json.loads(r.content)==docs[p.stem]}
    except Exception as e:result['official_lookup_error']=str(e)
    report=BASE/'SUPPLEMENTAL_SPLIT_CONSISTENCY.json';report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# HRSID 原始标注文件一致性补充审计','','图片与每份 JSON 的引用/尺寸/框结构检查通过，但三份 JSON 的对象集合并非完全等价。','',
           '- train2017：3642图、11047条标注；test2017：1962图、5922条；两者合计16969条。',
           '- train_test2017：5604图、16951条。合并版不能不经核对就替代独立train/test标签。',
           '- 按图片文件名对齐，忽略重新编号的image_id和annotation id后，完整标注比对：独立split中有22条不在合并版，合并版有4条不在split集合。',
           '- 全部原JSON原样保留。后续实验需明确固定使用哪套标签；本次不替换、不删框、不生成统一标签。',
           '- 这属于原始数据版本/内容一致性问题，不能把压缩包完整性通过解读为所有标签文件互为等价。','',
           '官方仓库：https://github.com/chaozhong2010/HRSID','',
           '详细逐条差异、检测框独立对比、官方仓库固定commit核对见同目录 SUPPLEMENTAL_SPLIT_CONSISTENCY.json。']
    if 'official_comparison' in result:
        lines+=['','官方文件比对：'+json.dumps(result['official_comparison'],ensure_ascii=False)]
    else:lines+=['','官方逐文件核对尚未完成：'+result.get('official_lookup_error','unknown')]
    (BASE/'SUPPLEMENTAL_SPLIT_CONSISTENCY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'comparison':{k:{kk:{f:v for f,v in vv.items() if f!='records'} for kk,vv in val.items()} for k,val in result['comparison'].items()},'official':result.get('official_comparison'),'error':result.get('official_lookup_error')},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
