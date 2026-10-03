"""Produce an honest progress MD snapshot from frozen manifest and transfer records."""
import collections
import datetime
import json
from pathlib import Path

BASE=Path('E:/datasets')
MGT=BASE/'_management'
m=json.loads((MGT/'UPLOAD_MANIFEST_20260921.json').read_text(encoding='utf-8'))
s=json.loads((MGT/'UPLOAD_STATE_20260921.json').read_text(encoding='utf-8'))
groups=collections.defaultdict(list)
for job in m['files']:groups[job['dataset']].append(job)
lines=['# 数据集上传进展快照','',f'生成时间：{datetime.datetime.now().isoformat(timespec="seconds")}','',
       '目标：`gpu8-team:/home/ubuntu/rz/object_detection_dataset_test`。原始包保留，解压文件位于各数据集 raw/ 下。','',
       '此表依据逐文件传输记录，不替代最后的远端独立全量验收；不保证进程当前仍在线。','',
       '| 数据集 | 完成文件/清单文件 | 原始包及说明 GB | 验收说明 |','|---|---:|---:|---|']
notes={'HIT-UAV':'原格式检查通过，DontCare保留为忽略标记；多格式图片副本不重复算独立样本',
       'DUO':'原格式检查通过；train/test同名文件按split区分',
       'AP-10K':'仅标注主包；姿态标签不等于穷尽性检测标注',
       'SeaClear':'原格式检查通过',
       'ExDark':'保留官方划分和1处图片扩展名大小写映射',
       'ORCA':'文件完整；caption负描述/小框label=-1语义警告及1张未引用图保留',
       'HRSID':'文件完整；split与合并JSON对象集合不等价，见SUPPLEMENTAL审计',
       'LLVIP':'文件完整；5个等横坐标框及标签版本说明保留',
       'VisDrone2019-DET':'train/val/test-dev；3条原始零高度框需读取策略审阅',
       'ZeroWaste-f':'见MANUAL_VALIDATION；保留原COCO和语义掩码'}
for ds,jobs in groups.items():
    done=sum(s['results'].get(j['remote'],{}).get('status')=='complete' and s['results'][j['remote']].get('sha256')==j['sha256'] for j in jobs)
    lines.append(f'| {ds} | {done}/{len(jobs)} | {sum(j["bytes"] for j in jobs)/1e9:.3f} | {notes.get(ds,"")} |')
lines+=['',f'- 最近状态：{s["status"]}；文件：{s.get("current")}；阶段：{s.get("phase")}。',
        '- 尚待加入清单：'+(', '.join(m.get('pending',[])) or '无')+'。',
        '- JODD无实际数据包，未上传；不能纳入成功计数。',
        '- 未修改原始标签，未转换YOLO，未训练/使用GPU，未改共享配置。','']
text='\n'.join(lines)
(MGT/'UPLOAD_PROGRESS_20260921.md').write_text(text,encoding='utf-8')
(Path(__file__).resolve().parents[1]/'outputs/yolo26_research/DATASET_UPLOAD_PROGRESS_2026-09-21.md').write_text(text,encoding='utf-8')
print(text,flush=True)
