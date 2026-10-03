from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(r"C:\Users\Lenovo\Documents\Codex\2026-09-17\referenced-chatgpt-conversation-this-is-an")
OUT = ROOT / "outputs" / "yolo26_research"
REPORT = OUT / "COMPREHENSIVE_DATASET_AND_YOLOV8N_BASELINE_REPORT_2026-09-22.md"
CLASS_CSV = OUT / "COMPREHENSIVE_CLASS_COUNTS_2026-09-22.csv"
DIM_CSV = OUT / "COMPREHENSIVE_IMAGE_DIMENSIONS_2026-09-22.csv"

DATASETS = [
    "AP-10K",
    "DUO",
    "ExDark",
    "HIT-UAV",
    "HRSID-JPG",
    "LLVIP-infrared",
    "LLVIP-visible",
    "ORCA-class-level-seen",
    "SeaClear",
    "VisDrone2019-DET",
    "ZeroWaste-f",
]

META = {
    "AP-10K": {
        "domain": "野生动物；RGB；原主任务为动物姿态估计",
        "annotation": "COCO式关键点/实例框；本报告仅使用检测框",
        "split": "官方 split1；原 train+val 合并为 train，官方 test 用作每轮评估",
        "processing": "未改原图；按官方框转为YOLO。检测框是否穷尽所有可见动物仍是姿态来源数据的使用边界。",
        "download_gb": "3.403",
        "server_gb": "6.852",
    },
    "DUO": {
        "domain": "水下机器人采捕；RGB",
        "annotation": "水平检测框；海参、海胆、扇贝、海星4类",
        "split": "官方 train/test；没有独立val",
        "processing": "裁剪953个越界框；1个框裁剪后无正面积而跳过；原始74,515框保留在源数据中。",
        "download_gb": "2.940",
        "server_gb": "6.087",
    },
    "ExDark": {
        "domain": "低照度自然场景；RGB",
        "annotation": "12类水平检测框",
        "split": "作者 train+val 合并为 train；作者 test 用作每轮评估",
        "processing": "裁剪50个越界框；显式处理1处图片/标签扩展名大小写差异；说明插图不作为样本。",
        "download_gb": "1.518",
        "server_gb": "3.107",
    },
    "HIT-UAV": {
        "domain": "无人机热红外目标检测",
        "annotation": "水平框分支；4个有效类别，另有DontCare",
        "split": "原 train+val 合并为 train；原 test 用作每轮评估",
        "processing": "排除148个DontCare框，不将其当作第5类；不使用旋转框分支。",
        "download_gb": "0.815",
        "server_gb": "1.645",
    },
    "HRSID-JPG": {
        "domain": "高分辨率SAR船舶检测",
        "annotation": "JPG版本、单类水平框",
        "split": "作者分开的train/test JSON；test用作每轮评估",
        "processing": "使用分split JSON的16,969框；不使用与其对象集合不一致的合并JSON（16,951框），也不混用PNG版。",
        "download_gb": "0.609",
        "server_gb": "1.268",
    },
    "LLVIP-infrared": {
        "domain": "低照度配对行人；热红外分支",
        "annotation": "VOC水平框，单类person；与可见光逐图配对并共享标签",
        "split": "作者 train/test；test用作每轮评估",
        "processing": "按Pascal VOC闭区间坐标转换；保留5个一像素宽框。与visible共享下载包和场景，但使用独立图像训练。",
        "download_gb": "4.008（与visible共享）",
        "server_gb": "8.063（与visible共享）",
    },
    "LLVIP-visible": {
        "domain": "低照度配对行人；可见光RGB分支",
        "annotation": "VOC水平框，单类person；与红外逐图配对并共享标签",
        "split": "作者 train/test；test用作每轮评估",
        "processing": "按Pascal VOC闭区间坐标转换；保留5个一像素宽框。与infrared共享下载包和场景，但使用独立图像训练。",
        "download_gb": "4.008（与infrared共享）",
        "server_gb": "8.063（与infrared共享）",
    },
    "ORCA-class-level-seen": {
        "domain": "水下/海洋物种识别；RGB",
        "annotation": "主JSON中的水平框；当前映射到24个class-level已见类",
        "split": "作者class-level seen协议；test用作每轮评估",
        "processing": "完整源包14,646图，主JSON引用14,645图/42,217标注；当前协议使用12,034图、36,274个去重检测框，移除357个派生重复框。caption正负描述不作为检测类别。",
        "download_gb": "1.098",
        "server_gb": "2.104",
    },
    "SeaClear": {
        "domain": "ROV水下垃圾与海洋环境；RGB",
        "annotation": "40类水平框/实例多边形；本报告使用水平框",
        "split": "无直接采用的固定检测划分；按SHA256稳定排序做80/20 train/test；test用作每轮评估",
        "processing": "移除6个派生完全重复框；该随机稳定划分未按地点/相机分组，因此不能作为严格跨域泛化协议。",
        "download_gb": "1.712",
        "server_gb": "3.630",
    },
    "VisDrone2019-DET": {
        "domain": "无人机航拍城市交通；RGB；静态图像DET任务",
        "annotation": "10个有效类别水平框，另有ignore/others",
        "split": "原train+val合并为train；带标签test-dev用作每轮评估",
        "processing": "跳过14,200条ignore/others、1个非正面积有效类框；移除4个派生重复框。未使用challenge无公开GT版本。",
        "download_gb": "1.943",
        "server_gb": "3.894",
    },
    "ZeroWaste-f": {
        "domain": "工业传送带垃圾分拣；RGB",
        "annotation": "4类实例分割/COCO框；本报告使用实例外接水平框",
        "split": "原train+val合并为train；原test用作每轮评估",
        "processing": "裁剪138个越界框；移除22个派生重复框；4,503张语义掩码保留在源数据但未作为检测输入。",
        "download_gb": "7.518",
        "server_gb": "15.052",
    },
}


REMOTE_AUDIT = r'''
from pathlib import Path
from collections import Counter
from PIL import Image
import csv, json, numpy as np, yaml

base=Path('/home/ubuntu/rz/object_detection_dataset_test/processed_yolo')
runs=Path('/home/ubuntu/rz/object_detection_dataset_test/runs/yolov8n_scratch_baselines')
datasets=['AP-10K','DUO','ExDark','HIT-UAV','HRSID-JPG','LLVIP-infrared','LLVIP-visible','ORCA-class-level-seen','SeaClear','VisDrone2019-DET','ZeroWaste-f']
exts={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
out={'datasets':{},'runs':[]}
for ds in datasets:
    root=base/ds
    cfg=yaml.safe_load((root/'dataset.yaml').read_text())
    names=cfg.get('names',{})
    if isinstance(names,list): names={i:v for i,v in enumerate(names)}
    names={int(k):str(v) for k,v in names.items()}
    item={'names':names,'splits':{},'dimensions':Counter()}
    for split in ['train','test']:
        idir=root/'images'/split; ldir=root/'labels'/split
        imgs=sorted(p for p in idir.iterdir() if p.suffix.lower() in exts)
        cls_boxes=Counter(); cls_images=Counter(); dims=Counter(); objs=[]
        bw640=[]; bh640=[]; area640=[]; bad=0; empty=0
        for ip in imgs:
            try:
                with Image.open(ip) as im: W,H=im.size
            except Exception:
                bad+=1; continue
            dims[(W,H)]+=1; item['dimensions'][(W,H)]+=1
            present=set(); n=0; lp=ldir/(ip.stem+'.txt')
            if lp.exists():
                for line in lp.read_text().splitlines():
                    z=line.split()
                    if len(z)<5: continue
                    c=int(float(z[0])); xc,yc,bw,bh=map(float,z[1:5])
                    cls_boxes[c]+=1; present.add(c); n+=1
                    scale=min(640/W,640/H)
                    wp=bw*W*scale; hp=bh*H*scale
                    bw640.append(wp); bh640.append(hp); area640.append(wp*hp)
            if n==0: empty+=1
            objs.append(n)
            for c in present: cls_images[c]+=1
        def pct(a,p):
            return float(np.quantile(np.asarray(a,float),p)) if a else None
        a=np.asarray(area640,float); ms=np.minimum(np.asarray(bw640,float),np.asarray(bh640,float)) if bw640 else np.asarray([])
        item['splits'][split]={
            'images':len(imgs),'boxes':int(sum(cls_boxes.values())),'empty_images':empty,'bad_images':bad,
            'objects_per_image_mean':float(np.mean(objs)),'objects_per_image_median':float(np.median(objs)),
            'objects_per_image_p90':pct(objs,.9),'objects_per_image_max':int(max(objs) if objs else 0),
            'class_boxes':{str(i):int(cls_boxes[i]) for i in names},
            'class_images':{str(i):int(cls_images[i]) for i in names},
            'dimensions':{f'{w}x{h}':int(n) for (w,h),n in sorted(dims.items())},
            'bbox640_width_median':pct(bw640,.5),'bbox640_height_median':pct(bh640,.5),
            'bbox640_area_median':pct(area640,.5),'bbox640_area_p10':pct(area640,.1),
            'bbox640_small_lt32sq_fraction':float(np.mean(a<32**2)) if len(a) else 0.0,
            'bbox640_min_side_lt8_fraction':float(np.mean(ms<8)) if len(ms) else 0.0,
            'bbox640_min_side_lt16_fraction':float(np.mean(ms<16)) if len(ms) else 0.0,
        }
    all_dims=item.pop('dimensions')
    item['all_dimensions']={f'{w}x{h}':int(n) for (w,h),n in sorted(all_dims.items())}
    widths=[int(k.split('x')[0]) for k in item['all_dimensions']]
    heights=[int(k.split('x')[1]) for k in item['all_dimensions']]
    item['dimension_summary']={
        'unique':len(all_dims),'width_min':min(widths),'width_max':max(widths),'height_min':min(heights),'height_max':max(heights),
        'top':sorted(item['all_dimensions'].items(),key=lambda kv:(-kv[1],kv[0]))[:10],
    }
    out['datasets'][ds]=item

for ds in datasets:
    for seed in [2027,2005]:
        p=runs/f'yolov8n_scratch_{ds}_e125_b64_i640_s{seed}'/'results.csv'
        with p.open() as f: rows=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
        r=rows[-1]
        out['runs'].append({
            'dataset':ds,'seed':seed,'time_seconds':float(r['time']),
            'train_box':float(r['train/box_loss']),'train_cls':float(r['train/cls_loss']),'train_dfl':float(r['train/dfl_loss']),
            'val_box':float(r['val/box_loss']),'val_cls':float(r['val/cls_loss']),'val_dfl':float(r['val/dfl_loss']),
        })
print(json.dumps(out,ensure_ascii=False,separators=(',',':')))
'''


def remote_audit() -> dict:
    proc = subprocess.run(
        ["ssh", "gpu8-team", "/home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python", "-"],
        input=REMOTE_AUDIT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def esc(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def fmt(x, n=4) -> str:
    return f"{float(x):.{n}f}"


def pm(mean, std, n=4) -> str:
    return f"{float(mean):.{n}f}±{float(std):.{n}f}"


def md_table(headers: list[str], rows: list[list[object]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["---"] * len(headers)
    out = ["| " + " | ".join(map(esc, headers)) + " |", "|" + "|".join(aligns) + "|"]
    out.extend("| " + " | ".join(esc(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def dimension_top(d: dict) -> str:
    return "；".join(f"{k}：{v:,}" for k, v in d["dimension_summary"]["top"])


def main() -> None:
    audit = remote_audit()
    per_run = read_csv(OUT / "runtime_audit" / "audit_per_run_metrics.csv")
    summary = read_csv(OUT / "runtime_audit" / "audit_two_seed_summary.csv")
    run_loss = {(r["dataset"], int(r["seed"])): r for r in audit["runs"]}

    with CLASS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "class_id", "class_name", "train_images", "train_boxes", "test_images", "test_boxes", "total_images_with_class", "total_boxes"])
        for ds in DATASETS:
            d = audit["datasets"][ds]
            for cid, name in sorted(((int(k), v) for k, v in d["names"].items())):
                tr, te = d["splits"]["train"], d["splits"]["test"]
                w.writerow([ds, cid, name, tr["class_images"][str(cid)], tr["class_boxes"][str(cid)], te["class_images"][str(cid)], te["class_boxes"][str(cid)], tr["class_images"][str(cid)] + te["class_images"][str(cid)], tr["class_boxes"][str(cid)] + te["class_boxes"][str(cid)]])

    with DIM_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "width", "height", "image_count"])
        for ds in DATASETS:
            for wh, n in audit["datasets"][ds]["all_dimensions"].items():
                width, height = wh.split("x")
                w.writerow([ds, width, height, n])

    total_images = sum(audit["datasets"][d]["splits"][s]["images"] for d in DATASETS for s in ("train", "test"))
    total_boxes = sum(audit["datasets"][d]["splits"][s]["boxes"] for d in DATASETS for s in ("train", "test"))

    lines: list[str] = []
    add = lines.append
    add("# 目标检测数据集与YOLOv8n随机初始化基线完整报告")
    add("")
    add("- 报告日期：2026-09-22（Asia/Shanghai）")
    add("- 数据根目录：`/home/ubuntu/rz/object_detection_dataset_test`")
    add("- 可训练派生目录：`/home/ubuntu/rz/object_detection_dataset_test/processed_yolo`")
    add("- 运行结果目录：`/home/ubuntu/rz/object_detection_dataset_test/runs/yolov8n_scratch_baselines`")
    add("- 实验状态：**22/22次训练完成，44/44个best/last checkpoint可读，最终审计verified_complete，errors=0**")
    add("")
    add("## 1. 范围与统计口径")
    add("")
    add(f"本报告覆盖10套原始数据集形成的11个YOLO检测版本（LLVIP拆成红外和可见光两个独立版本），共计{total_images:,}个派生图像条目、{total_boxes:,}个有效YOLO框、22次独立训练。LLVIP两个模态是同一批15,488个配对场景，但物理上有30,976张不同图像；其框描述同一批行人，不能把两模态框数解释为独立实例翻倍。")
    add("")
    add("“每类别图片数”按一张图只要含该类至少一个框就计1；多类别图会在多行重复计数，因此类别图片数不可直接相加为数据集总图数。“框数”是最终送入YOLO的有效、去重后水平框数。尺寸均写为宽×高。")
    add("")
    add("所有`dataset.yaml`的`val`和`test`均指向测试集；每个epoch都在test上评估，最佳epoch也按该test的mAP50-95选择。该协议是用户明确采用的快速基线协议，不存在独立未触碰测试集，因此best结果可能有模型选择偏乐观，不能直接作为严格论文最终数字。报告同时保留固定第125轮last结果。")
    add("")
    add("## 2. 数据集总览")
    add("")
    overview=[]
    for ds in DATASETS:
        d=audit["datasets"][ds]; tr=d["splits"]["train"]; te=d["splits"]["test"]; m=META[ds]
        overview.append([ds,len(d["names"]),f"{tr['images']:,}",f"{te['images']:,}",f"{tr['images']+te['images']:,}",f"{tr['boxes']:,}",f"{te['boxes']:,}",f"{tr['boxes']+te['boxes']:,}",f"{tr['empty_images']+te['empty_images']:,}",m["download_gb"],m["server_gb"]])
    add(md_table(["派生数据集","类数","train图","test图","总图","train框","test框","总框","空标签图","下载材料GB","服务器源目录GB"],overview,["---","---:","---:","---:","---:","---:","---:","---:","---:","---:","---:"]))
    add("")
    add("10套原始下载材料合计25,564,671,681 bytes（25.565GB，23.809GiB）；服务器源目录同时保存原包、解压内容与验收材料，合计约51.700GB。`processed_yolo`约98.7MB，主要是标签、清单和指向原图的绝对软链接，不复制原始图片。LLVIP体积只应计算一次。")
    add("")
    add("## 3. 原图尺寸总览")
    add("")
    dimrows=[]
    for ds in DATASETS:
        d=audit["datasets"][ds]; z=d["dimension_summary"]
        dimrows.append([ds,sum(d["all_dimensions"].values()),z["unique"],f"{z['width_min']}–{z['width_max']}",f"{z['height_min']}–{z['height_max']}",dimension_top(d)])
    add(md_table(["数据集","图片数","不同尺寸数","宽度范围","高度范围","最常见尺寸（最多10项）"],dimrows,["---","---:","---:","---:","---:","---"]))
    add("")
    add(f"高基数尺寸数据集（特别是AP-10K和ExDark）的全部精确尺寸频次另存于`{DIM_CSV.name}`；上表并非推测，而是全量尺寸扫描的压缩展示。训练时保持长宽比缩放并填充到640，不是把原图强行拉伸成640×640。")
    add("")
    add("## 4. 各数据集完整统计")
    add("")
    for ds in DATASETS:
        d=audit["datasets"][ds]; tr=d["splits"]["train"]; te=d["splits"]["test"]; m=META[ds]; z=d["dimension_summary"]
        add(f"### 4.{DATASETS.index(ds)+1} {ds}")
        add("")
        add(f"- 领域与成像：{m['domain']}")
        add(f"- 原始标注/任务：{m['annotation']}")
        add(f"- 当前划分：{m['split']}")
        add(f"- 转换与使用说明：{m['processing']}")
        add(f"- 原图尺寸：{z['unique']}种，宽{z['width_min']}–{z['width_max']}、高{z['height_min']}–{z['height_max']}；常见尺寸为{dimension_top(d)}。")
        add("")
        splitrows=[]
        for label,s in [("train",tr),("test/每轮评估",te),("合计",None)]:
            if s is not None:
                splitrows.append([label,f"{s['images']:,}",f"{s['boxes']:,}",f"{s['empty_images']:,}",fmt(s['objects_per_image_mean'],2),fmt(s['objects_per_image_median'],1),fmt(s['objects_per_image_p90'],1),f"{s['objects_per_image_max']:,}",f"{s['bbox640_width_median']:.1f}×{s['bbox640_height_median']:.1f}",f"{100*s['bbox640_small_lt32sq_fraction']:.1f}%",f"{100*s['bbox640_min_side_lt8_fraction']:.1f}%"])
            else:
                splitrows.append([label,f"{tr['images']+te['images']:,}",f"{tr['boxes']+te['boxes']:,}",f"{tr['empty_images']+te['empty_images']:,}","—","—","—","—","—","—","—"])
        add(md_table(["分区","图片","框","空标签图","框/图均值","框/图中位数","框/图P90","单图最多框","640后中位框宽×高","640后面积<32²","640后最短边<8"],splitrows,["---","---:","---:","---:","---:","---:","---:","---:","---:","---:","---:"]))
        add("")
        classrows=[]
        for cid,name in sorted(((int(k),v) for k,v in d["names"].items())):
            tb=tr["class_boxes"][str(cid)]; vb=te["class_boxes"][str(cid)]; ti=tr["class_images"][str(cid)]; vi=te["class_images"][str(cid)]
            classrows.append([cid,name,f"{ti:,}",f"{tb:,}",f"{vi:,}",f"{vb:,}",f"{ti+vi:,}",f"{tb+vb:,}",f"{100*(tb+vb)/(tr['boxes']+te['boxes']):.2f}%"])
        add(md_table(["ID","类别","train含类图片","train框","test含类图片","test框","总含类图片","总框","框占比"],classrows,["---:","---","---:","---:","---:","---:","---:","---:","---:"]))
        add("")
    add(f"所有类别统计也以长表形式保存于`{CLASS_CSV.name}`，便于后续绘图、重采样和长尾分析。")
    add("")
    add("## 5. 数据完整性与标签处理")
    add("")
    add("- 10套、68个冻结清单文件、25,564,671,681 bytes已完成本机到服务器的逐文件哈希核对；压缩包执行成员CRC或内容校验；解压成员集合及全部图片解码通过。")
    add("- 11个派生版本、22个train/test分区经Ultralytics 8.4.146全量扫描：损坏图0、最终重复标签警告0；每个分区真实取样均得到`3×640×640`输入张量。")
    add("- 原始包、原图和原始标注未为通过验收而修改；所有裁剪、忽略、去重只发生在`processed_yolo`派生标签中，并保留转换报告与source map。")
    add("- 显式移除389个派生重复标签行：ORCA 357、SeaClear 6、VisDrone 4、ZeroWaste-f 22。其他边界修正见各数据集小节。")
    add("- 用户决定不额外建立val，也暂不处理视频相邻帧或场景级泄漏。因此本报告不能声称这些结果是严格无泄漏的跨场景泛化结果。")
    add("")
    add("## 6. 模型与训练配置")
    add("")
    add("### 6.1 模型")
    add("")
    add("- 模型：Ultralytics YOLOv8n detection架构，源码YAML随机初始化；`pretrained=False`，不加载任何`.pt`作为训练初始权重。")
    add("- 模型YAML：`/home/ubuntu/rz/object_detection_dataset_test/models/yolov8n_scratch.yaml`。")
    add("- YAML SHA256：`44c9efc76c9dad388ad1ef6c4efb37e47ec885fe2ad12f110c76f6cdb750889d`。")
    add("- 名义结构（nc=80构图检查）：129层、3,157,200参数、3,157,184个可训练梯度、约8.9 GFLOPs；训练时检测头的`nc`由各数据集覆盖，因此不同类数版本的头部参数量略有变化。")
    add("- 检测尺度：P3/P4/P5，典型步长8/16/32；没有额外P2小目标头。")
    add("- 启动时Ultralytics曾为AMP兼容性自检自动加载`yolo26n.pt`，但该模型不进入YOLOv8n参数初始化；checkpoint审计确认真实训练模型仍为随机初始化YOLOv8n。")
    add("")
    add("### 6.2 冻结训练超参数")
    add("")
    add(md_table(["项目","值"],[
        ["任务","detect/train"],["epochs","125"],["种子","2027、2005；每数据集两次独立训练"],["batch","64，全局batch；4卡DDP时约16/卡"],["输入","imgsz=640，保持长宽比缩放+填充"],["优化器","SGD"],["初始/末端学习率因子","lr0=0.01，lrf=0.01"],["动量/权重衰减","momentum=0.937，weight_decay=0.0005"],["warmup","3 epochs；warmup_momentum=0.8；warmup_bias_lr=0.1"],["损失权重","box=7.5，cls=0.5，dfl=1.5"],["调度","cos_lr=False；patience=0，不早停"],["AMP","True"],["确定性","deterministic=True"],["workers","4（配置值；DDP实现按rank构造loader）"],["pinned memory","True：该代码版本`build_dataloader(..., pin_memory=True)`，CUDA加载器保持启用"],["cache","False"],["保存","best.pt、last.pt，且每25轮阶段保存"],["验证","val=True；实际val路径=test"],["评估NMS","iou=0.7，max_det=300，conf自动"],
    ],["---","---"]))
    add("")
    add("### 6.3 数据增强")
    add("")
    add(md_table(["增强项","值"],[["HSV","h=0.015，s=0.7，v=0.4"],["几何","degrees=0，translate=0.1，scale=0.5，shear=0，perspective=0"],["翻转","flipud=0，fliplr=0.5"],["Mosaic","1.0；最后10轮关闭"],["MixUp / CutMix / Copy-Paste","均为0"],["multi_scale","0.0"],["rect","False"],["dropout","0.0"]],["---","---"]))
    add("")
    add("## 7. 服务器、GPU与软件环境")
    add("")
    add(md_table(["项目","实际环境"],[
        ["服务器","ASUS ESC8000A-E13-DR；hostname `ubuntu`"],["系统","Ubuntu 24.04.5 LTS，x86-64"],["内核","Linux 6.17.0-23-generic"],["CPU","2× AMD EPYC 9V74 80-Core；160物理核/320线程"],["内存","约251GiB；2个NUMA node，各约128GiB；swap 8GiB"],["GPU总配置","8× NVIDIA GeForce RTX 3090，每卡24,576MiB"],["本实验GPU","仅物理GPU 4、5、6、7；`CUDA_VISIBLE_DEVICES=4,5,6,7`，进程内编号0–3"],["NUMA绑定","`numactl --cpunodebind=1 --membind=1`；node1 CPU为80–159、240–319"],["NVIDIA驱动","580.173.02"],["Python","3.11.16"],["PyTorch","2.6.0+cu118；torch CUDA 11.8；cuDNN 90100"],["Ultralytics","8.4.146；仓库commit `9d45ee4613e79d7bceef7a39e42c15c278b0d4e8`，采集时工作区clean"],["OpenCV / NumPy / Pillow","5.0.0 / 2.4.6 / 12.3.0"],["Conda前缀","`/home/ubuntu/rz/yolo_study/.conda/yolo26`"],
    ],["---","---"]))
    add("")
    add("历史内核日志显示NUMA node0对应物理内存发生不可纠正错误并向若干进程发送SIGBUS。经服务器所有者确认继续使用后，本队列将CPU与内存硬绑定到node1，并只使用与node1相邻的GPU 4–7。最终22次训练从00:47至07:39连续完成，未发生重启恢复、checkpoint损坏、新MCE/SIGBUS或GPU Xid。该规避不等于硬件已修复，后续论文最终结果仍建议在健康硬件复现。")
    add("")
    add("## 8. 检测结果")
    add("")
    add("### 8.1 双种子汇总")
    add("")
    sumrows=[]
    for r in summary:
        sumrows.append([r['dataset'],pm(r['last_precision_mean'],r['last_precision_std']),pm(r['last_recall_mean'],r['last_recall_std']),pm(r['last_F1_mean'],r['last_F1_std']),pm(r['last_mAP50_mean'],r['last_mAP50_std']),pm(r['last_mAP50-95_mean'],r['last_mAP50-95_std']),pm(r['best_epoch_mAP50-95_mean'],r['best_epoch_mAP50-95_std'])])
    add(md_table(["数据集","last P","last R","last F1","last mAP50","last mAP50-95","各种子best mAP50-95"],sumrows,["---","---:","---:","---:","---:","---:","---:"]))
    add("")
    add("标准差为两个种子的样本标准差。`best`列先在每个种子内部按mAP50-95选最佳epoch，再对两个种子求mean±std；不同种子的最佳epoch可以不同。")
    add("")
    add("### 8.2 22次独立运行的完整总体检测指标")
    add("")
    runrows=[]
    for r in per_run:
        runrows.append([r['dataset'],r['seed'],r['best_epoch_by_map50_95'],fmt(r['last_precision']),fmt(r['last_recall']),fmt(r['last_F1']),fmt(r['last_mAP50']),fmt(r['last_mAP50-95']),fmt(r['best_epoch_precision']),fmt(r['best_epoch_recall']),fmt(r['best_epoch_F1']),fmt(r['best_epoch_mAP50']),fmt(r['best_epoch_mAP50-95'])])
    add(md_table(["数据集","seed","best epoch","last P","last R","last F1","last AP50","last AP50-95","best P","best R","best F1","best AP50","best AP50-95"],runrows,["---","---:","---:","---:","---:","---:","---:","---:","---:","---:","---:","---:","---:"]))
    add("")
    add("### 8.3 第125轮损失与训练耗时")
    add("")
    lossrows=[]
    for r in per_run:
        x=run_loss[(r['dataset'],int(r['seed']))]
        lossrows.append([r['dataset'],r['seed'],fmt(x['train_box']),fmt(x['train_cls']),fmt(x['train_dfl']),fmt(x['val_box']),fmt(x['val_cls']),fmt(x['val_dfl']),f"{x['time_seconds']/60:.1f}"])
    add(md_table(["数据集","seed","train box","train cls","train dfl","test/val box","test/val cls","test/val dfl","results.csv累计时间/min"],lossrows,["---","---:","---:","---:","---:","---:","---:","---:","---:"]))
    add("")
    add("累计时间来自每个`results.csv`的最终`time`字段，用于同一实现内的运行记录，不是经过严格隔离的吞吐benchmark；共享服务器负载可能影响它。")
    add("")
    add("## 9. 结果解读与已确认的主要现象")
    add("")
    add("1. **最高的固定末轮mAP50-95**：HRSID-JPG 0.6302、DUO 0.6174、LLVIP-infrared 0.6124、AP-10K 0.6111。数据集类别数、目标尺度和评测难度不同，这不是模型跨数据集排行榜。")
    add("2. **LLVIP红外显著优于可见光**：末轮双种子mAP50-95为0.6124对0.4643，红外高0.1480；F1高0.1003。两个模态样本数、框数、尺寸和标签完全一致。1,000对测试图审计显示红外目标—局部背景绝对灰度差中位数20.21，可见光仅7.97，主要是低照场景下热目标信噪比更高。")
    add("3. **ORCA低分主因是极端长尾**：24类中train最大类12,696框，最小类仅1框，9类少于100框；目标中位尺寸并不小。两个种子best mAP50-95约0.2852且极稳定。")
    add("4. **VisDrone低分主因是640下极小且密集**：test中93.2%的框面积小于32²，52.4%的目标最短边小于8像素，中位框约9.0×10.5像素；每图平均46.6框。两个种子都在epoch114–115附近饱和，单纯增加epoch价值很小。")
    add("5. **ZeroWaste-f主要是长尾、杂乱/不规则目标及过拟合**：640下中位框约100×95像素，并非小目标主导；cardboard 14,302个train框而metal仅319。两个种子都在epoch92–93达峰，best mAP50-95均值0.2311，末轮降到0.2128；训练分类损失继续下降而test/val分类损失回升。")
    add("6. **从零训练是重要条件**：与公开论文或官方预训练模型比较时，必须明确这些数字没有ImageNet/COCO或检测权重初始化；特别是细粒度、多类别和极小目标任务会明显更难。")
    add("")
    add("## 10. 可复现性、限制与论文使用规则")
    add("")
    add("- 最终审计检查了22份`results.csv`严格包含epoch 1–125，44个`best.pt`/`last.pt`均可读取，checkpoint中的epochs、batch、imgsz、workers、seed、优化器、GPU映射、`pretrained=False`、模型和数据路径与冻结协议一致。")
    add("- 每个数据集仅两个种子。可用于初筛和建立方差感知基线；强论文结论通常仍需至少3个种子、置信区间或显著性检验。")
    add("- test参与每轮评估和best选择；论文最终实验应建立独立val用于模型选择，并只对test运行一次最终评估。当前last第125轮结果比best更少受到早期轮次选择影响，但训练开发过程仍看到了test。")
    add("- 未处理相邻视频帧、配对场景、地点/相机级泄漏；SeaClear还是稳定随机80/20划分。涉及泛化主张时必须重新按组划分。")
    add("- AP-10K来自姿态数据，框标注未必等同于穷尽式通用目标检测；ZeroWaste-f原生更适合实例分割；ORCA原生还包含文本描述/开放词汇语义。本报告只评价当前水平框闭集检测派生任务。")
    add("- 所有实验统一640和同一训练策略是为了基线可比，不代表每个数据集的最佳配置。VisDrone最需要更高分辨率/P2/切片；ZeroWaste可测试类别平衡和分割辅助；ORCA需要长尾/少样本协议。")
    add("- 不应直接把不同数据集的mAP数值横向解释为数据质量高低。类别数、目标大小、标注口径、场景难度和评测集都不同。")
    add("")
    add("## 11. 证据文件索引")
    add("")
    add("- `runtime_audit/audit_latest.json`：最终机器审计结果。")
    add("- `runtime_audit/audit_per_run_metrics.csv`：22次运行的last/best总体指标。")
    add("- `runtime_audit/audit_two_seed_summary.csv`：11个版本的双种子mean±std。")
    add("- `runtime_audit/manager_events.jsonl`：队列启动、完成和恢复事件。")
    add("- `YOLO_DATASET_PREPARATION_2026-09-21.md`：转换、扫描与就绪报告。")
    add("- `DATASET_SCALE_AND_SIZE_2026-09-22.md`：规模与磁盘口径。")
    add("- `DATASET_IMAGE_DIMENSIONS_2026-09-22.md`：尺寸汇总。")
    add(f"- `{CLASS_CSV.name}`：全部数据集、全部类别的图片数与框数长表。")
    add(f"- `{DIM_CSV.name}`：全部原图尺寸频次长表。")
    add("- 服务器每个运行目录内：`args.yaml`、`results.csv`、`results.png`、PR/P/F1曲线、混淆矩阵、预测样例与`weights/best.pt`、`weights/last.pt`。")
    add("")
    add("---")
    add("")
    add("本报告只汇总已经实际完成并通过审计的数据与实验，不包含未下载的JODD，也不把此前VOC/YOLO26研究或中断的其他探索混入本轮11版本YOLOv8n随机初始化基线。")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "class_csv": str(CLASS_CSV), "dimension_csv": str(DIM_CSV), "datasets": len(DATASETS), "runs": len(per_run), "report_bytes": REPORT.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
