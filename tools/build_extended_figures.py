"""Additional source-indexed diagnostic plots. No training or dataset images."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures';RES=ROOT/'results'
VOC='archives/local/voc_pgb_yolov8n/'
COLORS=['#245b91','#de7b27','#38947a','#b1505a','#8757a3','#736d39']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'savefig.dpi':180,'svg.fonttype':'none'})

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def csvrows(p):
 with (ROOT/p).open(encoding='utf-8',newline='') as f:
  return [{k.strip():v for k,v in r.items() if k is not None} for r in csv.DictReader(f)]
def save(name,fig):
 fig.tight_layout();fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
 fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def dump(name,data):
 keys=list(dict.fromkeys(k for r in data for k in r))
 with (RES/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)

# Cross-entropy saturation and train/test accuracy, rather than accuracy alone.
end=csvrows('results/cub_primary_and_secondary_endpoints.csv')
fig,axs=plt.subplots(1,2,figsize=(13,4.5))
for i,g in enumerate('ABCDEF'):
 r=next(r for r in end if r['group']==g);d=csvrows(r['source_csv'])
 axs[0].plot([int(x['epoch']) for x in d],[float(x['train_top1']) for x in d],color=COLORS[i],label=g+' train')
 axs[1].bar(g,100-float(r['test_top1']),color=COLORS[i])
axs[0].set(xlabel='Completed epoch',ylabel='Training Top1 (%)',ylim=(0,103));axs[0].legend(ncol=3)
axs[1].set(ylabel='Train Top1 − test Top1 (percentage points)',ylim=(0,100),title='All six arms reach 100% train Top1 at epoch125')
save('cub_train_accuracy_and_generalization_gap',fig)

# Synthetic geometry shows the implemented rectangular protection, not a bird.
yy,xx=np.mgrid[:500,:500];cx,cy,w,h=250,240,180,130
dx=np.maximum(np.abs(xx+.5-cx)-.6*w,0);dy=np.maximum(np.abs(yy+.5-cy)-.6*h,0)
dist=np.hypot(dx,dy);t=np.clip(dist/70,0,1);alpha=t*t*(3-2*t)
fig,axs=plt.subplots(1,3,figsize=(13,4))
axs[0].imshow(alpha,cmap='magma',vmin=0,vmax=1)
for ax in axs[:1]:
 ax.add_patch(Rectangle((cx-w/2,cy-h/2),w,h,fill=False,ec='cyan',lw=1.5,label='GT box'))
 ax.add_patch(Rectangle((cx-.6*w,cy-.6*h),1.2*w,1.2*h,fill=False,ec='white',lw=1.5,label='10% expansion'))
 ax.legend(fontsize=7,loc='lower left');ax.set_title('C/D processing strength (synthetic geometry)')
axs[1].plot(np.arange(500),alpha[cy],color=COLORS[0]);axs[1].set(xlabel='Pixel x at target-center row',ylabel='Strength a',ylim=(-.05,1.05),title='70px smoothstep outside expanded box')
levels=np.array([0,2,4,8.]);sigma=alpha[cy]*8
weights=np.zeros((4,500))
for i,(lo,hi) in enumerate(zip(levels[:-1],levels[1:])):
 selected=(sigma>=lo)&((sigma<=hi) if i==2 else (sigma<hi))
 blend=(sigma[selected]-lo)/(hi-lo)
 weights[i,selected]=1-blend;weights[i+1,selected]=blend
assert np.allclose(weights.sum(axis=0),1)
for level,weight in zip(levels,weights):axs[2].plot(np.arange(500),weight,label='σ='+str(int(level)))
axs[2].set(xlabel='Pixel x',ylabel='Renderer weight',ylim=(-.05,1.05),title='D interpolation: σ=8a, bank0/2/4/8');axs[2].legend(fontsize=8)
axs[0].set_xlabel('500px canvas');axs[0].set_ylabel('500px canvas')
save('cub_protection_geometry_illustration',fig)

# Pipeline diagram: three implemented families with distinctly placed GT use.
fig,axs=plt.subplots(3,1,figsize=(14,6))
def box(ax,x,text,color):
 patch=FancyBboxPatch((x,.20),.18,.53,boxstyle='round,pad=.018',fc=color,ec='#666666',lw=.8)
 ax.add_patch(patch);ax.text(x+.09,.465,text,ha='center',va='center',fontsize=9)
def arrow(ax,x):ax.annotate('',xy=(x+.035,.465),xytext=(x,.465),arrowprops={'arrowstyle':'->','color':'#555555'})
pipelines=[
 ('Learned detection frontend',['Raw RGB','Image-only\npermission predictor','Differentiable blur\n+ original residual','Unchanged YOLO\n(all objects)'],['#e8eef5','#dceded','#f9ecd7','#e8eef5']),
 ('CUB Oracle representations',['Raw RGB\n+ training RGB jitter','GT expanded box\n(train AND test)','C: fade to mean\nD: multi-scale blur','Native500\nResNet50'],['#e8eef5','#f5dddd','#f9ecd7','#e8eef5']),
 ('CUB G4 / G5 two-view',['Global image\n(or normalized zero)','GT crop\nlong-side resize500','Shared ResNet50\nG4/G5 mixed BN','2048 + 2048\n→ 200 logits'],['#e8eef5','#f5dddd','#dceded','#e8eef5'])]
for ax,(title,texts,colors) in zip(axs[:2],pipelines[:2]):
 ax.set_xlim(-.04,1.05);ax.set_ylim(0,1);ax.axis('off');ax.text(-.025,.94,title,fontweight='bold',fontsize=11)
 for x,text,color in zip([0,.26,.52,.78],texts,colors):box(ax,x,text,color)
 for x in [.205,.465,.725]:arrow(ax,x)
ax=axs[2];ax.set_xlim(-.04,1.05);ax.set_ylim(0,1);ax.axis('off');ax.text(-.025,.94,pipelines[2][0],fontweight='bold',fontsize=11)
for y,text,color in [(.55,'Global image\n(or normalized zero)','#e8eef5'),(.17,'GT local crop\nlong-side resize500','#f5dddd')]:
 ax.add_patch(FancyBboxPatch((0,y),.28,.26,boxstyle='round,pad=.018',fc=color,ec='#666666',lw=.8))
 ax.text(.14,y+.13,text,ha='center',va='center',fontsize=9)
 ax.annotate('',xy=(.52,.465),xytext=(.30,y+.13),arrowprops={'arrowstyle':'->','color':'#555555'})
box(ax,.52,'Shared ResNet50\nG4/G5 mixed BN','#dceded');box(ax,.78,'2048 + 2048\n→ 200 logits','#e8eef5');arrow(ax,.725)
axs[0].text(.26,.04,'Training GT supplies auxiliary supervision, not inference locations.',fontsize=9)
axs[2].text(.26,.04,'G6 shares convolution/affine weights but separates per-view BN statistics; stopped at50.',fontsize=9)
save('implemented_method_pipelines',fig)

# Whole-dev conservative gate behavior, with exact denominators in CSV.
gate=read(VOC+'SAFE_BACKGROUND_LONG_E125_METHOD_GATE_FULL_DEV.json')
gdata=[{'size':k,**v,'source_json':VOC+'SAFE_BACKGROUND_LONG_E125_METHOD_GATE_FULL_DEV.json'} for k,v in gate['instance_by_size'].items()]
dump('voc_gate_size_protection.csv',gdata)
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for i,r in enumerate(gdata):
 axs[0].bar(i,100*r['safe_fraction'],color=COLORS[i]);axs[0].text(i,100*r['safe_fraction']+1,f"{r['safe_at_0_1']}/{r['total']}",ha='center',fontsize=9)
axs[0].set_xticks(range(3),[r['size'] for r in gdata]);axs[0].set(ylabel='Safe targets (%)',ylim=(0,105),title='≥95% GT pixels have permission ≤.1')
regions=list(gate['pixel_region']);x=np.arange(len(regions));width=.25
for offset,key,label in [(-1,'mean_permission','Mean permission'),(0,'0.25','Pixels >.25'),(1,'0.5','Pixels >.5')]:
 vals=[gate['pixel_region'][k]['mean_permission'] if key=='mean_permission' else gate['pixel_region'][k]['fraction_permission_above'][key] for k in regions]
 axs[1].bar(x+offset*width,vals,width,label=label)
axs[1].set_xticks(x,regions);axs[1].set(ylabel='Permission / fraction',ylim=(0,1),title='Box region and far background are different');axs[1].legend(fontsize=8)
save('voc_gate_protection_and_permission',fig)

# HBS has distinct training curves; size bucket definition matches original JSON.
fig,axs=plt.subplots(1,2,figsize=(13,4.7));recalls=[]
for label,run,source in [
 ('IdentityAux','identity_aux_s2027_g0123_numa0','SET_HBS_VISDRONE_IDENTITY_LAST_SIZE_RECALL.json'),
 ('HBSonly','hbs_only_s2027_g0123_numa0','SET_HBS_VISDRONE_HBS_LAST_SIZE_RECALL.json')]:
 d=csvrows('archives/remote/progressive_blur/runs/set_hbs_visdrone_960_e125/'+run+'/results.csv')
 axs[0].plot([int(r['epoch']) for r in d],[100*float(r['metrics/mAP50-95(B)']) for r in d],label=label)
 counts=read(VOC+source)['counts'];names=['tiny_lt16','small_16to32','medium_32to96','large_ge96']
 vals=[100*counts[k]['0.75']['recall'] for k in names]
 axs[1].plot(names,vals,'-o',label=label)
 for k in names:
  for iou,v in counts[k].items():recalls.append({'arm':label,'size_bucket':k,'iou':iou,**v,'source_json':VOC+source})
axs[0].set(xlabel='Completed epoch',ylabel='Training-validator AP50–95 (%)');axs[0].legend()
axs[1].set(ylabel='One-to-one diagnostic recall at IoU .75 (%)',ylim=(0,100),title='Size at hypothetical640 letterbox; not AP_small');axs[1].legend();axs[1].tick_params(axis='x',rotation=30)
dump('hbs_size_recall.csv',recalls);save('hbs_learning_and_size_recall',fig)

# Fixed mask policy transfer: denominator restricted to fully-in-crop GT.
val=read(VOC+'FOVEATED_TARGET_PROTECTION_VAL548.json')['summary']['0.01']
test=read(VOC+'FOVEATED_PREDICTED_BG_BLUR_TESTDEV_FULL1610.json')['protection_audit']
pd=[{'split':'val548','eligible_gt':val['eligible_gt'],'protected95':val['protected95_gt'],
 'tiny_eligible':val['size']['lt8']['eligible'],'tiny_protected95':val['size']['lt8']['protected95'],
 'permitted_area':val['blur_permitted_fraction_selected_crop_pixels'],'source_json':VOC+'FOVEATED_TARGET_PROTECTION_VAL548.json'},
 {'split':'testdev1610','eligible_gt':test['gt_fully_in_selected_crop'],'protected95':test['gt_protected95'],
 'tiny_eligible':test['gt_lt8_fully_in_crop'],'tiny_protected95':test['gt_lt8_protected95'],
 'permitted_area':test['blur_permitted_fraction_selected_crops'],'source_json':VOC+'FOVEATED_PREDICTED_BG_BLUR_TESTDEV_FULL1610.json'}]
dump('foveation_mask_transfer_counts.csv',pd)
fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(3);width=.36
for i,r in enumerate(pd):
 vals=[100*r['protected95']/r['eligible_gt'],100*r['tiny_protected95']/r['tiny_eligible'],100*r['permitted_area']]
 ax.bar(x+(i-.5)*width,vals,width,label=r['split'])
ax.set_xticks(x,['Eligible targets safe','Tiny<8px targets safe','Selected-crop area permitted for blur']);ax.set(ylabel='Percent',ylim=(0,105),title='Same predicted-mask policy: GT fully inside selected crops only');ax.legend()
save('foveation_mask_val_test_transfer',fig)

# Assignment gap at all frozen checkpoints, plus the declared stop gate.
src='archives/remote/yolo26/audits/fca_p0_formal_A1_20260919_1912/decision.json'
fc=read(src);cp=fc['checkpoint_metrics'];dump('fca_o2o_checkpoint_coverage_gap.csv',[{'source_json':src,**r} for r in cp])
fig,axs=plt.subplots(1,2,figsize=(12,4.5));labels=[r['checkpoint'].replace('.pt','') for r in cp]
axs[0].bar(labels,[r['gap_r000'] for r in cp],color=COLORS[0]);axs[0].set(ylabel='Avoidable O2O target-coverage gaps',title='Original filename index retained; six checkpoints');axs[0].tick_params(axis='x',rotation=30)
rate=100*fc['bootstrap']['rate'];lo=100*fc['bootstrap']['ci95_lower'];hi=100*fc['bootstrap']['ci95_upper']
axs[1].errorbar([0],[rate],yerr=[[rate-lo],[hi-rate]],fmt='o',capsize=5,label='Image-cluster bootstrap95% CI')
axs[1].axhline(.25,color=COLORS[3],linestyle='--',label='Preregistered training gate .25%')
axs[1].set(xticks=[0],xticklabels=['Pooled31 / 240,348'],ylabel='Avoidable gap rate (%)',ylim=(0,.28),title='Below gate: no coverage-first formal training');axs[1].legend(fontsize=8)
save('fca_coverage_gap_screening',fig)

# Source-indexed directory of all generated figures, identical evidence in both languages.
descriptions={
 'cub_A_F_learning_curves':('CUB A–F test accuracy and losses','CUB A–F 测试准确率与训练/测试损失','results/cub_primary_and_secondary_endpoints.csv'),
 'cub_A_F_endpoints_and_cost':('CUB fixed125 endpoints and historical epoch cost','CUB 固定125轮结果与历史耗时','results/cub_primary_and_secondary_endpoints.csv'),
 'cub_global_local_and_incomplete_bn':('G4/G5 and incomplete G6 trajectories','G4/G5及未完成G6逐轮轨迹','results/cub_primary_and_secondary_endpoints.csv'),
 'cub_class_distribution':('All200 CUB classes, train/test counts','CUB 全部200类 train/test 数量','results/cub_all_200_class_counts.csv'),
 'dataset_22_baseline_endpoints':('22 scratch detection baselines','22组从零检测基线','results/dataset_22_run_endpoints.csv'),
 'yolo26_short_ablation_endpoints':('YOLO26 short-budget controls, seeds42/0; truncated axis','YOLO26短训对照，种子42/0；横轴截断','results/yolo26_run_endpoints.csv'),
 'yolo26_150_baseline_reproduction':('YOLO26 baseline/gain-zero trajectories, different validation heads','YOLO26基线/零增益轨迹，评测头不同','results/yolo26_run_endpoints.csv'),
 'asac_paired_training':('ASAC A2/A0 AP and actual q','ASAC A2/A0 AP及实际q','results/asac_controller_epoch_states.csv'),
 'voc125_classwise':('VOC independent20-class AP difference','VOC独立评测20类AP差值','results/voc125_classwise.csv'),
 'visdrone_hbs_classwise':('HBS independent10-class comparison','HBS独立评测10类对照','results/visdrone_hbs_classwise.csv'),
 'voc125_curves_and_map_loss':('VOC long-run AP and auxiliary map loss','VOC长训AP与模糊辅助损失','results/all_canonical_epoch_curves.csv'),
 'testtime_blur_GT_vs_prediction':('Three-strength inference, GT/predicted protection','三档推理模糊，GT/预测保护','results/detection_metric_ledger.csv'),
 'foveation_val_and_testdev_endpoints':('Foveation endpoints; truncated axis','凹视val/test-dev结果；纵轴截断','results/foveation_same_protocol_endpoints.csv'),
 'foveation_TP_FP_by_class':('Per-class fixed-threshold true/false positives','固定阈值逐类真阳性/误报','results/foveation_fixed_threshold_classwise_TP_FP.csv'),
 'cub_train_accuracy_and_generalization_gap':('CUB train accuracy and generalization gap','CUB训练准确率及泛化差距','results/cub_primary_and_secondary_endpoints.csv'),
 'cub_protection_geometry_illustration':('Synthetic rectangular protection geometry, NOT an experimental image','矩形保护几何示意，非真实实验图','tools/build_extended_figures.py'),
 'implemented_method_pipelines':('Implemented input/model/GT pathways','已实现的输入、模型及真值使用路径','docs/en/03_voc_blur.md'),
 'voc_gate_protection_and_permission':('Target protection and background permission','目标保护与背景处理许可','results/voc_gate_size_protection.csv'),
 'hbs_learning_and_size_recall':('HBS AP curves and size-specific recall','HBS AP曲线与按尺寸召回','results/hbs_size_recall.csv'),
 'foveation_mask_val_test_transfer':('Mask protection and area transfer, eligible targets only','掩膜保护率/面积跨集变化，仅统计合格目标','results/foveation_mask_transfer_counts.csv'),
 'fca_coverage_gap_screening':('FCA full coverage audit and go/no-go threshold','FCA全量覆盖审计及启动阈值','results/fca_o2o_checkpoint_coverage_gap.csv')}
dump('figure_source_index.csv',[{'figure':name,'description_en':v[0],'description_zh':v[1],'primary_source':v[2]} for name,v in descriptions.items()])
for lang in ['en','zh-CN']:
 cn=lang=='zh-CN';text='# '+('图表及数据来源' if cn else 'Figures and source data')+'\n\n'
 text+=('图内英文标签便于两版本共享。所有检测AP先换算为百分数；差值用百分点。没有合成实验结果、未保存的置信区间或额外训练。\n\n' if cn else 'English labels are shared between editions. Detection AP is scaled to percent; deltas are percentage points. No experimental observations, unsaved confidence intervals or new training are synthesized.\n\n')
 for name,v in descriptions.items():
  text+='## '+v[1 if cn else 0]+'\n\n'
  text+=f'![{v[1 if cn else 0]}](../../figures/{name}.png)\n\n'
  text+=f'[SVG](../../figures/{name}.svg) · ['+('数据来源' if cn else 'Source')+f'](../../{v[2]})\n\n'
 (ROOT/'docs'/lang/'figures.md').write_text(text,encoding='utf-8')
