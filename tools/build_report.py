"""Build publication tables/figures from archived evidence; never runs training.

Run: python tools/build_report.py (Python 3.11+, numpy, matplotlib).
All detection APs are fractions in CSV; plot axes use percent/AP points.
Classification Top1/Top5 already use percent in their original CSVs.
"""
from pathlib import Path
import csv, json, math, re, statistics, hashlib, runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'; FIG=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':180,'svg.fonttype':'none'})
COLORS=['#245b91','#de7b27','#38947a','#b1505a','#8757a3','#736d39','#299baf','#b38b18','#333333']
plt.rcParams['axes.prop_cycle']=matplotlib.cycler(color=COLORS)

def rows(path):
 with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:
  return [{k.strip():v for k,v in r.items() if k is not None} for r in csv.DictReader(f)]
def dump(name,data):
 data=list(data)
 if not data:return
 keys=list(dict.fromkeys(k for r in data for k in r))
 with (RESULTS/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def num(x):
 try: return float(x)
 except (ValueError,TypeError): return float('nan')
def rel(p): return p.relative_to(ROOT).as_posix()
def save(name,fig):
 fig.tight_layout();fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
 fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def line(ax,data,col,label,scale=1,**kw):
 ax.plot([num(r['epoch']) for r in data],[num(r.get(col)) * scale for r in data],label=label,**kw)
 ax.set_xlabel('Completed epoch');ax.grid(alpha=.2)
def table_md(data,cols,labels=None):
 labels=labels or cols
 out=['| '+' | '.join(labels)+' |','| '+' | '.join('---' for _ in cols)+' |']
 for r in data:
  vals=[]
  for k in cols:
   v=r.get(k,'')
   if isinstance(v,float):v=f'{v:.6f}'
   vals.append(str(v).replace('|','/').replace('\n',' '))
  out.append('| '+' | '.join(vals)+' |')
 return '\n'.join(out)

# One canonical server CSV per run; local snapshots do not count as new experiments.
registry=[];curves=[]
for family in ['yolo26','dataset_baselines','progressive_blur','classification']:
 for p in sorted((ROOT/'archives/remote'/family).rglob('results.csv')):
  rp=rel(p); data=rows(rp)
  if not data or 'epoch' not in data[0]: continue
  epochs=[int(num(r['epoch'])) for r in data]
  small=any(s in rp.lower() for s in ['smoke','pilot','tiny','ddp_audit','w0_nopin'])
  reg={'family':family,'run':p.parent.name,'source_csv':rp,'completed_epochs':len(data),
       'last_epoch':epochs[-1],'continuous':epochs==list(range(1,epochs[-1]+1)),
       'classification':'engineering_smoke_or_pilot' if small else 'training_run'}
  if family=='yolo26':reg['detector_head']='O2M+NMS' if reg['run'].startswith('voc_yolo26n_baseline_') else 'O2O'
  seed_match=re.search(r'_s(\d+)(?:_|$)',reg['run'])
  if seed_match:reg['seed_from_run_name']=int(seed_match[1])
  args_path=p.parent/'args.yaml'
  if args_path.exists():
   reg['source_args_yaml']=rel(args_path)
   args_text=args_path.read_text(encoding='utf-8')
   for key in ['epochs','batch','imgsz','workers','optimizer','model','pretrained','resume','lr0','lrf','seed']:
    found=re.search(r'^'+key+r':\s*(.*)$',args_text,re.M)
    if found:reg['args_'+key]=found[1].strip()
  if family=='classification' and 'G6_' in rp:reg['classification']='user_stopped_incomplete'
  last=data[-1]
  for k in ['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)',
   'train_loss','train_top1','train_top5','test_loss','test_top1','test_top5','lr',
   'train/box_loss','train/cls_loss','train/dfl_loss','val/box_loss','val/cls_loss','val/dfl_loss','time']:
   if k in last:reg[k]=num(last[k])
  registry.append(reg)
  for r in data:curves.append({'family':family,'run':p.parent.name,'source_csv':rp,**r})
dump('training_run_registry.csv',registry);dump('all_canonical_epoch_curves.csv',curves)

# JSON metric ledger and per-class evidence: each context keeps its own protocol.
metric_ledger=[];class_ledger=[];numeric_diagnostics=[]
def visit(value,src,ptr='$'):
 if isinstance(value,dict):
  keys={'precision':['precision','P'],'recall':['recall','R'],'map50':['map50','mAP50','mAP50(B)'],
        'map50_95':['map50_95','mAP50_95','mAP50-95','mAP50-95(B)']}
  got={k:next((value[z] for z in aliases if z in value),None) for k,aliases in keys.items()}
  if isinstance(got['map50_95'],(int,float)):
   metric_ledger.append({'source_json':src,'json_pointer':ptr,**{k:v for k,v in got.items() if v is not None}})
  ap95=value.get('ap50_95',value.get('AP50_95'))
  if isinstance(ap95,(int,float)):
   class_ledger.append({'source_json':src,'json_pointer':ptr,'ap50_95':ap95,
    'ap50':value.get('ap50',value.get('AP50')),**{k:value[k] for k in
    ['id','name','precision','recall','gt','instances','images'] if k in value}})
  # These compact diagnostic rows preserve denominators/working points in nested JSON.
  if (('tp' in value and 'fp' in value) or ('matched' in value and 'targets' in value)
      or ('protected95_gt' in value and 'eligible_gt' in value)):
   numeric_diagnostics.append({'source_json':src,'json_pointer':ptr,**{k:v for k,v in value.items() if isinstance(v,(int,float,str,bool))}})
  for k,v in value.items():visit(v,src,ptr+'.'+str(k))
 elif isinstance(value,list):
  for i,v in enumerate(value):visit(v,src,ptr+f'[{i}]')
for p in sorted((ROOT/'archives').rglob('*.json')):
 if p.stat().st_size>12*1024*1024:continue
 try:visit(load(rel(p)),rel(p))
 except (ValueError,UnicodeError):pass
dump('detection_metric_ledger.csv',metric_ledger);dump('detection_classwise_ledger.csv',class_ledger)
dump('diagnostic_counts_ledger.csv',numeric_diagnostics)

# Classification primary and secondary endpoints from exact complete CSVs.
class_rows=[];class_curves={}
for reg in registry:
 if reg['family']!='classification':continue
 run=reg['run'];group=run.split('_')[1];data=rows(reg['source_csv']);class_curves[group]=data
 last=data[-1];best=max(data,key=lambda r:num(r['test_top1']))
 item={'group':group,'run':run,'seed':2005,'actual_epochs':len(data),'lr_horizon_epochs':200,
 'source_csv':reg['source_csv'],'status':'complete_audited_125' if len(data)==125 else 'user_stopped_at_50',
 **{k:num(last[k]) for k in ['train_top1','train_top5','train_loss','test_top1','test_top5','test_loss','lr']},
 'best_test_top1_epoch':int(num(best['epoch'])),'best_test_top1':num(best['test_top1']),
 'best_epoch_test_top5':num(best['test_top5']),'best_epoch_test_loss':num(best['test_loss'])}
 for k in ['train_seconds','test_seconds','epoch_seconds']:
  item['mean_'+k]=statistics.mean(num(r[k]) for r in data)
  item['sum_'+k]=sum(num(r[k]) for r in data)
 class_rows.append(item)
class_rows.sort(key=lambda x:x['group']);dump('cub_primary_and_secondary_endpoints.csv',class_rows)
fig,axes=plt.subplots(2,2,figsize=(13,8))
for g in 'ABCDEF':
 data=class_curves[g]
 for ax,col in zip(axes.flat,['test_top1','test_top5','train_loss','test_loss']):line(ax,data,col,g)
for ax,label in zip(axes.flat,['Test Top1 (%)','Test Top5 (%)','Training cross-entropy','Test cross-entropy']):ax.set_ylabel(label);ax.legend(ncol=3)
axes[1,0].set_yscale('log');save('cub_A_F_learning_curves',fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5));completed=[r for r in class_rows if r['group'] in 'ABCDEF']
x=np.arange(len(completed));w=.36
axes[0].bar(x-w/2,[r['test_top1'] for r in completed],w,label='Test Top1')
axes[0].bar(x+w/2,[r['test_top5'] for r in completed],w,label='Test Top5');axes[0].set_ylim(0,100)
axes[0].set_xticks(x,[r['group'] for r in completed]);axes[0].set_ylabel('Accuracy at epoch 125 (%)');axes[0].legend()
axes[1].bar(x,[r['mean_train_seconds'] for r in completed],label='Train')
axes[1].bar(x,[r['mean_test_seconds'] for r in completed],bottom=[r['mean_train_seconds'] for r in completed],label='Entire test set')
axes[1].set_xticks(x,[r['group'] for r in completed]);axes[1].set_ylabel('Mean seconds / epoch');axes[1].legend()
axes[1].set_title('Historical runtime; F shares GPU, not a controlled benchmark')
save('cub_A_F_endpoints_and_cost',fig)
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for g in ['G4','G5','G6']:
 for ax,col in zip(axes,['test_top1','train_top1','test_loss']):line(ax,class_curves[g],col,g+(' (stopped at 50)' if g=='G6' else ''),linestyle='--' if g=='G6' else '-')
for ax,label in zip(axes,['Test Top1 (%)','Training Top1 (%)','Test cross-entropy']):ax.set_ylabel(label);ax.legend(fontsize=8)
save('cub_global_local_and_incomplete_bn',fig)
profile=load('archives/local/foveated_classification/CUB_DATASET_FULL_PROFILE_2026-09-25.json')
dump('cub_all_200_class_counts.csv',profile['per_class'])
fig,ax=plt.subplots(figsize=(13,4));pc=profile['per_class'];ax.bar([r['id'] for r in pc],[r['train'] for r in pc],label='Train');ax.bar([r['id'] for r in pc],[r['test'] for r in pc],bottom=[r['train'] for r in pc],label='Test');ax.set_xlabel('CUB class ID (1–200)');ax.set_ylabel('Images');ax.legend();save('cub_class_distribution',fig)

# Dataset suite primary endpoints and test-selected best (secondary).
suite=[]
for reg in registry:
 if reg['family']!='dataset_baselines':continue
 data=rows(reg['source_csv']);last=data[-1];best=max(data,key=lambda r:num(r['metrics/mAP50-95(B)']))
 ds=re.search(r'yolov8n_scratch_(.+)_e125_b64_i640_s(\d+)',reg['run']);assert ds
 item={'dataset':ds[1],'seed':int(ds[2]),'epochs':len(data),'imgsz':640,'batch':64,'initialization':'scratch',
 'last_precision':num(last['metrics/precision(B)']),'last_recall':num(last['metrics/recall(B)']),
 'last_map50':num(last['metrics/mAP50(B)']),'last_map50_95':num(last['metrics/mAP50-95(B)']),
 'best_epoch':int(num(best['epoch'])),'best_map50':num(best['metrics/mAP50(B)']),
 'best_map50_95':num(best['metrics/mAP50-95(B)']),'source_csv':reg['source_csv']}
 item['derived_H_last_P_R']=2*item['last_precision']*item['last_recall']/(item['last_precision']+item['last_recall'])
 for k in ['train/box_loss','train/cls_loss','train/dfl_loss','val/box_loss','val/cls_loss','val/dfl_loss','time']:
  if k in last:item[k]=num(last[k])
 suite.append(item)
suite.sort(key=lambda r:(r['dataset'],r['seed']));dump('dataset_22_run_endpoints.csv',suite)
fig,axes=plt.subplots(1,2,figsize=(15,5));names=sorted(set(r['dataset'] for r in suite));x=np.arange(len(names))
for i,s in enumerate([2005,2027]):
 data=[next(r for r in suite if r['dataset']==n and r['seed']==s) for n in names]
 for ax,k in zip(axes,['last_map50_95','best_map50_95']):ax.bar(x+(i-.5)*.38,[100*r[k] for r in data],.38,label=str(s))
for ax,t in zip(axes,['Fixed epoch 125','Test-selected best (not blind)']):
 ax.set_xticks(x,names,rotation=55,ha='right',fontsize=8);ax.set_ylabel('AP50–95 (%)');ax.set_title(t);ax.legend(title='Seed')
save('dataset_22_baseline_endpoints',fig)

# YOLO26 fixed-budget, head-specific comparison; includes negative seeds.
y26=[r for r in registry if r['family']=='yolo26' and r['classification']=='training_run']
dump('yolo26_run_endpoints.csv',y26)
fig,axes=plt.subplots(1,2,figsize=(13,5))
for tag,title in [('s42','Seed 42, fixed 50 epochs'),('s0','Seed 0, fixed 50 epochs')]:
 ax=axes[0 if tag=='s42' else 1]
 data=[r for r in y26 if r['completed_epochs']==50 and tag in r['run'] and 'reboot_r1' not in r['run'] and not r['run'].startswith('asac')]
 labels=[r['run'].split('_e50')[0].replace('_g4567','').replace('_no_kd','') for r in data]
 ax.barh(labels,[100*num(r.get('metrics/mAP50-95(B)')) for r in data]);ax.set_xlim(49,53);ax.set_xlabel('CSV O2O AP50–95 (%)');ax.set_title(title);ax.tick_params(axis='y',labelsize=8)
save('yolo26_short_ablation_endpoints',fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for reg in y26:
 if reg['completed_epochs']==150:
  ax=axs[0 if reg['detector_head']=='O2M+NMS' else 1]
  line(ax,rows(reg['source_csv']),'metrics/mAP50-95(B)',reg['run'].split('_4x')[0],100)
  ax.set_title(reg['detector_head']+' training validator');ax.set_ylabel('AP50–95 (%)');ax.legend(fontsize=8)
fig.suptitle('B0 / D0: different recorded validation heads; weights match, scores must not be equated')
save('yolo26_150_baseline_reproduction',fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
controller=[]
for reg in y26:
 if reg['run'].startswith('asac_'):
  d=rows(reg['source_csv']);line(axes[0],d,'metrics/mAP50-95(B)',reg['run'].split('_e50')[0],100)
  state_path=ROOT/Path(reg['source_csv']).parent/'asac_state.jsonl'
  if state_path.exists():
   states=[json.loads(s) for s in state_path.read_text(encoding='utf-8').splitlines() if s.strip()]
   if states[0].get('variant')=='A2':line(axes[1],states,'train_q','A2 actual q',color=COLORS[1])
   for state in states:
    controller.append({'source_jsonl':rel(state_path),**{k:v for k,v in state.items() if isinstance(v,(int,float,str,bool)) or v is None}})
axes[1].axhline(1.,color=COLORS[1],linestyle='--',label='A0 fixed q=1')
dump('asac_controller_epoch_states.csv',controller)
axes[0].set_ylabel('O2O AP50–95 (%)');axes[0].legend(fontsize=8);axes[1].set_ylabel('Actual augmentation q');axes[1].set_xlabel('Completed epoch');axes[1].legend(fontsize=8);axes[1].set_ylim(.2,1.05);save('asac_paired_training',fig)

VOC='archives/local/voc_pgb_yolov8n/'
def paired_class_plot(name,paths,labels):
 ds=[load(VOC+p) for p in paths];pcs=[d['per_class'] for d in ds]
 names=[r['name'] for r in pcs[0]];fig,axes=plt.subplots(1,2,figsize=(14,5))
 for pc,label in zip(pcs,labels):axes[0].plot(names,[100*r['ap50_95'] for r in pc],'-o',label=label,markersize=3)
 delta=100*(np.array([r['ap50_95'] for r in pcs[1]])-np.array([r['ap50_95'] for r in pcs[0]]))
 axes[1].bar(names,delta,color=[COLORS[2] if v>=0 else COLORS[3] for v in delta]);axes[1].axhline(0,color='gray',lw=.7)
 for ax in axes:ax.tick_params(axis='x',rotation=60,labelsize=8)
 axes[0].set_ylabel('Independent last-checkpoint AP50–95 (%)');axes[0].legend()
 axes[1].set_ylabel(f'{labels[1]} − {labels[0]} (AP points)');save(name,fig)
 dump(name+'.csv',[{'class':n,labels[0]+'_ap95':pcs[0][i]['ap50_95'],labels[1]+'_ap95':pcs[1][i]['ap50_95'],'delta_AP_points':float(delta[i])} for i,n in enumerate(names)])
paired_class_plot('voc125_classwise',['SAFE_BACKGROUND_LONG_E125_BASELINE_LAST_INDEPENDENT.json','SAFE_BACKGROUND_LONG_E125_METHOD_LAST_INDEPENDENT.json'],['Baseline','SafeBlur'])
paired_class_plot('visdrone_hbs_classwise',['SET_HBS_VISDRONE_IDENTITY_LAST_VAL548.json','SET_HBS_VISDRONE_HBS_LAST_VAL548.json'],['IdentityAux','HBSonly'])
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for reg in registry:
 if '/sbg_long_e125/' in reg['source_csv']:
  d=rows(reg['source_csv']);label='SafeBlur' if 'sbg_joint' in reg['run'] else 'Baseline'
  line(axes[0],d,'metrics/mAP50-95(B)',label,100)
  mapcol=next((k for k in d[0] if k=='train/map_loss'),None)
  if mapcol:line(axes[1],d,mapcol,label+' map loss')
axes[0].set_ylabel('Dev AP50–95 (%)');axes[0].legend();axes[1].set_ylabel('Unweighted auxiliary map loss');axes[1].set_title('Configured λ=.25; effective joint λ=.75 (broadcast bug)');save('voc125_curves_and_map_loss',fig)

# Test-time three-strength GT/predicted experiments; separate model/split panels.
def blur_series(root):
 out=[]
 for mode in ['raw','light','medium','heavy']:
  p=root/(mode+'.json');d=load(rel(p))
  # Metric structure differs across early evaluator versions.
  def find(x):
   if isinstance(x,dict):
    for key in ['map50_95','mAP50_95']:
     if isinstance(x.get(key),(int,float)):return {**x,'map50_95':x[key]}
    for v in x.values():
     z=find(v)
     if z:return z
   elif isinstance(x,list):
    for v in x:
     z=find(v)
     if z:return z
  out.append(find(d))
 return out
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for sub,label in [('testtime_predicted_blur_960','Predicted-box protection'),('testtime_groundtruth_blur_960','GT Oracle protection')]:
 ds=blur_series(ROOT/VOC/sub);axes[0].plot([0,3,8,16],[100*d['map50_95'] for d in ds],'-o',label=label)
yroot=ROOT/'archives/local/yolo26_research/voc_yolo26n_testtime_gt_oracle_640/o2o'
if (yroot/'raw.json').exists():
 ds=blur_series(yroot);axes[1].plot([0,3,8,16],[100*d['map50_95'] for d in ds],'-o',label='O2O + GT Oracle')
axes[0].set_title('YOLOv8n / VisDrone val548 / best125');axes[1].set_title('YOLO26n O2O / VOC test4952 / last150')
for ax in axes:ax.set_xlabel('Gaussian σ at model input');ax.set_ylabel('AP50–95 (%)');ax.legend();ax.grid(alpha=.2)
save('testtime_blur_GT_vs_prediction',fig)

# Unified-candidate foveation experiments, count and classwise tables.
fovea=[];fixed=[]
for name in ['FOVEATED_VISDRONE_CROP_PILOT_VAL548_MULTILABEL.json','FOVEATED_SMALL_ONLY_TESTDEV_FULL1610.json','FOVEATED_PREDICTED_BG_BLUR_TESTDEV_FULL1610.json','FOVEATED_GT_ORACLE_VAL548_FULL.json','FOVEATED_CROP_LIGHT_BLUR_VAL548.json']:
 d=load(VOC+name)
 for condition,m in d.get('conditions',{}).items():fovea.append({'source_json':VOC+name,'images':d['images'],'condition':condition,**{k:v for k,v in m.items() if isinstance(v,(int,float))}})
 for condition,classes in d.get('fixed_conf_025_iou_05',{}).items():
  for c,v in classes.items():fixed.append({'source_json':VOC+name,'images':d['images'],'condition':condition,'class':c,**v})
dump('foveation_same_protocol_endpoints.csv',fovea);dump('foveation_fixed_threshold_classwise_TP_FP.csv',fixed)
fig,axes=plt.subplots(1,2,figsize=(13,4.7))
for ax,sel in zip(axes,['FOVEATED_GT_ORACLE_VAL548_FULL.json','FOVEATED_SMALL_ONLY_TESTDEV_FULL1610.json']):
 ds=[r for r in fovea if r['source_json'].endswith(sel)];labels=[r['condition'].replace('full_','').replace('_background','\nbackground').replace('_',' ') for r in ds]
 ax.bar(labels,[100*r['map50_95'] for r in ds]);ax.set_ylabel('AP50–95 (%)');ax.set_ylim(15,25);ax.set_title('val548' if 'VAL548' in sel else 'test-dev1610')
 ax.tick_params(axis='x',labelsize=8)
save('foveation_val_and_testdev_endpoints',fig)
fig,axes=plt.subplots(1,2,figsize=(14,5))
fd=load(VOC+'FOVEATED_GT_ORACLE_VAL548_FULL.json')['fixed_conf_025_iou_05'];names=list(fd['full'])
for cond,label in [('full','Full'),('full_raw4','Raw crops'),('full_predicted_background_blur4','Predicted blur'),('full_gt_oracle_background_blur4','GT blur')]:
 for ax,k in zip(axes,['tp','fp']):ax.plot(names,[fd[cond][n][k] for n in names],'-o',label=label,markersize=3)
for ax,k in zip(axes,['TP','FP']):ax.set_ylabel(k+' at confidence .25, IoU .5');ax.tick_params(axis='x',rotation=60,labelsize=8);ax.legend(fontsize=8)
save('foveation_TP_FP_by_class',fig)

# Public run / source indices are complete, even when a branch did not train.
manifest=rows('results/evidence_manifest.csv')
md=[r for r in manifest if r['path'].endswith('.md')]
code=[r for r in manifest if r['path'].endswith('.py')]
for language,title in [('en','Complete historical document index'),('zh-CN','全部历史文档索引')]:
 content='# '+title+'\n\n'+('Historical snapshots may be superseded. Current chapters and explicit stop/audit records take precedence.\n\n' if language=='en' else '历史快照可能已经被后续审计或停止记录取代。以本次整理的当前状态及明确的最终审计/停止证据为准。\n\n')
 for fam in ['yolo26_research','yolo26n_voc_baseline_e150','voc_pgb_yolov8n','foveated_classification','yolo26','progressive_blur','classification','dataset_baselines','historical','original_pgb_core']:
  subset=[r for r in md if '/'+fam+'/' in r['path']]
  if not subset:continue
  content+='## '+fam+'\n\n'
  for r in subset:content+=f"- [{r['path']}](../../{r['path']})\n"
 (ROOT/'docs'/language/'archive_index.md').write_text(content,encoding='utf-8')
dump('code_index.csv',[{'path':r['path'],'origin':r['origin'],'published_sha256':r['published_sha256']} for r in code])

# Machine-generated bilingual table appendix (same values, translated headings).
zh_columns={'group':'组别','actual_epochs':'实际轮数','train_top1':'训练Top1(%)','train_top5':'训练Top5(%)','train_loss':'训练loss',
 'test_top1':'测试Top1(%)','test_top5':'测试Top5(%)','test_loss':'测试loss','best_test_top1_epoch':'best轮','best_test_top1':'best测试Top1(%)',
 'mean_epoch_seconds':'平均秒/轮','dataset':'数据版本','seed':'种子','seed_from_run_name':'记录种子','run':'原始run/CSV',
 'last_precision':'末轮P','last_recall':'末轮R','last_map50':'末轮AP50','last_map50_95':'末轮AP95','best_epoch':'best轮','best_map50_95':'best AP95',
 'completed_epochs':'完成轮数','detector_head':'评测头','metrics/precision(B)':'P','metrics/recall(B)':'R','metrics/mAP50(B)':'AP50','metrics/mAP50-95(B)':'AP95',
 'images':'评测图片数','condition':'处理条件','precision':'P','recall':'R','map50':'AP50','map50_95':'AP95','predictions':'总预测数',
 'family':'研究阶段','classification':'记录类型','continuous':'轮次连续','args_imgsz':'原args输入','args_batch':'原args batch',
 'train/box_loss':'训练box loss','train/cls_loss':'训练cls loss','train/dfl_loss':'训练dfl loss'}
for language in ['en','zh-CN']:
 cn=language=='zh-CN'
 def ttable(data,cols):return table_md(data,cols,[zh_columns.get(k,k) for k in cols] if cn else None)
 text='# '+('完整训练与指标附录' if cn else 'Complete training and metric appendix')+'\n\n'
 text+=('检测 AP 为 0–1；分类 Top1/Top5 为百分数。每张表明确区分数据划分、checkpoint 和评测口径，不能横向混用不同协议。best 仅为次要结果。\n\n' if cn else 'Detection AP is in [0,1]; classification Top1/Top5 are percentages. Different splits, heads and evaluators must not be compared as a paired experiment. Best is secondary.\n\n')
 text+='## '+('CUB A–F、G4/G5及未完成G6' if cn else 'CUB A–F, G4/G5 and incomplete G6')+'\n\n'+ttable(class_rows,['group','actual_epochs','train_top1','train_top5','train_loss','test_top1','test_top5','test_loss','best_test_top1_epoch','best_test_top1','mean_epoch_seconds'])+'\n\n'
 text+='## '+('22组YOLOv8n从零跨数据集训练' if cn else '22 YOLOv8n scratch dataset runs')+'\n\n'+ttable(suite,['dataset','seed','last_precision','last_recall','last_map50','last_map50_95','best_epoch','best_map50_95'])+'\n\n'
 text+='## '+('YOLO26全部训练末轮记录' if cn else 'YOLO26 recorded training endpoints')+'\n\n'+ttable(y26,['run','completed_epochs','detector_head','metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)'])+'\n\n'
 text+='## '+('冻结模型凹视评测' if cn else 'Frozen-model foveation evaluations')+'\n\n'+ttable(fovea,['images','condition','precision','recall','map50','map50_95','predictions'])+'\n\n'
 text+='## '+('规范CSV登记表（含短测/未完成）' if cn else 'Canonical CSV registry (including smoke / incomplete)')+'\n\n'+ttable(registry,['family','run','completed_epochs','classification','continuous'])+'\n\n'
 text+='## '+('全部服务器CSV末轮数值（含短测，非独立复评）' if cn else 'Every server CSV endpoint (including smoke, not standalone reevaluation)')+'\n\n'
 text+=('种子以run名称/原始args为依据；每个run名称链接到完整CSV。不能把不同预算、smoke或未完成组当作方法对照。\n\n' if cn else 'Seeds follow the run names and original args. Every linked run opens its complete CSV. Different budgets, smoke and incomplete runs are not interchangeable method controls.\n\n')
 for family in ['yolo26','dataset_baselines','progressive_blur','classification']:
  linked=[{**r,'run':f"[{r['run']}](../../{r['source_csv']})"} for r in registry if r['family']==family]
  cols=['run','seed_from_run_name','completed_epochs','args_imgsz','args_batch']
  cols+=['train_top1','train_top5','train_loss','test_top1','test_top5','test_loss'] if family=='classification' else ['detector_head','metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)','train/box_loss','train/cls_loss','train/dfl_loss']
  text+='### '+family+'\n\n'+ttable(linked,cols)+'\n\n'
 text+=('完整逐轮数值与列名保存在 [CSV](../../results/all_canonical_epoch_curves.csv)，JSON 中所有可识别 AP 结果和逐类结果分别保存在 [指标账本](../../results/detection_metric_ledger.csv)、[逐类账本](../../results/detection_classwise_ledger.csv)，TP/FP/覆盖率分母保存在 [诊断账本](../../results/diagnostic_counts_ledger.csv)。源文件与 SHA 见 [证据清单](../../results/evidence_manifest.csv)。\n' if cn else 'All epoch values and original column names: [CSV](../../results/all_canonical_epoch_curves.csv). JSON AP records: [metric ledger](../../results/detection_metric_ledger.csv); per-class AP: [classwise ledger](../../results/detection_classwise_ledger.csv); TP/FP/coverage denominators: [diagnostic ledger](../../results/diagnostic_counts_ledger.csv). Paths and SHA fingerprints: [manifest](../../results/evidence_manifest.csv).\n')
 (ROOT/'docs'/language/'tables.md').write_text(text,encoding='utf-8')
runpy.run_path(str(ROOT/'tools/build_extended_figures.py'),run_name='__main__')
print(json.dumps({'canonical_training_csvs':len(registry),'epoch_rows':len(curves),'json_metric_records':len(metric_ledger),'classwise_records':len(class_ledger),'diagnostic_records':len(numeric_diagnostics),'classification_runs':len(class_rows),'dataset_baselines':len(suite),'historical_md':len(md),'code_files':len(code),'figures_png':len(list(FIG.glob('*.png')))},indent=2))
