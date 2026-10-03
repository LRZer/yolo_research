"""Validate public evidence and derived reports. Never runs training.

The checks confirm archived bytes and saved audit records, not unavailable
checkpoint binaries. Historical Markdown absolute links are preserved provenance;
only current curated documentation is required to have portable local links.
"""
from pathlib import Path
import csv, gzip, hashlib, json, math, re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
errors=[]
def require(test,msg):
 if not test:errors.append(msg)
def rows(p):
 with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:
  return [{k.strip():v for k,v in r.items() if k is not None} for r in csv.DictReader(f)]
def digest(p,compressed=False):
 h=hashlib.sha256()
 with (gzip.open(p,'rb') if compressed else p.open('rb')) as f:
  while block:=f.read(1024*1024):h.update(block)
 return h.hexdigest()

manifest=rows('results/evidence_manifest.csv')
require(len(manifest)==1812,'Expected1812 archived evidence files')
seen=set();large=0
for r in manifest:
 path=r['path'];require(path not in seen,'Duplicate manifest path: '+path);seen.add(path)
 p=ROOT/path
 if not p.is_file():errors.append('Missing evidence: '+path);continue
 require(p.stat().st_size==int(r['bytes']),'Size mismatch: '+path)
 require(digest(p)==r['published_sha256'],'Published SHA mismatch: '+path)
 if path.endswith('.gz'):
  large+=1;require(digest(p,True)==r['source_sha256'],'Uncompressed SHA mismatch: '+path)
require(large==4,'Expected4 lossless numerical gzip records')

# All archive/code evidence must be manifested, not just counted.
for directory in ['archives','code']:
 for p in (ROOT/directory).rglob('*'):
  if p.is_file():require(p.relative_to(ROOT).as_posix() in seen,'Unmanifested source: '+str(p))

registry=rows('results/training_run_registry.csv')
require(len(registry)==96,'Expected96 canonical training CSVs')
require(sum(int(r['completed_epochs']) for r in registry)==6056,'Expected6056 epoch rows')
all_epochs=rows('results/all_canonical_epoch_curves.csv')
require(len(all_epochs)==6056,'Aggregated epoch rows differ')
for r in registry:
 data=rows(r['source_csv'])
 require(len(data)==int(r['completed_epochs']),'Registry row count mismatch: '+r['run'])
 epochs=[int(float(x['epoch'])) for x in data]
 continuous=epochs==list(range(1,epochs[-1]+1))
 require(str(continuous)==r['continuous'],'Continuity flag mismatch: '+r['run'])

suite=rows('results/dataset_22_run_endpoints.csv')
require(len(suite)==22,'Expected22 cross-domain baselines')
require(len({r['dataset'] for r in suite})==11,'Expected11 dataset versions')
for r in suite:
 data=rows(r['source_csv'])
 require([int(float(x['epoch'])) for x in data]==list(range(1,126)),'Suite is not125 continuous: '+r['dataset'])
 for x in data:
  for k,v in x.items():
   if v:require(math.isfinite(float(v)),'Nonfinite suite field: '+r['dataset']+'/'+k)
 require(int(r['seed']) in [2005,2027],'Unexpected suite seed')

classification=rows('results/cub_primary_and_secondary_endpoints.csv')
require({r['group'] for r in classification}==set('ABCDEF')|{'G4','G5','G6'},'CUB groups mismatch')
for r in classification:
 group=r['group'];expected=50 if group=='G6' else 125;data=rows(r['source_csv'])
 require([int(float(x['epoch'])) for x in data]==list(range(1,expected+1)),'CUB continuity: '+group)
 require(int(r['seed'])==2005 and int(r['lr_horizon_epochs'])==200,'CUB seed/LR horizon: '+group)
 for x in data:
  for k in ['lr','train_loss','train_top1','train_top5','test_loss','test_top1','test_top5','train_seconds','test_seconds','epoch_seconds']:
   require(math.isfinite(float(x[k])),'Nonfinite CUB: '+group+'/'+k)
  for k,count in [('train_images',5994),('test_images',5794)]:
   if k in x:require(int(x[k])==count,'Incorrect CUB sample count: '+group+'/'+k)
 last=data[-1]
 for k in ['train_top1','train_top5','train_loss','test_top1','test_top5','test_loss','lr']:
  require(float(last[k])==float(r[k]),'CUB endpoint mismatch: '+group+'/'+k)
 best=max(data,key=lambda x:float(x['test_top1']))
 require(int(best['epoch'])==int(r['best_test_top1_epoch']),'CUB best epoch mismatch: '+group)
 require(float(best['test_top1'])==float(r['best_test_top1']),'CUB best Top1 mismatch: '+group)
 if group!='G6':require(abs(float(last['lr'])-.008074211936403517)<1e-14,'CUB125 schedule mismatch: '+group)
 if group in 'ABCDEF':
  audit=json.loads((ROOT/Path(r['source_csv']).parent/'audit.json').read_text(encoding='utf-8'))
  require(audit['status']=='verified_revised_e125','Saved CUB audit not valid: '+group)
stop=json.loads((ROOT/'archives/remote/classification/reports/G6_splitstats_DDP23_2026-10-02_USER_STOP.json').read_text(encoding='utf-8'))
require('stop' in json.dumps(stop).lower(),'G6 authoritative stop record missing')

require(len(rows('results/cub_all_200_class_counts.csv'))==200,'CUB class count rows')
for name,count in [('detection_metric_ledger.csv',1140),('detection_classwise_ledger.csv',1581),('diagnostic_counts_ledger.csv',1289)]:
 require(len(rows('results/'+name))==count,'Ledger count mismatch: '+name)
require(len(rows('results/code_index.csv'))==362,'Expected362 archived Python files')
require(len(list((ROOT/'figures').glob('*.png')))==21,'Expected21 figure panels')
for p in (ROOT/'figures').glob('*.png'):require(p.with_suffix('.svg').is_file(),'SVG missing: '+p.name)

# Portable links and case are required for current editions, not old server notes.
files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
exact={p.relative_to(ROOT).as_posix() for p in files}
directories={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_dir() and '.git' not in p.parts}
curated=[ROOT/'README.md',ROOT/'README.zh-CN.md',ROOT/'PUBLICATION_SCOPE.md',ROOT/'LICENSING.md']+list((ROOT/'docs').rglob('*.md'))
links_checked=0
for p in curated:
 text=p.read_text(encoding='utf-8')
 for link in re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)',text):
  link=link.strip().strip('<>')
  if re.match(r'^(?:https?:|mailto:|#)',link):continue
  link=unquote(link.split('#',1)[0])
  if not link:continue
  links_checked+=1;target=(p.parent/link).resolve()
  try:key=target.relative_to(ROOT).as_posix()
  except ValueError:errors.append('Link leaves repository: '+str(p)+' -> '+link);continue
  require(key in exact or key in directories,'Broken/case-mismatched local link: '+p.relative_to(ROOT).as_posix()+' -> '+link)

# No private material or oversized GitHub object, even if it escaped manifest.
secret=re.compile(r'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
for p in files:
 key=p.relative_to(ROOT).as_posix()
 require(p.stat().st_size<100*1024*1024,'File exceeds GitHub100MiB: '+key)
 require(p.suffix.lower() not in {'.pt','.pth','.onnx','.engine','.pem','.key','.pdf'},'Excluded binary/credential found: '+key)
 if p.suffix.lower() in {'.py','.md','.csv','.json','.jsonl','.yaml','.yml','.toml','.txt'}:
  require(not secret.search(p.read_text(encoding='utf-8')),'Credential-pattern finding: '+key)

report={'status':'PASS' if not errors else 'FAIL','archived_evidence_files':len(manifest),
 'canonical_training_csvs':len(registry),'canonical_epoch_rows':len(all_epochs),
 'dataset_baselines':len(suite),'cub_completed125':8,'cub_stopped50':1,
 'lossless_large_records_verified':large,'archived_python_files':362,
 'historical_markdown':374,'figure_panels_png_svg':21,'curated_local_links_checked':links_checked,
 'checkpoint_binaries_reloaded':False,
 'checkpoint_scope':'Saved server loading audits are preserved; weight binaries are not redistributed or reloaded by this validator.',
 'errors':errors}
(ROOT/'results/publication_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
