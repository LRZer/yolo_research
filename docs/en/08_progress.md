# 8. Timeline, deliverables and unperformed work

[Contents](../../README.md) · [中文](../zh-CN/08_progress.md) · [Every historical document](archive_index.md)

## Chronological research progression

| Date (Asia/Shanghai) | Actual work | Endpoint/status |
|---|---|---|
| Sep17–18 | Recovered mature scratch YOLO26n/VOC150 baseline; audited O2M/O2O, dataset scale and pinned source; D0 gain-zero reproduction | Baseline and numerical reproduction completed |
| Sep18–19 | D1/D2, KD calibration, loss/gradient compatibility, clipping diagnostics; hardware-failed starts retained; D2-GM and S-QGAT | Small original D2 gain erased by scale matching |
| Sep19 | TQGB T0/T1/T1-NP/T2,second seed and same-boot repeat; native-F1 retrospective audit | Two-seed confirmation failed; route closed |
| Sep19 | PBGA,scale probes,SARC staging/real-source CPU integration,IGB implementation,FCA full graph audit | Naive reattachment and clean-VOC coverage route rejected; SARC/IGB no formal training |
| Sep19–20 | ASAC P0,golden/DDP gates,interrupted invalid start,retry/live state audit,paired A2/A0 formal50 | Both valid runs complete; live controller lost0.510AP point |
| Sep20–22 | Cross-domain data screening,downloads,transfer,conversion,integrity/class/geometry audits | Ten source datasets,eleven views; JODD not obtained |
| Sep22 |22 scratch YOLOv8n125 runs,seeds2005/2027,independent checkpoint/CSV audit |22/22 completed; all epoch data retained |
| Sep23 | PGB core integration,smoke,P1 scratch20 baseline/learned/Oracle; strong previews,blackout; train-only weak/strong/p50 controls | Learned/raw generalization not improved; blackout severely failed on raw |
| Sep23–24 | VisDrone official-view640/P2/960 controls,train-only Oracle125,predicted vsGT inference sigma3/8/16; YOLO26 O2O Oracle150 evaluation | GT privileged signal present; predicted blur worsened |
| Sep24 early morning | Safe-background warmup/pilots,matched125 additional epochs,strict standalone dev reevaluation,gate/paired recall/cost | Negative; aux coefficient broadcast defect disclosed |
| Sep24 morning | SET HBS-only vsidentity125,strict validation,size recall,cost | Negative; full SET not implemented |
| Sep24 | Four priority paper review,frozen adaptive/fixed4 crop pilot,corrected multi-label candidates,provenance/geometry/FP and mask-area audits | More calls/lower AP; evaluator error corrected |
| Sep24–25 | Small-only candidate filtering,test-dev predicted-background blur,one fixedval GT-mask diagnostic | At most tiny inconsistent AP change; crop route rejected |
| Sep24–25 | New classification question,CUB main+mask download/transfer/profile,BCD approved previews; A–E sequential training,200→actual125 revision,D reboot recovery | All five strictly audited; C/D Oracle positive |
| Sep30–Oct1 | Scharr preview approval,environment recovery/resource checks,F sharedGPU6 training125/audit | Stand-alone edge representation negative |
| Oct1 | GT local-crop dimension previews,global+crop vsblank+crop model,CPU/AMP/DDP/full sampler tests,G4/G5 concurrent-sequence execution onGPU2/3 | Both125 audited; G5>G4 by12.409Top1 points |
| Oct2 | Split-statistics BN design,CPU/normalization/DDP smoke,G6 launch,G4 initialization-control checks | User stop at50,not a full ablation |
| Oct2–3 | Task stop verification and GPU resource inspection; contemplated GPU6 rerun | No GPU6 rerun launched |
| Oct4 | Whole-history recovery,server numerical export,source and hash manifest,bilingual synthesis/tables/plots | Publication only; no new training |

Individual date-stamped events may overlap; the [374-research-document index](archive_index.md) and raw logs preserve precise timestamps. A report's existence does not prove that its proposed future experiment was run.

## Delivered research and engineering assets

- Head-specific mature YOLO26 and 22 cross-domain baseline trajectories.
- Quality-transfer variants and causal controls, assignment/gradient/protection/false-positive numerical diagnostics.
- Differentiable initial PGB and integrated larger safe-background module; train-only Oracle/blackout/strength controls.
- HBS-only training-branch port with appropriate identity-extra-loss comparator, not a claimed original method.
- Frozen foveated inference and fusion implementations, candidate-policy validation, per-class TP/FP, mask eligibility and transfer analyses.
- CUB native500 six-representation benchmark, accurate input transformations, reference-mask audit, full train/test curves and strict checkpoint recovery.
- Two-view shared-backbone classification, exact distributed sampler/gradient accumulation, shared/statistics-split normalization code and user-stop provenance.
- Source/config/data/checkpoint SHA fingerprints, unit/smoke tests, launch/recovery/audit scripts and an evidence-based history of no-go decisions.
- Bilingual current research chapters, full raw historical records, curated tables, source-indexed figures and reproducible publication tooling.

These are implemented/artifact-level outputs. A completed paper, patent, accepted workshop contribution, SOTA improvement and deployable annotation-free foveated classifier are **not** recorded accomplishments.

## Unperformed or incomplete experiments

| Branch | Not completed | Why this matters |
|---|---|---|
| TQGB | Further seeds/150epochs/COCO; BRC candidate training | Two-seed short-budget result did not justify confirmation |
| SARC | Formal R1/R2/R3 GPU training/deployment/export validation | CPU integration/fusion is not AP evidence |
| IGB | Real-model CUDA VJP panel and quota-correction training | Implemented tests do not establish proposed mechanism |
| FCA | Coverage-first matcher training | Mechanism effect below preregistered gate |
| ASAC | A1 matched schedule,A3 shuffled feedback,additional seeds,long budgets | A2 vsA0 failed gate |
| Initial PGB | Complete identity/single-scale/no-map/shuffled-map factorial | Planned controls must not be claimed as executed |
| YOLO26 GT blur | Complete O2M light/medium/heavy series | Only O2O three-strength series is complete |
| Classification | Oxford-IIIT Pet,radial/random/radius sweep,nonuniform compute,learned fixation,multiglimpse,segmentation model | Current results establish only narrowOracle input interventions |
| Global/local | Proper single-local model,multiple seeds,batch-count-matched BN controls | G5 includes an active zero branch |
| G6 |75 remaining epochs,complete125 audit,requested GPU6 restart | Explicit user-stopped50; not automatic failure or completednegative result |
| Deployment | Blind annotation-free test,controlled end-to-end cost,multiple model families | Still needed before practical/publication claims |

## Evidence navigation

Use current chapters for final interpretation; the [archive index](archive_index.md) for every prior plan/report; [training registry](../../results/training_run_registry.csv) for actual CSV-bearing runs; [all epochs](../../results/all_canonical_epoch_curves.csv) for curves; [metric/per-class ledgers](../../results/detection_metric_ledger.csv) for JSON contexts; [code index](../../results/code_index.csv) for implementation files; [manifest](../../results/evidence_manifest.csv) for byte provenance. Historical snapshots remain original-language evidence, while both current narrative editions cover the full chronology and scientific boundaries.
