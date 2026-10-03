# Pre-registration: strong Oracle blur on 50% of training images

Recorded 2026-09-23 before formal training. This is an exploratory
diagnostic, not a claim of paper novelty.

## Question and one-variable contrast

Does mixing exactly clean images into GT-protected, strongly blurred
training batches reduce the train-versus-raw-evaluation appearance gap?
Compare the existing strong-100% run with this strong-50% run. Only
`oracle_train_probability` changes from 1.0 to 0.5; Gaussian levels stay
0/2/4/8/16, maximum residual mix stays 0.5, protected-box construction,
seed, model initialization, data, optimizer, batch, epochs and training
schedule stay fixed. Per-image Bernoulli selection is sampled anew in each
training batch; nonselected images remain numerically identical to original
input. No GT-based processing occurs during any validation or prediction.

## Fixed setup and checks

Scratch YOLOv8n, VOC P1 14,896 train / 1,655 development, seed 2027,
20 epochs, batch 64, 640 input, workers 4, no ordinary augmentation,
five clean-start epochs then five ramp epochs. Use only physical GPUs 4–7,
NUMA 0, task-specific run directory and manager. Audit 20 finite CSV rows,
epoch continuity and readable best/last checkpoints. Same standalone
raw-image evaluator for baseline, strong-100%, and strong-50%. Primary
metric: mAP50–95 on raw development images using the built-in raw-selected
`best.pt`; secondary: mAP50, precision, recall and last checkpoint. The
original VOC07 test is not used for this decision.

## Decision rule

If the 50% run does not clearly exceed the scratch baseline in this pilot,
do not pursue more sigma/probability tuning on VOC P1. If it does, repeat
with another seed and longer training before any efficacy or novelty claim.
Even a pilot gain would establish only an augmentation effect, not a new
deployable foreground predictor; evaluate runtime and existing blur
augmentation precedents before a paper framing.

Implementation: `pgb/oracle_integration.py` per-image stochastic branch;
configuration: `configs/voc_oracle_strong_p50_trainonly_p1_e20.json`.
20 unit tests passed, and one 4-GPU, 1%-fraction smoke epoch completed before
the formal run. The smoke confirms the distributed path and raw validator,
not AP efficacy.
