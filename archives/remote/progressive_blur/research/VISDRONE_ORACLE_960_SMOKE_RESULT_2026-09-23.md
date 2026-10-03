# VisDrone 960 train-only Oracle strong-blur smoke — 2026-09-23

This is a **functional smoke only**, not an AP experiment. It followed
the pre-result hardware-gate amendment in
`VISDRONE_TRAIN_ONLY_BLUR_960_PROTOCOL_2026-09-23.md`: one epoch,
1% training fraction, seed2027, batch64, physical GPU4–7,
`numactl --cpunodebind=0 --membind=0`, exact target construction,
Gaussian levels 0/2/4/8/16, mix0.5 and train probability1.0. The
smoke config has zero warmup/ramp, so epoch index0 uses mix0.5; this
was checked directly with the trainer schedule and the locked config.
No changes were made to shared data or processes.

## Actual result and code-path evidence

- Manager returned `verified_complete` at **06:35:39 CST** with exactly
  one finite CSV epoch and readable `best.pt`/`last.pt`; saved state is
  `VISDRONE_ORACLE_960_SMOKE_STATE.json`. The task-owned log reports
  65 train images, **2/2 train batches**, 548 raw-val images and
  38,759 converted val boxes. It launched 4 DDP processes in the
  physical-GPU4–7 visibility mask and completed in ~18 seconds
  wall time; rank-0 training log reached about 5.1 GiB `GPU_mem`.
- Entrypoint `run_experiment.py` maps `oracle_train_only` to
  `OracleTrainOnlyBlurTrainer`. That class inherits
  `OracleBlurTrainer.preprocess_batch`, which calls
  `apply_stochastic_oracle_blur`; the smoke config has probability1,
  zero warmup and zero ramp, so all selected training images take the
  exact GT-guided blur branch at mix0.5. Its overridden
  `get_validator()` returns the ordinary `DetectionValidator`, not
  `OracleBlurValidator`, so validation remains on raw images.
  Ultralytics 8.4.146's DDP generator serializes the concrete trainer
  class and `vars(trainer.args)` into its temporary worker state; the
  smoke DDP startup therefore preserves the subclass and custom
  options. This is **code-path verification**, not a saved pixelwise
  training-batch trace. The earlier CPU functional smoke checked
  finite/changed 960-sized output.
- The smoke's tiny training subset yields near-zero val AP, which is
  deliberately **not** compared with the 125-epoch baseline and does
  not support any method claim.

## Hardware and next gate

Before launch, EDAC was `mc0=0/mc1=12`, unchanged for about 35
minutes after the last 05:58 node1-mapped burst; boot, idle GPUs,
absence of task processes/run directory and remote config hashes
were checked. Immediately after smoke, EDAC was still `0/12`, no
matching kernel MCE/SIGBUS appeared since 06:33, and physical
GPUs4–7 returned to 2 MiB/0%. The machine is still **unhealthy**;
one completed smoke does not validate 125 epochs or guarantee safety.
Formal launch requires a separate fresh low-frequency trend check,
idle devices/no task process, a valid formal config, and explicit
logging of the decision. If a new UE or task SIGBUS appears, stop new
GPU work and do not brute-force retries. Any eventual result must be
rerun on healthy hardware before publication-level claims.
