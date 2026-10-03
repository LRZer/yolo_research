# TQGB projected T1 seed-0 archive manifest

Run: `tqgb_t1_cls_m2o_proj_g008_e50_g4567_s0_w4_pin_numa1`

Status: **50/50 completed; did not replicate the seed-42 gain; faulty-host exploratory**.

## Protocol

- Randomly initialized YOLO26n, VOC07+12 trainval, fixed 50 epochs, seed 0.
- Physical GPUs 4–7, NUMA node 1 binding, batch 64, BF16, `workers=4` per rank, pinned memory, O2O NMS-free validation.
- TQGB O2M→O2O assigned-class route gain `0.08`, exact same-GT route, receiver supervision-segment projection enabled; reverse route and legacy KD disabled.
- Locked config SHA256: `7a6bbc772c556284e18b2ad67830633b68880d5134eb1c047924adce3f80ff51`.
- Locked `loss.py` SHA256: `6dae8e66c4b94a0c16b11a219e4831994c895cc036210648a6a23d9e282e5db9`.

## Fixed result

The fixed epoch-50 training validation is P/R/mAP50/mAP50-95=`0.73036/0.65809/0.71900/0.51144`. The old seed-0 B50 values are `0.75173/0.64725/0.72389/0.51359`, so the mAP50-95 delta is `-0.00215`. Last-5 and last-10 paired mean deltas are `-0.00220/-0.00306`; T1 wins zero of those 10 epochs.

Independent `last.pt` evaluation confirms O2O `0.51145839` versus B50 `0.51346802` (`-0.00200963`) and O2M `0.52846890` versus B50 `0.53142374` (`-0.00295483`).

## Integrity and hardware

- `results.csv` contains the contiguous epoch sequence 1–50; the log reports `50 epochs completed in 0.521 hours` and contains no matched traceback, runtime, bus, NCCL, OOM, or NaN error.
- Training plus four independent evaluations produced no current-boot `Memory failure`, `Sending SIGBUS`, uncorrectable-memory, or EDAC-UE record after the 04:50 launch.
- The host has a known active RAM fault history, so this remains exploratory even though this window was clean. The old B50 seed-0 reference was hardware-contaminated and occurred before a reboot; a same-boot baseline repeat is therefore mandatory before strong seed-0 attribution.
- All 19 copied source artifacts total 97,434,923 bytes; every local SHA256 in `SHA256SUMS.txt` matches the remote digest.
