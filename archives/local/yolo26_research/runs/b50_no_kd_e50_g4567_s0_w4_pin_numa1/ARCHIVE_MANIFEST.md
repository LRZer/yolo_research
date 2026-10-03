# B50 seed-0 archive manifest

Run: `b50_no_kd_e50_g4567_s0_w4_pin_numa1`

Status: **50/50 epochs completed, hardware-contaminated / faulty-host exploratory**.

## Source and protocol

- Remote run: `/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/b50_no_kd_e50_g4567_s0_w4_pin_numa1`
- Remote training log: `/home/ubuntu/rz/yolo_study/yolo26_voc_research/logs/b50_no_kd_e50_g4567_s0_w4_pin_numa1.log`
- Remote launch config: `/home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/research/b50_no_kd_e50_s0_w4_pin_numa1.yaml`
- Protocol: YOLO26n, random initialization, VOC trainval, 50 epochs, seed 0, batch 64, image size 640, physical GPUs 4–7, `workers=4` per rank (16 total workers), pinned-memory launch policy, BF16, deterministic mode, `rect=false`, `multi_scale=0`, `fraction=1.0`, and all KD/TQGB gains zero.

## Completion and fixed metrics

- `results.csv` contains exactly 50 data rows with the contiguous epoch sequence 1–50.
- The training log records `50 epochs completed in 0.520 hours.`
- The fixed epoch-50 O2O metrics are P/R/mAP50/mAP50-95 = `0.75173/0.64725/0.72389/0.51359`.
- Epochs 46–50 mAP50-95 values are `0.51158, 0.51175, 0.51290, 0.51269, 0.51359`; their mean is `0.512502`.
- Epochs 46–50 mean P/R/mAP50 are `0.739854/0.654162/0.723438`.
- `last.pt` and `best.pt` were optimizer-stripped and the final best-checkpoint validation completed.
- The original trainer/DDP/loader processes were absent after completion. A contemporaneous NVML query before the later reboot reported no compute process on physical GPUs 4–7.
- Remote timestamps: `results.csv` 04:42:57.283 +08:00, `last.pt` 04:42:57.540 +08:00, and training log 04:43:00.074 +08:00 on 2026-09-19.

## Integrity

The eleven entries in `SHA256SUMS.txt` were computed on the remote host and recomputed after local archival; every digest and byte size matched. The archive contains the complete run directory plus the separately stored training log and launch config.

## Hardware qualification

This is not a clean hardware window. Five EDAC uncorrectable/deferred memory events occurred while the trainer was alive at 04:36:24–04:36:25. No `Sending SIGBUS` record or training-log exception was observed, and training completed, but silent influence cannot be excluded. See `HARDWARE_CONTAMINATION.md`.

The host rebooted at 04:47:05, about four minutes after the completed artifacts were written. The reboot erased the volatile previous-boot journal, but did not alter the archived run files: their post-reboot hashes match the hashes captured before archival.
