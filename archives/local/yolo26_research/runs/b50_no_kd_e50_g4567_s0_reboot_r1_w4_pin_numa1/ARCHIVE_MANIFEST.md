# B50 seed0 same-boot repeat archive manifest

Run: `b50_no_kd_e50_g4567_s0_reboot_r1_w4_pin_numa1`  
Status: **50/50 completed; independent O2O/O2M evaluated; exact-reproduction audit passed**.

## Protocol

- Randomly initialized YOLO26n on VOC07+12 trainval, fixed 50 epochs, seed 0.
- Physical GPUs 4–7, NUMA1, batch 64, BF16, `workers=4` per rank, pinned memory.
- All auxiliary KD/TQGB routes disabled; frozen config SHA256: `1f3588848bca74bf716455bca4a8207fefe939557dc2357a0d5ed4efac9a8632`.
- This control was launched after the projected and non-projected TQGB seed0 results were frozen, so it cannot be a result-conditioned baseline configuration.

## Results

- Fixed epoch-50 P/R/mAP50/mAP50-95: `0.75173/0.64725/0.72389/0.51359`.
- Independent optimizer-stripped `last.pt`: O2O NMS-free `0.5134680219`; O2M+NMS `0.5314237361`.
- Training duration: `0.517 hours`.

## Exact reproduction audit

- Compared with the old seed0 B50 after excluding only the wall-clock `time` column.
- All `5,350` comparable numeric CSV values across all 50 epochs are exactly equal; maximum absolute difference `0`.
- The two optimizer-stripped `last.pt` model states are exactly equal across 708 tensors and 2,530,706 elements:
  - shared/O2M: 588 tensors, 2,403,960 elements;
  - O2O box: 42 tensors, 72,018 elements;
  - O2O class: 78 tensors, 54,728 elements.
- Raw checkpoint/CSV file hashes differ because serialization metadata and wall-clock time differ; this is not a numeric model difference.

## Integrity

- Training and independent-evaluation windows contained no new qualifying MCE/SIGBUS/EDAC event.
- `results.csv` contains contiguous epochs 1–50 and the log records `50 epochs completed in 0.517 hours`.
- Eighteen remote source artifacts totaling 97,375,400 bytes were copied locally and compared directly; every local SHA256 matched its remote source.
- `SHA256SUMS.txt` records the 18 copied artifacts. This manifest is intentionally excluded from that list.
- The host remains historically faulty; all evidence is exploratory until healthy-hardware reproduction.
