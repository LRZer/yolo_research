# TQGB T1-NP seed0 archive manifest

Run: `tqgb_t1_np_cls_m2o_g008_e50_g4567_s0_w4_pin_numa1`  
Status: **50/50 completed; independent O2O/O2M evaluated; waiting for same-boot B50 attribution**.

## Protocol

- Randomly initialized YOLO26n on VOC07+12 trainval, fixed 50 epochs, seed 0.
- Physical GPUs 4–7, NUMA1, batch 64, BF16, `workers=4` per rank, pinned memory.
- O2M→O2O assigned-class route gain `0.08`, exact same-GT/same-class correspondence, receiver supervision projection disabled; reverse route and legacy KD disabled.
- Frozen config SHA256: `44103c5a24ef29e0bc85380ce255fa5b48492659ba9e65887546af3de8427ae5`.
- Frozen `train_research.py`/`loss.py`/`default.yaml` SHA256: `d8f39b...` / `6dae8e...` / `1294dd...` (full values recorded in the launch gate and research log).

## Results

- Fixed epoch-50 P/R/mAP50/mAP50-95: `0.74126/0.65567/0.72312/0.51303`.
- Versus old B50 seed0: final `-0.00056`, last-5 `-0.00039`, last-10 `-0.00075` mAP50-95.
- Versus projected T1 seed0: final `+0.00159`, last-5 `+0.00182`, last-10 `+0.00232`.
- Independent optimizer-stripped `last.pt`: O2O `0.51332068`, O2M `0.52876220`.
- Route was active: all-epoch mean route rate `11.9729%`, diagnostic projection survival `95.0039%`, scaled loss `0.001325`.

## Integrity

- Training and independent evaluation windows contained no new qualifying MCE/SIGBUS/EDAC event.
- `results.csv` contains contiguous epochs 1–50 and the log records `50 epochs completed in 0.521 hours`.
- Sixteen remote source artifacts (run, weights, logs, config, and independent eval JSON/logs) totaling 97,383,983 bytes were compared directly; every local SHA256 matched its remote source.
- Eight deterministic local analysis artifacts were added afterward. `SHA256SUMS.txt` records all 24 non-manifest files.
- Host remains historically faulty; all evidence is exploratory until healthy-hardware reproduction.
