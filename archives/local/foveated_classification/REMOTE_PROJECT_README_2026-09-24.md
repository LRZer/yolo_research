# Foveated classification research — server handoff

2026-09-24 22:37 CST. This is a new image-classification research project, separate from the prior object-detection experiments. Server root: `/home/ubuntu/rz/foveated_classification_research`. No training or GPU work has been started for this project.

## Layout

```text
foveated_classification_research/
├── data/CUB-200-2011/
│   ├── original_archives/
│   │   ├── CUB_200_2011.tgz
│   │   └── segmentations.tgz
│   └── extracted/
│       ├── CUB_200_2011/       # images, classes, boxes, parts, attributes, official split
│       └── segmentations/      # optional bird masks, matched by relative path
├── configs/                    # reserved for frozen experiment configs
├── src/                        # reserved for implementation
├── runs/                       # reserved for isolated training/evaluation outputs
├── reports/                    # research plan, data audit, research log
└── _management/audit_cub_official.py
```

The official training images are selected by `data/CUB-200-2011/extracted/CUB_200_2011/train_test_split.txt` value `1` (5,994 images); value `0` is the official test set (5,794 images). A future development split should be created **only within the 5,994 training images**, with its manifest, seed, and hash saved under this project's `configs/`. Do not use test for model selection. The first proposed experiments are GT-box-conditioned Oracle representation studies, not annotation-free deployment.

## Sources and integrity

The two archives came from [CaltechDATA CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) and [CaltechDATA Segmentations](https://data.caltech.edu/records/w9d68-gec53). Their on-server MD5 digests match Caltech's published values and their on-server SHA-256 digests match the local original archives:

| File | Bytes | MD5 | SHA-256 |
|---|---:|---|---|
| `CUB_200_2011.tgz` | 1,150,585,339 | `97eceeb196236b17998738112f37df78` | `0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081` |
| `segmentations.tgz` | 39,272,883 | `4d47ba1228eae64f2fa547c47bc65255` | `dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f` |

Both archives remain untouched. The extracted tree uses about 1.29 GB (logical bytes) in addition to the about 1.19 GB archive files. At handoff, the server volume had about 360 GB free.

## Full server-side audit

Run the read-only audit with:

```bash
python3 /home/ubuntu/rz/foveated_classification_research/_management/audit_cub_official.py /home/ubuntu/rz/foveated_classification_research/data/CUB-200-2011/extracted
```

It verified all 11,788 images, all 11,788 corresponding masks, all 200 classes, matching image/class/box/split IDs, full pixel decoding and identical image/mask dimensions. No missing or undecodable files were found. Important source-data exceptions, also present in the local copy: 8 grayscale JPGs, 2 LA-mode PNGs, and 30 boxes beyond image boundaries (largest excess 32 pixels). Do **not** change the original annotations; any necessary clipping/channel conversion belongs in a documented derived loader and must be tested before training.

The dataset is present and structurally verified, but no classification baseline, foveation experiment, train/dev manifest, or server-specific training environment has been created yet. The proposed experiment design and local data receipt are under `reports/`.
