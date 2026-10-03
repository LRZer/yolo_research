# VOC P1 per-class raw-development AP analysis

Date: 2026-09-23. All four already-selected `best.pt` checkpoints were
evaluated on the same 1,655 untouched VOC development images, 640 input,
same YOLO evaluator. AP is averaged over IoU 0.50:0.95. This re-evaluation
did **not** select new checkpoints or use VOC2007 test.

| Class | Scratch baseline | Weak 100% | Strong 100% | Strong 50% |
|---|---:|---:|---:|---:|
| aeroplane | .346 | .325 | .299 | .322 |
| bicycle | .252 | .260 | .238 | .258 |
| bird | .080 | .087 | .082 | .081 |
| boat | .054 | .059 | .068 | .062 |
| bottle | .024 | .029 | .016 | .025 |
| bus | .336 | .373 | .362 | .363 |
| car | .239 | .243 | .239 | .255 |
| cat | .187 | .212 | .190 | .168 |
| chair | .038 | .037 | .045 | .046 |
| cow | .079 | .080 | .094 | .092 |
| diningtable | .075 | .063 | .072 | .078 |
| dog | .205 | .194 | .176 | .201 |
| horse | .169 | .174 | .178 | .184 |
| motorbike | .233 | .237 | .235 | .211 |
| person | .244 | .231 | .235 | .237 |
| pottedplant | .025 | .025 | .030 | .031 |
| sheep | .107 | .123 | .143 | .123 |
| sofa | .105 | .081 | .088 | .110 |
| train | .234 | .210 | .171 | .181 |
| tvmonitor | .204 | .183 | .187 | .206 |

The rounded table is for visual comparison; exact values and checkpoint
hashes are in `P1_CLASSWISE_RAW_VALIDATION.json`. The strong-100% aggregate
loss is not uniform across classes: train and aeroplane drop roughly 0.06
and 0.05 absolute AP, while sheep and bus rise roughly 0.04 and 0.03.
Mixing in raw views restores aggregate AP mainly through several offsetting
changes, not every class: train remains about 0.05 below baseline while
car and bus are about 0.02 and 0.03 above. These are one-seed, short-run,
per-class exploratory signals. We have not established whether background
context, image size, class frequency, or training variance caused them.
Object counts and image-level failure cases would be needed to test such
mechanisms. The aggregate result remains a near-tie with baseline, not
a method win.
