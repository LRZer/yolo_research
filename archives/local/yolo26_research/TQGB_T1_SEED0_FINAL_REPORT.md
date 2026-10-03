# TQGB projected T1 seed-0 final report

Date: 2026-09-19  
Decision: **negative replication; TQGB is not yet a stable improvement**

## Main result

The predeclared primary comparison is the fixed epoch 50, not a selected intermediate checkpoint.

| run | P | R | mAP50 | mAP50-95 | ΔmAP50-95 |
|---|---:|---:|---:|---:|---:|
| B50 seed0, old reference | 0.75173 | 0.64725 | 0.72389 | 0.51359 | — |
| projected T1 seed0 | 0.73036 | 0.65809 | 0.71900 | 0.51144 | **-0.00215** |

The method trades precision for recall but loses both mAP measures. Its last-5 paired mean is `-0.00220`; its last-10 paired mean is `-0.00306`, with 0/10 epoch wins. Therefore the fixed result is not a single unlucky final-point spike.

Independent evaluation of the optimizer-stripped `last.pt` gives the same verdict:

| path | B50 seed0 | projected T1 seed0 | delta |
|---|---:|---:|---:|
| O2O NMS-free | 0.51346802 | 0.51145839 | **-0.00200963** |
| O2M + NMS | 0.53142374 | 0.52846890 | **-0.00295483** |

## The route really ran

The negative result is not caused by a disabled mechanism. Across 50 epochs, mean common-anchor count is `67.648` per batch, mean M2O route count `8.337`, route rate `12.447%`, projection survival `94.367%`, and scaled auxiliary classification loss `0.001350`. Over the final 10 epochs the route rate is `14.320%`, projection survival `98.150%`, and scaled loss `0.001494`.

## Cross-seed interpretation

| seed | B50 fixed e50 | projected T1 fixed e50 | paired delta |
|---:|---:|---:|---:|
| 42 | 0.51276 | 0.51946 | +0.00670 |
| 0 | 0.51359 | 0.51144 | -0.00215 |

The two-seed mean delta is only `+0.002275`, the sample standard deviation is approximately `0.00626`, and only 1/2 pairs is positive. This fails the predeclared multi-seed requirement; the seed-42 gain cannot be called stable or publishable.

The O2M path also changes sign across seeds: seed 42 improved by about `+0.00519`, while seed 0 fell by `-0.00295`. Because the explicit route targets the O2O classification head, this shared/O2M drift is consistent with the known global-gradient-clipping trajectory coupling. It does not prove that clipping caused the entire effect, but it blocks any clean claim that projected routing alone improved the shared representation. The prepared BRC clipping control remains mechanistically relevant, but it must not be portrayed as a rescue experiment chosen from test performance.

## Evidence qualification and next action

This training/evaluation window had no matched MCE/SIGBUS/EDAC event, but the host is historically faulty. In addition, the old B50 seed0 reference ran before the reboot and had deferred EDAC UE events. The tiny epoch-1 drift while TQGB route loss was still zero shows the pair is not bitwise same-boot evidence. Therefore:

1. Continue the already frozen T1-NP seed0 run to test whether projection is better than the no-projection route under the same boot.
2. Run the frozen same-boot B50 seed0 repeat to quantify restart/run-to-run drift.
3. Do not tune route thresholds or gain on VOC07 test.
4. Use seed1 plus the same-boot repeat to decide whether TQGB stops or earns a BRC/150-epoch escalation.
5. In parallel, preserve the independent IGB/SARC routes so a negative TQGB result triggers a real mechanism pivot rather than cosmetic tuning.

Full run, logs, evaluations, weights, manifest, and hashes are under `outputs/yolo26_research/runs/tqgb_t1_cls_m2o_proj_g008_e50_g4567_s0_w4_pin_numa1/`.
