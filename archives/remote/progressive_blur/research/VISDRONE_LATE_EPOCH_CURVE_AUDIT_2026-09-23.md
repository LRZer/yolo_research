# VisDrone standard-640 / P2-640 / standard-960 training-curve audit

Date: 2026-09-23. CPU-only read of three already-audited 125-row
`results.csv` files; no new training, inference, checkpoint selection or
GPU usage. Values below are the **built-in raw-val** `metrics/mAP50-95(B)`
recorded per epoch, not independent re-evaluation. The independent
best-checkpoint values and class/size diagnostics remain in
`VISDRONE_P2_640_CONTROL_RESULT_2026-09-23.md`.

| Epoch | Standard 640 | P2 640 | Standard 960 |
|---:|---:|---:|---:|
| 20 | 0.11136 | 0.12907 | 0.15792 |
| 50 | 0.14825 | 0.17005 | 0.20826 |
| 75 | 0.16276 | 0.18585 | 0.22123 |
| 100 | 0.16936 | 0.19580 | 0.23057 |
| 125 | 0.16999 | 0.19689 | 0.23167 |

The ordering `standard960 > P2-640 > standard640` holds on **114 of
116 epochs 10–125**. Only epochs 12 and 14 have the standard-640 curve
slightly above P2-640; standard960 remains highest there. The three
epoch100→125 gains are **+0.00063**, **+0.00109**, and **+0.00110**
respectively. From epoch105→125 they are +0.00069, +0.00006 and
+0.00004. Best recorded epochs were 115, 115 and 111, respectively;
the last-10-epoch mean APs are 0.169990, 0.197117 and 0.231398.
At epoch125, standard960 still exceeds P2-640 by **0.03478** internal
AP50–95. Its independently re-evaluated best-checkpoint advantage is
0.033950.

**Interpretation:** within this seed, data split and 125-epoch schedule,
P2 did not close the resolution gap at any mature stage, and all curves
show small late gains. It is therefore weak to explain the P2/960 gap
as just an unfinished 125-epoch run. This is **not** proof that a much
longer or differently tuned schedule could not change the ranking;
nor is it a multiple-seed uncertainty estimate. Built-in AP curves
also cannot identify whether 960 wins through fewer false positives,
better confidence ordering, class discrimination or box localization.
The fixed low-confidence GT recall parity previously observed for P2
and 960 at IoU .50 is compatible with several of these explanations.
Future evaluation should explicitly decompose PR curves and matched
errors on an identical raw-val prediction cache when safe hardware is
available, rather than introducing blur as an assumed fix.

Inputs: `VISDRONE_OFFICIAL_640_S2027_RESULTS.csv`,
`VISDRONE_OFFICIAL_P2_640_S2027_RESULTS.csv`, and
`VISDRONE_OFFICIAL_960_S2027_RESULTS.csv`. No test-dev result was used.
