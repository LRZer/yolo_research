# ASAC assignment-stability probe

**Decision: GO** — both native heads show intensity-sensitive and quality-relevant affine assignment stability.

> Fixed train-only panel; deterministic scale/translate views; inference-only fixed checkpoints; no optimizer, backward pass, BN-stat update, validation/test AP, or policy tuning.

Panel: `256` images, `4` batches, fingerprint `5bc7260afc2919b071948b2e496f017cdc72d9f5d424bb62cd7f02705dea8266`.

Frozen image+label content: `8636cc2983ff65e77de6351629817fb43d75ebb36a440df44e4026c13747a0fa`.

Protocol: `PREREGISTERED`; trajectory `0e6b09538be74edefb21185dc1edca18541fc454d5d87e365f2b6e0abdde9292`; source bundle `e978653fe83b8825188c1476d3eb2e7fb6c91186e0a2e423bf9978caed2e3231`.

## Geometry survival

Clean GT: `657`; all-view intersection: `624` (`94.98%`).

| q | retained | rate |
|---:|---:|---:|
| 0.25 | 647 | 98.48% |
| 0.50 | 634 | 96.50% |
| 0.75 | 624 | 94.98% |

Retained/dropped GT area and boundary-distance quantiles are in JSON; per-GT survival flags are in the geometry CSV.

## Stability summaries

| epoch | head | q | objects | coverage | mean S | matched JS | Q clean→view | Q agreement | rho(S,Q-retain) | level flip |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | O2M | 0.25 | 624 | 100.00% | 0.9668 | 0.0358 | 0.4598→0.4622 | 0.8292 | 0.2545 | 0.0673 |
| 11 | O2M | 0.50 | 624 | 100.00% | 0.9405 | 0.0685 | 0.4598→0.4564 | 0.8132 | 0.3026 | 0.1202 |
| 11 | O2M | 0.75 | 624 | 99.84% | 0.9140 | 0.1008 | 0.4598→0.4607 | 0.7921 | 0.2220 | 0.1621 |
| 11 | O2O | 0.25 | 624 | 100.00% | 0.9190 | 0.1089 | 0.3828→0.3781 | 0.7183 | 0.2959 | 0.0673 |
| 11 | O2O | 0.50 | 624 | 100.00% | 0.8965 | 0.1430 | 0.3828→0.3816 | 0.6873 | 0.2955 | 0.0978 |
| 11 | O2O | 0.75 | 624 | 100.00% | 0.8884 | 0.1554 | 0.3828→0.3782 | 0.6796 | 0.3139 | 0.1106 |
| 21 | O2M | 0.25 | 624 | 99.84% | 0.9668 | 0.0339 | 0.5628→0.5605 | 0.8665 | 0.2926 | 0.0690 |
| 21 | O2M | 0.50 | 624 | 99.84% | 0.9466 | 0.0579 | 0.5628→0.5603 | 0.8463 | 0.3401 | 0.0979 |
| 21 | O2M | 0.75 | 624 | 99.84% | 0.9183 | 0.0958 | 0.5628→0.5559 | 0.8270 | 0.3163 | 0.1300 |
| 21 | O2O | 0.25 | 624 | 100.00% | 0.9223 | 0.1054 | 0.4974→0.4944 | 0.7612 | 0.3259 | 0.0689 |
| 21 | O2O | 0.50 | 624 | 100.00% | 0.9068 | 0.1270 | 0.4974→0.4965 | 0.7362 | 0.3589 | 0.0833 |
| 21 | O2O | 0.75 | 624 | 100.00% | 0.8864 | 0.1582 | 0.4974→0.4852 | 0.7140 | 0.3372 | 0.1122 |
| 31 | O2M | 0.25 | 624 | 100.00% | 0.9705 | 0.0313 | 0.6313→0.6342 | 0.8859 | 0.3422 | 0.0561 |
| 31 | O2M | 0.50 | 624 | 100.00% | 0.9508 | 0.0543 | 0.6313→0.6309 | 0.8755 | 0.2519 | 0.0913 |
| 31 | O2M | 0.75 | 624 | 100.00% | 0.9287 | 0.0817 | 0.6313→0.6215 | 0.8566 | 0.2865 | 0.1266 |
| 31 | O2O | 0.25 | 624 | 100.00% | 0.9327 | 0.0898 | 0.5754→0.5820 | 0.8087 | 0.2801 | 0.0561 |
| 31 | O2O | 0.50 | 624 | 100.00% | 0.9156 | 0.1143 | 0.5754→0.5716 | 0.7862 | 0.3205 | 0.0753 |
| 31 | O2O | 0.75 | 624 | 100.00% | 0.8950 | 0.1459 | 0.5754→0.5609 | 0.7643 | 0.2947 | 0.1042 |
| 41 | O2M | 0.25 | 624 | 99.84% | 0.9715 | 0.0284 | 0.6628→0.6674 | 0.8927 | 0.3081 | 0.0401 |
| 41 | O2M | 0.50 | 624 | 99.84% | 0.9537 | 0.0491 | 0.6628→0.6583 | 0.8758 | 0.2753 | 0.0754 |
| 41 | O2M | 0.75 | 624 | 99.84% | 0.9281 | 0.0812 | 0.6628→0.6559 | 0.8625 | 0.3309 | 0.1091 |
| 41 | O2O | 0.25 | 624 | 100.00% | 0.9337 | 0.0879 | 0.6212→0.6288 | 0.8192 | 0.2833 | 0.0529 |
| 41 | O2O | 0.50 | 624 | 100.00% | 0.9172 | 0.1115 | 0.6212→0.6216 | 0.8066 | 0.3314 | 0.0721 |
| 41 | O2O | 0.75 | 624 | 100.00% | 0.8963 | 0.1453 | 0.6212→0.6122 | 0.7835 | 0.2964 | 0.1074 |
| 50 | O2M | 0.25 | 624 | 100.00% | 0.9743 | 0.0270 | 0.7017→0.7042 | 0.9027 | 0.3022 | 0.0401 |
| 50 | O2M | 0.50 | 624 | 100.00% | 0.9545 | 0.0496 | 0.7017→0.6976 | 0.8874 | 0.2696 | 0.0721 |
| 50 | O2M | 0.75 | 624 | 100.00% | 0.9340 | 0.0753 | 0.7017→0.6935 | 0.8789 | 0.3114 | 0.0994 |
| 50 | O2O | 0.25 | 624 | 100.00% | 0.9357 | 0.0848 | 0.6592→0.6639 | 0.8352 | 0.3369 | 0.0513 |
| 50 | O2O | 0.50 | 624 | 100.00% | 0.9192 | 0.1096 | 0.6592→0.6579 | 0.8096 | 0.3748 | 0.0721 |
| 50 | O2O | 0.75 | 624 | 100.00% | 0.8945 | 0.1470 | 0.6592→0.6494 | 0.7957 | 0.3948 | 0.1058 |

## Automatic gate

- O2M: monotonic checkpoints `5` [11, 21, 31, 41, 50]; mean weak→strong drop `0.0454`; correlated checkpoints `5` [11, 21, 31, 41, 50].
- O2O: monotonic checkpoints `5` [11, 21, 31, 41, 50]; mean weak→strong drop `0.0366`; correlated checkpoints `5` [11, 21, 31, 41, 50].

GO requires, for both heads: at least `3` checkpoints with non-increasing S (tolerance `0.01`), mean weak→strong drop `≥0.02`, and strongest-view positive Spearman `ρ≥0.2` at at least `2` checkpoints.
Every cell also requires `≥100` objects, the same number assigned in both views, and coverage `≥0.5`; otherwise INCONCLUSIVE.

## Warnings

- GO authorizes only a scale/translate ASAC prototype; Mosaic correspondence was not tested.
