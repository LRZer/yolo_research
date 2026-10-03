# Hardware contamination record

## Verdict

This run must be cited only as **hardware-contaminated / faulty-host exploratory**. It is suitable for paired screening on the authorized faulty host, not as clean paper evidence. Any surviving method must be reproduced from random initialization on healthy hardware.

## Contemporaneously observed events

During the approximately 04:11–04:43 training window on 2026-09-19, the kernel emitted five EDAC MC0 uncorrectable/deferred memory events at 04:36:24–04:36:25. The captured records identified CPU 0, MC21, `UECC|Deferred|Scrub`, FRU `P0MC0`, and the following addresses:

| Physical address | EDAC page | Offset |
|---|---:|---:|
| `0x0000000615bc1900` | `0x18c6f07` | `0x300` |
| `0x0000000615bcf900` | `0x18c6f3e` | `0x200` |
| `0x0000000615bdd900` | `0x18c6f76` | `0x200` |
| `0x0000000615beb900` | `0x18c6fae` | `0x300` |
| `0x0000000615bf9900` | `0x18c6fe6` | `0x300` |

The first captured machine-check status was `0xd42031000400011b`; the event family was reported as deferred and did not terminate the trainer.

## What can and cannot be concluded

- No `Sending SIGBUS` line was present for this window.
- The training log contained no `SIGBUS`, bus error, traceback, runtime error, OOM, NCCL failure, or NaN marker, and all 50 result rows and checkpoints were written.
- `P0MC0` and the address region suggest the known NUMA-0-side fault pattern, while the job was launched with NUMA-1 binding. This is only a topology-based suspicion; it does **not** prove that the training process or its data were unaffected.
- Successful completion and matching file hashes prove archival integrity, not numerical correctness in the presence of faulty memory.
- The host rebooted at 04:47:05. Because the journal was volatile, the previous boot could not be re-read after reconnection; the event details above come from the contemporaneous kernel capture made before the reboot.

Accordingly, the run is retained for transparent exploratory comparison but must not be described as a clean window.
