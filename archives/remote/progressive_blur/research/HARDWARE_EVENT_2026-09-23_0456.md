# New server memory event during P2 preflight — 2026-09-23

Time: kernel journal **04:56:39 CST**, after the 960 training manager
had already finished as `verified_complete` at 04:51:50 CST. This is
a shared server and we did not alter, terminate, or repair any system
process or hardware configuration.

## Read-only evidence

- Kernel: `mce: Uncorrected hardware memory error in user-access at
  26937ba648`; `Memory failure: 0x26937ba: Sending SIGBUS to
  update-notifier:14497 due to hardware memory corruption`; `recovery
  action for dirty LRU page: Recovered`.
- EDAC on this boot: `mc0/ue_count=0`, `mc1/ue_count=1` after the event;
  both were 0 during the 960 run at the previous check.
- The physical address falls in Linux memory block **77** (block size
  `0x80000000` = 2 GiB). Sysfs links
  `/sys/devices/system/memory/memory77/node1`, confirming this
  particular failing page resides on **NUMA node 1**. We use explicit
  `numactl --cpunodebind=0 --membind=0` for our jobs.
- Sysfs does not expose a simple `mc*/device` or `mc*/node` link here,
  so do **not** infer the general `mc0↔node0` mapping solely from the
  controller index. The exact *new failing page* is independently
  mapped to node1; any future physical address must be checked again.
- No task training/evaluation process was alive at the P2 preflight;
  physical GPUs 4–7 were idle. The SIGBUS victim was not our process.

## Decision and limits

The originally pre-registered P2 hardware gate required no new EDAC UE
counts, and **that gate was tripped**. Do not retrospectively claim it
passed. We defer P2 GPU launch while checking for a further rise.
The user previously obtained the server owner's permission to continue
research on this machine despite known memory faults; this is not
evidence the hardware is healthy. A documented, narrower continuation
gate is possible because the exact new page is on node1, whereas our
jobs bind CPU and memory to node0. Before any new P2 smoke, require:

1. At least 10 minutes after 04:56:39 with no additional new MCE/SIGBUS
   entries and stable EDAC counts (`mc0=0`, `mc1=1`).
2. Physical GPUs 4–7 remain idle, no task training process exists, and
   P2 smoke run directory does not already exist.
3. P2 smoke uses `CUDA_VISIBLE_DEVICES=4,5,6,7` and `numactl`
   CPU/memory binding to node0. If `mc0` UE rises, a new fault maps to
   node0, our process receives SIGBUS, or faults recur, stop the new
   experiment and reassess;
   do not brute-force repeated launches.

Even if these checks pass, residual hardware risk remains. Any
publication-strength result should be rerun on healthy hardware as
the user has already planned. We are not modifying shared daemons,
global environment, or fault counters.

## Follow-up at 05:07–05:09 CST

After >10 minutes, the boot was unchanged, EDAC stayed `mc0=0/mc1=1`,
and the kernel showed no subsequent MCE/SIGBUS after 04:56:39. GPU
4–7 and the task process list were idle. The P2 1-epoch 4-GPU smoke
finished with no new fault, and the same checks remained stable
before formal training. This meets only the **documented narrower
continuation gate**, not the original unmodified no-new-UE gate.

## Second event at 05:23:26 CST during the already-running P2 experiment

- Kernel reported `Uncorrected hardware memory error in user-access at
  26937c3340`, `Memory failure: 0x26937c3: Sending SIGBUS to
  teamviewerd:12439 due to hardware memory corruption`, and a recovered
  dirty-LRU page. The victim is unrelated to this research task. We did
  not stop or modify that process.
- EDAC changed from `mc0=0/mc1=1` at the 05:19 check to `mc0=0/mc1=2`
  at 05:29. The failing physical address divided by the sysfs-advertised
  2-GiB memory-block size maps to block 77; the live sysfs `node1`
  symlink for block 77 verifies this *specific* second fault is on NUMA1.
  This is not evidence that the rest of node0 or the machine is healthy.
- At 05:29 the task P2 manager PID 47597 was still alive, the training CSV
  had reached epoch 88, and no task SIGBUS appeared in the kernel scan.
  No attempt was made to restart or alter the ongoing job. Its completed
  metrics require normal strict audit and an independent val evaluation.
- **Decision:** recurrence trips the earlier "if faults recur, stop new
  experiment and reassess" condition. Do not start the pre-registered
  VisDrone train-only blur smoke/formal run now. Continue low-frequency
  read-only observation of the already-running P2 task. Any future GPU
  launch needs a fresh documented hardware/resource judgment after at
  least 10 quiet minutes from this event; if faults continue, a node0
  page fails, or a task process receives SIGBUS, do not launch/relaunch.
  The server-owner approval to work on this faulty server remains in
  context but does not cancel this safety gate.

## Burst of five deferred uncorrectable errors at 05:43:19–05:43:20

After P2 returned `verified_complete` at 05:37 and after the short
independent P2 val/size diagnostics ran, EDAC `mc1/ue_count` climbed from
2 to **7** (mc0 remained 0). Kernel journal contains five `Deferred
error, no action required` / `UECC|Deferred|Scrub` records on
`mc#1csrow#1channel#0` with EDAC pages `0x2393705`, `0x239373c`,
`0x2393774`, `0x23937ac`, `0x23937e4`. Each page multiplied by the
4-KiB page size falls in memory block **71**, for which the live sysfs
link is `/sys/devices/system/memory/memory71/node1`. This is an exact
page-location check, not a claim that all mc1 faults are necessarily
node1. These journal lines did **not** report a new userspace SIGBUS,
but five more uncorrectable events within seconds are significant.

No new task GPU training is launched. The P2 control and its independent
raw-val metrics were obtained before this observed burst and have
readable artifacts, but all results remain provisional on faulty
hardware. The previously proposed minimum 10-minute quiet gate is not
by itself sufficient to justify immediately launching a long new
experiment after this cluster. Reassess trend and safety conservatively;
do not alter shared daemons, EDAC counters, memory online status or
other processes. If the machine continues to emit UEs, defer new GPU
training and preserve the validated outputs for healthy-hardware rerun.

## Additional burst at 05:58:26 CST; checked at 06:03 CST

The next scheduled low-frequency read-only check found **five further
deferred uncorrectable errors**. EDAC is now `mc0/ue_count=0`,
`mc1/ue_count=12`, up from 7 at 05:53, with the same boot start
`2026-09-23 03:01:16`. The 05:58:26 kernel records are
`UECC|Deferred|Scrub` on `mc#1csrow#1channel#6` at EDAC pages
`0x23c8105`, `0x23c813c`, `0x23c8174`, `0x23c81ac`, and `0x23c81e4`.
The advertised memory-block size is still `0x80000000` bytes (2 GiB);
each 4-KiB page address resolves to block 71, previously verified as
node1. No new userspace SIGBUS appears in this burst. These faults
occurred while no new research GPU job had been launched; do not
attribute them to our model or infer that unaffected memory is healthy.

**Decision:** the fault trend is still active, not a quiet recovery.
Continue holding the pre-registered blur smoke/formal run and all other
new GPU training/evaluation. Do CPU-only literature, audit and code
checks; preserve all task artifacts. Do not alter shared hardware,
memory online state, daemons, fault counters or others' processes.
The next scheduled check remains low-frequency, not an immediate retry.

## 06:33 CST read-only trend and one bounded smoke decision

There were no further journal MCE/UE/SIGBUS matches after 05:58:27,
and EDAC remained `mc0=0/mc1=12` on the same 03:01:16 boot. This is
about 35 quiet minutes, following several earlier bursts; it **does
not** make the machine healthy. Physical GPUs4–7 showed 2 MiB/0% each,
no task manager/trainer process matched, and the task-owned smoke
run/state did not exist. Both task configs matched recorded SHA256.
NUMA0-bound, CPU-only analyses completed without a task SIGBUS.

On the basis of the server owner's earlier permission to continue and
the user-specified GPU/NUMA isolation, the research protocol has been
amended **before** the result to allow one 1%-data, one-epoch 4-GPU
Oracle smoke that exercises the real blur path. The original strict
“no new UE” gate is still recorded as failed. Do not infer permission
for formal training from a smoke start or from the quiet interval alone.
After smoke, audit hardware counters/journal, task-only processes,
checkpoint/CSV and actual blur-path execution. Any recurrence or task
SIGBUS means no further new GPU work pending reassessment. Do not
modify shared daemon, EDAC state, memory online state, or others'
processes.

## 06:35 post-smoke and 06:43 formal preflight

The bounded 1%-data/one-epoch Oracle train-only smoke completed
`verified_complete` at 06:35:39. Afterward, EDAC remained
`mc0=0/mc1=12`, no new matching kernel fault was found, and physical
GPUs4–7 were idle. A separate check at 06:43, about 45 minutes after
the last 05:58 UE, again found the same boot and counts, no matching
kernel MCE/SIGBUS after 06:33, GPUs4–7 at 2 MiB/0%, no task manager or
trainer, formal run/state absent, and the formal config's SHA256
unchanged. This is enough for a **single, task-owned formal launch**
under the already documented narrowed continuation judgment; it is
not evidence of repaired hardware, not permission to retry a failed
checkpoint uncritically, and not publication-grade validation.
