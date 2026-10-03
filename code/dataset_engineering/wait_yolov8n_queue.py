#!/usr/bin/env python3
"""Low-noise waiter for a durable YOLO queue state transition."""

import json
import sys
import time
from pathlib import Path


STATE = Path("/home/ubuntu/rz/object_detection_dataset_test/_management/yolov8n_scratch_queue/queue_state.json")
target = int(sys.argv[1])
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
deadline = time.monotonic() + timeout

while time.monotonic() < deadline:
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        completed = int(state.get("completed_jobs", 0))
        status = state.get("status")
        if completed >= target or status in {"blocked_repeated_failure", "all_22_jobs_completed"}:
            print(json.dumps({
                "status": status,
                "completed_jobs": completed,
                "total_jobs": state.get("total_jobs"),
                "active_job": state.get("active_job"),
                "active_child_pid": state.get("active_child_pid"),
            }, ensure_ascii=False), flush=True)
            raise SystemExit(0)
    except FileNotFoundError:
        pass
    time.sleep(30)

print(json.dumps({"status": "wait_timeout", "target_completed_jobs": target}), flush=True)
raise SystemExit(124)
