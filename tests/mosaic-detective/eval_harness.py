# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Blind evaluation harness for the Mosaic Detective skill.

Runs trials: restore the cluster, inject a fault, ask Claude to diagnose it
blind, and record the answer against ground truth.

Injection is manual because the fault commands need sudo on the cluster.
The harness pauses and prompts; everything else is automated.
"""

import os
import subprocess
import time
from datetime import datetime

FAULT_DIR = "~/mosaic/tests/fault-injection"
SKILL_DIR = "~/.claude/skills/mosaic-detective"

FAULTS = {
    "clock_clamp": {
        "inject": "make inject-slow-gpu RANK=3 DEADMAN=900",
        "settle": 15,
        "restarts_workload": False,
        "truth": "GPU clock clamp on one GPU (rank 3, golf)",
    },
    "netem_delay": {
        "inject": "make inject-netem-delay MS=20 DEADMAN=900",
        "settle": 15,
        "restarts_workload": False,
        "truth": "network degradation (added latency on the interconnect)",
    },
    "kill_collector": {
        "inject": "make inject-kill-collector DEADMAN=600",
        "settle": 15,
        "restarts_workload": False,
        "truth": "the OTel collector was killed (telemetry gap, cluster healthy)",
    },
    "kill_rank": {
        "inject": "make inject-kill-rank RANK=2",
        "settle": 15,
        "restarts_workload": True,
        "truth": "a rank process was killed (rank 2 on golf)",
    },
}


def bravo(command, timeout=300):
    """Run a make command in the fault-injection dir on bravo."""
    full = f"cd {FAULT_DIR} && {command}"
    result = subprocess.run(
        ["ssh", "bravo", full],
        capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr


def ask_claude(prompt="Kowalski, analysis", timeout=900):
    """Run one headless Claude diagnosis, return its answer."""
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=timeout,
        cwd=os.path.expanduser(SKILL_DIR))
    return result.stdout


def run_trial(fault_name):
    print("[0/6] syncing skill files...")
    subprocess.run(["./sync-skill.sh"], check=True,cwd=os.path.dirname(os.path.abspath(__file__)))
    """Run one blind trial: restore, inject (manual), diagnose, restore."""
    fault = FAULTS[fault_name]
    print(f"\n{'='*60}\nTRIAL: {fault_name}\n{'='*60}")

    print("[1/6] restoring cluster...")
    bravo("make restore")

    print("[2/6] ensuring workload is running...")
    status = bravo("make status")
    if "bravo      0" in status or "golf       0" in status:
        print("      workload down, restarting...")
        bravo("make workload")
        print("      waiting 120s for it to stabilise...")
        time.sleep(120)

    print("\n>>> INJECT NOW on bravo:")
    print(f">>>   cd ~/mosaic/tests/fault-injection && {fault['inject']}")
    input(">>> press ENTER when the injection has completed... ")

    print(f"[4/6] waiting {fault['settle']}s for the fault to register...")
    time.sleep(fault["settle"])

    print("[5/6] asking Claude...")
    started = time.time()
    answer = ask_claude()
    elapsed = time.time() - started
    print(f"      got {len(answer)} chars in {elapsed:.0f}s")

    print("[6/6] restoring...")
    bravo("make restore")
    if fault["restarts_workload"]:
        bravo("make workload")

    return {
        "timestamp": datetime.now().isoformat(),
        "fault": fault_name,
        "truth": fault["truth"],
        "answer": answer,
        "elapsed_s": round(elapsed),
    }


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "clock_clamp"
    result = run_trial(name)
    print("\n" + "="*60)
    print(f"TRUTH:  {result['truth']}")
    print("="*60)
    print(result["answer"])
