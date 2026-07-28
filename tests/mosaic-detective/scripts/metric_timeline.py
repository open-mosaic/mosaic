# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Show how a metric moved over a recent time window."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaic_queries import MosaicClient, MosaicQueryError, parse_range


def find_transitions(values, threshold):
    """Return [(timestamp, before, after)] where the value changed by more than
    `threshold` (a fraction of the previous value)."""
    out = []
    prev_ts, prev_val = values[0]
    for ts, val in values[1:]:
        if prev_val != 0 and abs(val - prev_val) / abs(prev_val) > threshold:
            out.append((ts, prev_val, val))
        prev_ts, prev_val = ts, val
    return out

def main():
    parser = argparse.ArgumentParser(description="Show a metric's recent timeline.")
    parser.add_argument("metric", help="metric name, e.g. nccl_profiler_collective_bytes_total")
    parser.add_argument("--minutes", type=int, default=30, help="window size in minutes")
    parser.add_argument("--step", default="15s", help="sample interval")
    parser.add_argument("--transitions", action="store_true",help="report timestamps where the value changed sharply")
    parser.add_argument("--find-anomaly", action="store_true",help="scan for the most recent change in rate of change")
    parser.add_argument("--threshold", type=float, default=0.2,help="fractional change counted as a transition, for --transitions only (default 0.2 = 20%%)")
    args = parser.parse_args()

    end = time.time()
    start = end - args.minutes * 60

    client = MosaicClient()
    try:
        data = client.query_range(args.metric, start=start, end=end, step=args.step)
    except MosaicQueryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    series_list = parse_range(data)

    if not series_list:
        print(f"no data for {args.metric} in the last {args.minutes}m")
        return 0

    for series in series_list:
        vals = [v for _, v in series.values]
        if not vals:
            continue
        labels = {k: v for k, v in series.labels.items() if k != "__name__"}
        print(f"{labels}")
        print(f"    first={vals[0]:.2f}  last={vals[-1]:.2f}  "
        f"min={min(vals):.2f}  max={max(vals):.2f}  points={len(vals)}")
        if args.transitions:
            for ts, before, after in find_transitions(series.values, args.threshold):
                when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                print(f"    transition at {when}: {before:.1f} -> {after:.1f}")
        if args.find_anomaly:
            anomalies = find_anomalies(series.values)
            if anomalies:
                for ts, before, after in anomalies:
                    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                    print(f"    ANOMALY at {when}: rate {before:.2f}/s -> {after:.2f}/s")
            else:
                print("    no rate change detected")
        

    print(f"\n{len(series_list)} series over {args.minutes}m")
    return 0

def find_anomalies(values, threshold=0.5):
    """Find where a series' rate of change shifts sharply.

    Works for both gauges (value jumps) and counters (growth stops or slows).
    Returns [(timestamp, rate_before, rate_after)], most recent last.
    """
    if len(values) < 4:
        return []

    # Per-step rate of change.
    rates = []
    for i in range(1, len(values)):
        dt = values[i][0] - values[i - 1][0]
        dv = values[i][1] - values[i - 1][1]
        rates.append((values[i][0], dv / dt if dt else 0.0))

    out = []
    for i in range(1, len(rates)):
        prev_rate = rates[i - 1][1]
        curr_rate = rates[i][1]
        # A shift is significant if it moves by more than `threshold` of the
        # larger of the two rates. Using the larger side means a drop to zero
        # registers as a full change rather than dividing by zero.
        scale = max(abs(prev_rate), abs(curr_rate))
        if scale == 0:
            continue
        if abs(curr_rate - prev_rate) / scale > threshold:
            out.append((rates[i][0], prev_rate, curr_rate))
    return out


if __name__ == "__main__":
    sys.exit(main())
