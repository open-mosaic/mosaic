# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Show how a metric moved over a recent time window."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaic_queries import MosaicClient, MosaicQueryError, parse_range


def main():
    parser = argparse.ArgumentParser(description="Show a metric's recent timeline.")
    parser.add_argument("metric", help="metric name, e.g. nccl_profiler_collective_bytes_total")
    parser.add_argument("--minutes", type=int, default=30, help="window size in minutes")
    parser.add_argument("--step", default="15s", help="sample interval")
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

    print(f"\n{len(series_list)} series over {args.minutes}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
