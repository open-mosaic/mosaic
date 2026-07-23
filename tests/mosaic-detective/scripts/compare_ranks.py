# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Compare a metric across NCCL ranks to localise a fault."""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaic_queries import MosaicClient, MosaicQueryError, parse_range


def main():
    parser = argparse.ArgumentParser(description="Compare a metric across ranks.")
    parser.add_argument("metric", help="metric name to compare across ranks")
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--step", default="15s")
    parser.add_argument("--stat", choices=["delta", "mean"], default="delta",help="delta for counters (default), mean for gauges like clocks")
    args = parser.parse_args()

    end = time.time()
    start = end - args.minutes * 60

    client = MosaicClient()
    try:
        topology = client.rank_topology(minutes=args.minutes)
    except MosaicQueryError:
        topology = {}
    uuid_to_rank = {info["gpu_uuid"]: rank for rank, info in topology.items()}

    try:
        data = client.query_range(args.metric, start=start, end=end, step=args.step)
    except MosaicQueryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ranks = {}
    for series in parse_range(data):
        if not series.values:
            continue
        rank = series.labels.get("rank")
        if rank is None:
            uuid = series.labels.get("UUID", "").replace("GPU-", "")
            rank = uuid_to_rank.get(uuid)
        if rank is None:
            continue
        vals = [v for _, v in series.values]
        ranks[rank] = {
            
            "mean": sum(vals) / len(vals),
            "delta": vals[-1] - vals[0],
            "last": vals[-1],
            "hostname": (series.labels.get("hostname")
            or series.labels.get("host")
            or topology.get(rank, {}).get("hostname", "?")),
        }


    if not ranks:
        print(f"no per-rank data for {args.metric} in the last {args.minutes}m")
        return 0

 

    med = statistics.median([r[args.stat] for r in ranks.values()])

    if med == 0:
        print(f"median is zero for {args.metric}; cannot compute ratios")
        return 0

    for rank, info in sorted(ranks.items(), key=lambda kv: int(kv[0])):
        value = info[args.stat]
        ratio = value / med
        print(f"rank {rank:>2}  {info['hostname']:8}  {args.stat}={value:>14.1f}  ratio={ratio:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
