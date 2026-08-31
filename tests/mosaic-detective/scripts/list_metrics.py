# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Print every metric name this Prometheus knows about."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaic_queries import MosaicClient, MosaicQueryError


def main():
    parser = argparse.ArgumentParser(description="List metric names from Prometheus.")
    parser.add_argument("--filter", dest="contains",
                        help="only show names containing this substring")
    args = parser.parse_args()

    client = MosaicClient()
    try:
        names = client.list_metric_names()
    except MosaicQueryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.contains:
        names = [n for n in names if args.contains.lower() in n.lower()]
    for name in sorted(names):
        print(name)
    print(f"\n{len(names)} metric(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
