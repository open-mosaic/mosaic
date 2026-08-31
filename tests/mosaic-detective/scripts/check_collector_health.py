# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Report which Prometheus scrape targets are up or down."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaic_queries import MosaicClient, MosaicQueryError, parse_instant


def main():
    client = MosaicClient()
    try:
        samples = parse_instant(client.query_instant("up"))
    except MosaicQueryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    down = []
    for s in samples:
        job = s.labels.get("job", "?")
        instance = s.labels.get("instance", "?")
        state = "UP" if s.value[1] == 1.0 else "DOWN"
        print(f"{state:4}  {job:30}  {instance}")
        if s.value[1] != 1.0:
            down.append(f"{job}/{instance}")

    if down:
        print(f"\n{len(down)} target(s) DOWN: {', '.join(down)}")
    else:
        print(f"\nall {len(samples)} targets up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
