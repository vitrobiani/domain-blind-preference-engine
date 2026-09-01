#!/usr/bin/env python3
"""
Convert a MovieLens u.data-format file into a JSON body for POST /interactions.

Input format (tab-separated, no header):
    user_id \t item_id \t rating \t unix_timestamp

Output shape (matches api.InteractionBatchIn):
    {"interactions": [{"user_id": ..., "item_id": ..., "value": ..., "ts": ...}, ...]}
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> int:
    interactions = []
    with input_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                print(
                    f"warn: {input_path}:{lineno}: expected 4 tab-separated fields, "
                    f"got {len(parts)} - skipping",
                    file=sys.stderr,
                )
                continue
            user_id, item_id, rating, ts = parts
            interactions.append({
                "user_id": user_id,
                "item_id": item_id,
                "value": float(rating),
                "ts": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(),
            })

    output_path.write_text(json.dumps({"interactions": interactions}))
    return len(interactions)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <input u.data> <output json>",
            file=sys.stderr,
        )
        return 1

    n = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Wrote {n} interactions to {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
