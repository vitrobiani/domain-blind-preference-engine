#!/usr/bin/env python3
"""
Convert the INES 2025 STATA CSV into a JSON body for POST /interactions.

Mirrors adapters/elections/adapter.py: one interaction per respondent (their
v104 answer), sentinel codes filtered out, party codes mapped to slugs.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PARTY_NAMES: dict[int, str] = {
    1: "Likud",
    2: "YeshAtid",
    3: "NationalUnity",
    4: "ReligiousZionism",
    5: "Shas",
    6: "UnitedTorahJudaism",
    7: "YisraelBeiteinu",
    8: "HaDemocratim",
    9: "OtzmaYehudit",
    10: "NationalRight",
    11: "HadashTaal",
    12: "Raam",
    13: "Balad",
}


def convert(input_path: Path, output_path: Path) -> int:
    interactions = []
    with input_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v104_raw = row.get("v104", "").strip()
            if not v104_raw:
                continue
            try:
                code = int(float(v104_raw))
            except ValueError:
                continue
            party = PARTY_NAMES.get(code)
            if party is None:
                continue

            resp_id = row.get("resp_id", "").strip()
            if not resp_id:
                continue

            ts_iso = None
            date_raw = row.get("date", "").strip()
            if date_raw:
                try:
                    ts_iso = datetime.strptime(date_raw, "%d/%m/%Y").replace(
                        tzinfo=timezone.utc
                    ).isoformat()
                except ValueError:
                    ts_iso = None

            interactions.append({
                "user_id": resp_id,
                "item_id": party,
                "value": 1.0,
                "ts": ts_iso,
            })

    output_path.write_text(json.dumps({"interactions": interactions}))
    return len(interactions)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <input 2025_STATA.csv> <output json>",
            file=sys.stderr,
        )
        return 1

    n = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Wrote {n} interactions to {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
