"""Remove one crop line from a fixed-game schedule without changing anything else."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    crop = args.crop.upper()
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    for day in schedule.get("days", []):
        day["plantings"] = [row for row in day.get("plantings", []) if row.get("crop") != crop]
        day["purchases"] = [row for row in day.get("purchases", [])
                            if row.get("item") != f"{crop}_SEED"]
        day["sales"] = [row for row in day.get("sales", []) if row.get("product") != crop]
    line = schedule.get("crop_lines", {}) or {}
    line["priority_products"] = [item for item in line.get("priority_products", []) if item != crop]
    for key in ("season_drain", "target_tiles", "last_plant_day"):
        (line.get(key, {}) or {}).pop(crop, None)
    schedule["family"] = f"{schedule.get('family', 'schedule')}_without_{crop.lower()}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
