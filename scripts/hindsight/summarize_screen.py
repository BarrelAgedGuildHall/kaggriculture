"""Summarize the fixed hs3/control screen with whole-seed bootstrap intervals."""
from __future__ import annotations

import json
import random
import sqlite3
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "data" / "results" / "hindsight-v3" / "fresh_screen.json"


def _interval(values: list[float]) -> list[float]:
    rng = random.Random(42)
    draws = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(10_000))
    return [draws[250], draws[9750]]


def _one(tag: str, arena: Path) -> dict:
    report = json.loads((REPO / "data" / "results" / tag / "results.json").read_text(encoding="utf-8"))
    opponent_names = ("V46", "tapeselect")
    summaries = []
    with sqlite3.connect((arena / "arena.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = list(db.execute("select * from games where run_id=?", (report["run_id"],)))
    for opponent_name, opponent_row in zip(opponent_names, report["per_opponent"]):
        opponent = opponent_row["opponent"]
        by_seed: dict[int, list[tuple[float, float]]] = {}
        for row in rows:
            if opponent not in (row["p0"], row["p1"]):
                continue
            seat = 0 if row["p0"] == report["agent"] else 1
            bank = float(row[f"money{seat}"])
            margin = bank - float(row[f"money{1-seat}"])
            by_seed.setdefault(int(row["seed"]), []).append((bank, margin))
        assert len(by_seed) == 8 and all(len(v) == 2 for v in by_seed.values())
        seed_banks = [statistics.fmean(x[0] for x in by_seed[s]) for s in sorted(by_seed)]
        seed_margins = [statistics.fmean(x[1] for x in by_seed[s]) for s in sorted(by_seed)]
        summaries.append({"opponent": opponent_name, "opponent_id": opponent,
                          "games": 16, "seeds": 8,
                          "mean_bank": statistics.fmean(seed_banks), "bank_ci95": _interval(seed_banks),
                          "mean_margin": statistics.fmean(seed_margins), "margin_ci95": _interval(seed_margins),
                          "faults": opponent_row["crashes"] + opponent_row["timeouts"],
                          "crashes": opponent_row["crashes"], "timeouts": opponent_row["timeouts"],
                          "p99_ms": opponent_row["turn_ms_p99_max"], "max_ms": opponent_row["turn_ms_max"]})
    return {"tag": tag, "agent_id": report["agent"], "run_id": report["run_id"], "opponents": summaries}


def main() -> None:
    payload = {"method": "whole-seed paired bootstrap, 10000 resamples, both seats averaged per seed",
               "seed_base": 4100, "seed_count": 8, "games_per_opponent": 16,
               "hs3": _one("hs3-screen", REPO / "data" / "arena-hs3-screen"),
               "control": _one("hs3-control", REPO / "data" / "arena-hs3-control")}
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
