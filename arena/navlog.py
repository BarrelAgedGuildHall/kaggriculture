"""S-3: per-episode NAV trajectories, recorded by the Pit.

What this is for
----------------
`nav` (S-1) marks a farm's horizon-NPV balance sheet. Until now the only NAV numbers we had
came from offline studies on hand-picked episodes. S-3 makes the arena itself produce them, so
"does NAV predict the final bank, on the population of games we actually play?" is a query
rather than an experiment -- and so a gate run can carry a win-probability calibration read
beside its win rate.

Two constraints shaped every decision here.

**1. Zero timing impact on measured arms.** The per-turn `duration` the store keeps is measured
by the framework *around the agent call*, so anything computed after `env.run()` returns cannot
move it -- and nothing here runs during the episode. The cost is wall-clock only (~35 ms per
seat on a 720-step game, against ~1-10 s for the game itself), it lands on the ladder, and it is
off by default everywhere else. Gates therefore measure exactly what they measured before.

**2. The Pit does not retain replays** (crash captures only, deliberately -- 145k games would be
terabytes). So the trajectory is computed at game end from the episode still in memory, before
the env is discarded. No retention flag, no replay parsing, nothing to clean up later.

The marks are replayed through `nav.Nav`, i.e. the same code an agent runs, and the curves only
ever see points up to the step being marked -- so a step's mark is what the agent could have
known at that step, not hindsight.

Volume
------
The ladder plays tens of thousands of games a day, so recording every one is not free: a
31-point trajectory is ~7 kB per seat. `rate` (percent) selects a deterministic, unbiased
sample by `game_id` -- the same pairing+seed is always in or always out, so a re-run of a game
re-records it and the sample does not drift with traffic.

Both seats are marked from their OWN observation, which includes their private shed and seeds.
That is *more* than `Nav.opponent_sheet()` can see in-game, and it is the right thing offline:
`phi0 - phi1` from two stored rows is the exact relative potential, against which an agent's
in-game (public-only) estimate can be scored.
"""
from __future__ import annotations

import json
import statistics
import traceback

DEFAULT_SAMPLE_EVERY = 24      # one mark per game day
DEFAULT_RATE = 5               # percent of ladder games that record a trajectory
VARIANT = "drain"              # the most conservative mark (docs/nav.md); stamped on every row
LAM_NAME = "LAMBDA_CRITIC"     # exp-N1's Test-2 arm -- the one calibrated against the final bank

# Dropped from every stored point: `detail` is a per-line breakdown that triples the row size,
# and `lam`/`variant`/`schema_version`/`seat` are constant for the whole trajectory and live in
# the row's own columns instead. What remains is `Sheet.as_dict()`'s scalar half.
POINT_DROP = ("detail", "lam", "variant", "schema_version", "seat")


def should_record(game_id: str, rate: int) -> bool:
    """Deterministic sampling by game id: stable across re-runs, independent of traffic."""
    rate = int(rate)
    if rate <= 0:
        return False
    if rate >= 100:
        return True
    try:
        return int(str(game_id)[:8], 16) % 100 < rate
    except ValueError:
        return False


def sample_steps(n_steps: int, sample_every: int) -> list[int]:
    """Every `sample_every`-th snapshot, and always the terminal one.

    The terminal snapshot is the point of the exercise: `H(719) = 0`, so `phi` there must be 0
    by construction (exp-N1 Test 1b). A trajectory that stops at 696 cannot show that it does.
    """
    step = max(1, int(sample_every))
    return sorted(set(range(0, n_steps, step)) | {n_steps - 1}) if n_steps > 0 else []


def _point(sheet, detail: bool) -> dict:
    d = sheet.as_dict()
    drop = tuple(k for k in POINT_DROP if not (detail and k == "detail"))
    return {k: v for k, v in d.items() if k not in drop}


def seat_row(observations, *, seat: int, meta: dict, sample_every: int = DEFAULT_SAMPLE_EVERY,
             variant: str = VARIANT, lam=None, detail: bool = False) -> dict:
    """Mark one seat's whole episode and return the row that goes into the store."""
    import nav

    lam = lam or nav.LAMBDA_CRITIC
    row = {
        **meta, "seat": int(seat),
        "nav_schema": nav.SCHEMA_VERSION, "nav_source": nav.source_hash(),
        "variant": variant, "lam_opp": float(lam.opp), "lam_wage": float(lam.wage),
        "lam_name": LAM_NAME, "sample_every": int(sample_every),
        "n_points": 0, "final_step": -1, "final_cash": 0.0, "final_nav": 0.0, "final_phi": 0.0,
        "provisional": 0, "tripwire": "", "diagnostics": "{}", "points": "[]", "error": "",
    }
    try:
        at = sample_steps(len(observations), sample_every)
        points = [_point(s, detail) for _, s in
                  nav.Nav.replay_sheets(observations, at=at, variant=variant, lam=lam)]
        if points:
            last = points[-1]
            row.update(n_points=len(points), final_step=last["step"], final_cash=last["cash"],
                       final_nav=last["nav"], final_phi=last["phi"],
                       provisional=int(any(p["provisional"] for p in points)),
                       diagnostics=json.dumps(last.get("diagnostics") or {}),
                       tripwire=json.dumps((last.get("diagnostics") or {}).get("tripwire", "")),
                       points=json.dumps(points, separators=(",", ":")))
    except Exception:
        # Telemetry never costs a game. A seat that could not be marked says so in its own row,
        # which is also how a `nav` bump that breaks the replay path becomes visible.
        row["error"] = traceback.format_exc()[-400:]
    return row


def game_rows(env, game: dict, *, sample_every: int = DEFAULT_SAMPLE_EVERY, variant: str = VARIANT,
              detail: bool = False) -> list[dict]:
    """Both seats of a finished episode, from the in-memory env. Never raises."""
    rows = []
    for seat in (0, 1):
        meta = {
            "game_id": game["game_id"], "run_id": game.get("run_id", ""), "mode": game.get("mode", ""),
            "agent": game.get(f"p{seat}", ""), "seed": int(game.get("seed", -1)),
            "engine": game.get("engine", ""), "arena_sha": game.get("arena_sha", ""),
            "host": game.get("host", ""), "money": float(game.get(f"money{seat}", 0.0)),
            "result": float(game.get("result0", 0.5)) if seat == 0 else 1.0 - float(game.get("result0", 0.5)),
            "created_at": game.get("created_at", ""),
        }
        try:
            observations = [s[seat].observation for s in env.steps]
        except Exception:
            rows.append({**meta, "seat": seat, "error": "no observations for this seat",
                         "nav_schema": 0, "nav_source": "", "variant": variant, "lam_opp": 0.0,
                         "lam_wage": 0.0, "lam_name": LAM_NAME, "sample_every": int(sample_every),
                         "n_points": 0, "final_step": -1, "final_cash": 0.0, "final_nav": 0.0,
                         "final_phi": 0.0, "provisional": 0, "tripwire": "", "diagnostics": "{}",
                         "points": "[]"})
            continue
        rows.append(seat_row(observations, seat=seat, meta=meta, sample_every=sample_every,
                             variant=variant, detail=detail))
    return rows


def trajectory(row: dict) -> list[dict]:
    """The stored points of one row, decoded."""
    try:
        return json.loads(row.get("points") or "[]")
    except (TypeError, ValueError):
        return []


# ------------------------------------------------------------------------------- calibration
MIN_CALIBRATION_N = 8      # below this a correlation is noise wearing a number's clothes


def _pearson(xs, ys) -> float | None:
    try:
        return round(statistics.correlation(xs, ys), 4)
    except (statistics.StatisticsError, ValueError):
        return None          # a constant column (every PASS agent banks 3000) has no correlation


def calibration(rows) -> list[dict]:
    """Per sampled step: does the mark predict the outcome?

    Three numbers, and the third is why the first two are worth reading:

    * `r_nav` -- correlation between NAV at this step and the seat's realised final bank.
    * `r_cash` -- **the mandatory cash-alone control** (exp-N1's spec): the same correlation
      using cash only. If NAV does not beat it, the marks are adding nothing and the potential
      is decoration. Reporting one without the other is how a NAV study talks itself into a
      result.
    * `sign_acc` -- over games where both seats were recorded, how often the sign of the NAV gap
      (ours minus theirs) already agrees with who won. This is the win-probability read, and it
      is computed from two seats marked with FULL information, so it is the ceiling an in-game
      public-only estimate is scored against, not the estimate itself.

    Ties are excluded from `sign_acc` (there is no sign to get right) and counted separately.
    """
    per_seat: dict[tuple[str, int], dict] = {}
    for r in rows:
        pts = trajectory(r)
        if not pts:
            continue
        # `or` defaults are wrong here and quietly so: a LOSS is result 0.0, which `or 0.5`
        # turns into a tie -- every loss would drop out of the sign accuracy.
        money = r.get("money")
        result = r.get("result")
        per_seat[(r["game_id"], int(r["seat"]))] = {
            "money": float(0.0 if money is None else money),
            "result": float(0.5 if result is None else result),
            "by_step": {int(p["step"]): p for p in pts},
        }
    steps = sorted({s for v in per_seat.values() for s in v["by_step"]})
    out = []
    for step in steps:
        navs, cashes, moneys = [], [], []
        for v in per_seat.values():
            p = v["by_step"].get(step)
            if p is None:
                continue
            navs.append(p["nav"]); cashes.append(p["cash"]); moneys.append(v["money"])
        pairs = 0
        right = 0
        ties = 0
        for (gid, seat), v in per_seat.items():
            if seat != 0:
                continue
            them = per_seat.get((gid, 1))
            if them is None or step not in v["by_step"] or step not in them["by_step"]:
                continue
            gap = v["by_step"][step]["nav"] - them["by_step"][step]["nav"]
            if v["result"] == 0.5 or gap == 0:
                ties += 1
                continue
            pairs += 1
            right += int((gap > 0) == (v["result"] > 0.5))
        if len(navs) < MIN_CALIBRATION_N:
            continue
        out.append({
            "step": step, "day": step // 24, "n": len(navs),
            "r_nav": _pearson(navs, moneys), "r_cash": _pearson(cashes, moneys),
            "n_pairs": pairs, "n_ties": ties,
            "sign_acc": round(right / pairs, 4) if pairs else None,
        })
    return out
