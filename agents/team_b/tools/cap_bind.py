"""expB-304b-caps / R-47 — where do the tile-cap constants actually bind?

`identity_check.py` answers the yes/no question (delete the constant, is the
agent the same agent?).  When the answer is DIVERGENT it stops being useful,
because two trajectories that differ at step 40 differ at every later step for
reasons that have nothing to do with the cap.  This tool answers the follow-up
the ruling asks for — *which* turns bind — with a one-step-deviation measurement
along the CHAMPION's own trajectory:

    at every turn the champion plans and acts; the same turn is then re-planned
    from a DEEP COPY of the champion's own pre-turn state with the cap deleted
    (`melon_rule`/`strawberry_rule` = "uncapped": the ask becomes the structural
    site budget).  A turn is BINDING for a cap when that re-plan emits a
    different action.

Two properties make this the right reading and neither is free:

  - the deviation shares the champion's entire state — the same ledgers, router
    and market model — so a difference is attributable to the constant and not
    to drift.  A persistent shadow agent was tried first and is uninterpretable:
    once it plans one different action its lambda ledger separates and it
    reports ~95% "binding" for the rest of the season.
  - nothing the deviation produces is ever played, so the trajectory stays
    exactly the champion's and every turn is measured against one reference.

`_plan` ignores its `deadline` argument (verified), so the re-plan is not a
timing-dependent reading.

Usage (repo root):
  uv run python agents/team_b/tools/cap_bind.py --seeds 0-7 --both-seats \\
      --workers 8 --out bench/runs/expB-304b-caps/bind/report.json

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))

# The gated Champion[B] entry configuration, s32m6 — the literal from
# agents/team_b/millrace/main.py, asserted against it by `--assert-entry`.
CHAMPION = {"melon_replant": True, "melon_rule": "fixed",
            "melon_tiles": 6,
            "strawberry_tiles": 32,
            "dispatch": "priced", "dig": True,
            "dedupe_ontile": True,
            "ongoing_water_span": True}
# The gate's incumbent opponent, TeamB-Millrace-pricer@hdd12354335.
PRICER = {"melon_replant": True, "melon_rule": "mr",
          "strawberry_tiles": 16,
          "dispatch": "priced", "dig": True,
          "dedupe_ontile": True,
          "ongoing_water_span": True}

ARMS = {
    "melon": {"melon_rule": "uncapped"},
    "straw": {"strawberry_rule": "uncapped"},
}
TURNS_PER_DAY = 24


def _canon(a) -> str:
    return json.dumps(a, sort_keys=True, separators=(",", ":"), default=str)


def _market(a):
    return _canon(a.get("market", []) if isinstance(a, dict) else [])


def _units(a):
    if not isinstance(a, dict):
        return _canon(a)
    return _canon([a.get("farmer")] + list(a.get("hands") or []))


def _counts(farm):
    counts, empties = {}, 0
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                empties += 1
            elif isinstance(t, dict) and "crop" in t:
                counts[t["crop"]] = counts.get(t["crop"], 0) + 1
    return counts, empties


def run_episode(seed: int, seat: int, steps: int) -> dict:
    """One seat-season of the champion, with a one-step deviation per cap per turn.

    The deviation shares the champion's ENTIRE pre-turn state (a deep copy: the
    same ledgers, the same router, the same market model) and differs only in
    `P`.  That is what makes a difference in the emitted action attributable to
    the cap rather than to accumulated drift.
    """
    import copy
    import time

    from _engine import import_engine, engine_version
    from millrace import chassis as ch
    from millrace.feedchain import _plan, make_feedchain_agent

    ke = import_engine()
    champ = make_feedchain_agent(CHAMPION)
    opp = make_feedchain_agent(PRICER)

    env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed},
                  debug=False)
    env.reset()
    cfg = env.configuration
    rows = []
    while not env.done:
        obs_us = env.state[seat].observation
        obs_them = env.state[1 - seat].observation
        pre = copy.deepcopy(champ.state)          # BEFORE the champion mutates it
        act_us = champ(obs_us, cfg)
        act_them = opp(obs_them, cfg)
        farm = obs_us.farms[seat]
        counts, empties = _counts(farm)
        seeds_held = (obs_us.private.get("seeds", {}) or {})
        # `step` is a SHARED observation field and is present on seat 0 only
        # (measured); day/hour are per-seat, so derive it.
        step = int(obs_us["day"]) * TURNS_PER_DAY + int(obs_us["hour"])
        row = {"step": step, "day": step // TURNS_PER_DAY, "hour": step % TURNS_PER_DAY,
               "empties": empties, "money": float(farm.get("money", 0)),
               "standing": {c: int(counts.get(c, 0)) for c in ("MELON", "STRAWBERRY")},
               "held": {c: int(seeds_held.get(c, 0)) for c in ("MELON", "STRAWBERRY")},
               "want": {"MELON": int(champ.state["st"].get("want_melon", -1)),
                        "STRAWBERRY": int(champ.state["st"].get("want_straw", -1))},
               "arms": {}}
        for name, override in ARMS.items():
            sstate = copy.deepcopy(pre)
            sstate["P"] = {**sstate["P"], **override}
            act_sh = ch.sanitise(obs_us, _plan(obs_us, sstate, time.perf_counter() + 60.0))
            m_diff = _market(act_sh) != _market(act_us)
            u_diff = _units(act_sh) != _units(act_us)
            key = "want_melon" if name == "melon" else "want_straw"
            row["arms"][name] = {
                "market_diff": m_diff, "units_diff": u_diff,
                "bind": bool(m_diff or u_diff),
                "want_shadow": int(sstate["st"].get(key, -1)),
                "market_champ": act_us.get("market", []) if m_diff else None,
                "market_shadow": act_sh.get("market", []) if m_diff else None,
            }
        rows.append(row)
        env.step([act_us, act_them] if seat == 0 else [act_them, act_us])

    return {"seed": seed, "seat": seat, "engine": engine_version(),
            "money": [float(env.state[0].observation.farms[i]["money"]) for i in (0, 1)],
            "plan_exceptions": int(getattr(champ, "plan_exceptions", 0) or 0),
            "rows": rows}


def summarise(eps: list[dict]) -> dict:
    out = {}
    for name in ARMS:
        n_turns = 0
        binds = 0
        market_only = units_only = both = 0
        by_day = Counter()
        turns_by_day = Counter()
        first_days = []
        peak_hits = 0          # turns where standing == want (the cap is AT its limit)
        want_gap = []
        for ep in eps:
            first = None
            for r in ep["rows"]:
                a = r["arms"][name]
                n_turns += 1
                turns_by_day[r["day"]] += 1
                crop = "MELON" if name == "melon" else "STRAWBERRY"
                if r["standing"][crop] >= r["want"][crop] > 0:
                    peak_hits += 1
                want_gap.append(a["want_shadow"] - r["want"][crop])
                if a["bind"]:
                    binds += 1
                    by_day[r["day"]] += 1
                    if a["market_diff"] and a["units_diff"]:
                        both += 1
                    elif a["market_diff"]:
                        market_only += 1
                    else:
                        units_only += 1
                    if first is None:
                        first = r["step"]
            first_days.append(None if first is None else first // TURNS_PER_DAY)
        prof = {str(d): round(by_day[d] / max(1, turns_by_day[d]), 4)
                for d in sorted(turns_by_day)}
        firsts = [d for d in first_days if d is not None]
        out[name] = {
            "turns": n_turns,
            "bind_turns": binds,
            "bind_share": round(binds / max(1, n_turns), 4),
            "channel": {"market_only": market_only, "units_only": units_only, "both": both},
            "seat_seasons_with_a_bind": len(firsts),
            "seat_seasons": len(eps),
            "first_bind_day_min": min(firsts) if firsts else None,
            "first_bind_day_median": sorted(firsts)[len(firsts) // 2] if firsts else None,
            "first_bind_day_max": max(firsts) if firsts else None,
            "at_cap_share": round(peak_hits / max(1, n_turns), 4),
            "mean_want_gap": round(sum(want_gap) / max(1, len(want_gap)), 2),
            "bind_share_by_day": prof,
        }
    return out


def _job(args):
    seed, seat, steps = args
    return run_episode(seed, seat, steps)


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--both-seats", action="store_true")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--assert-entry", action="store_true",
                    help="check CHAMPION against the shipped main.py literal")
    a = ap.parse_args(argv)

    if a.assert_entry:
        txt = (REPO_ROOT / "agents/team_b/millrace/main.py").read_text(encoding="utf-8")
        for k, v in CHAMPION.items():
            lit = repr(v) if isinstance(v, bool) else json.dumps(v)
            assert f'"{k}": {lit}' in txt, f"{k} not pinned in main.py as {lit}"
        print("entry config matches main.py")

    seats = (0, 1) if a.both_seats else (0,)
    jobs = [(s, seat, a.steps) for s in parse_seeds(a.seeds) for seat in seats]
    eps = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for ep in ex.map(_job, jobs):
            eps.append(ep)
            print(f"  seed {ep['seed']} seat {ep['seat']}: money {ep['money']} "
                  f"plan_exceptions {ep['plan_exceptions']}", flush=True)

    rep = {"experiment": "expB-304b-caps", "ruling": "R-47",
           "engine": eps[0]["engine"], "seat_seasons": len(eps),
           "champion": CHAMPION, "opponent": PRICER, "arms": ARMS,
           "summary": summarise(eps)}
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        # the per-turn rows are large; keep them beside the report
        (p.parent / "rows.json").write_text(
            json.dumps([{k: v for k, v in ep.items() if k != "rows"} |
                        {"rows": ep["rows"]} for ep in eps]), encoding="utf-8")
    print(json.dumps(rep["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
