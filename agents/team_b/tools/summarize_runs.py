"""Summarize instrumented-run JSONs (sim-checklist expB-102/107/108/111)."""
import json
import sys
from pathlib import Path

for f in sorted(Path(sys.argv[1]).glob("**/*.json")):
    s = json.loads(f.read_text())
    acct = s.get("final_accounting") or [{}, {}]
    leak = []
    for a in acct:
        leak.append(round((a.get("carried_value") or 0) + (a.get("shed_value") or 0)
                          + (a.get("unharvested_value") or 0)))
    wasted = [a.get("lastday_wasted_actions") for a in acct]
    print(f"--- {f.relative_to(Path(sys.argv[1]))}")
    print(f"  agents={s['agents']} seed={s['seed']} engine={s['engine']}")
    print(f"  money={s['final_money']} fert_floor_day={s['fertilizer_floor_day']}"
          f" fert_sold={s['fert_units_sold']} fert_rev={s['fert_revenue']}")
    print(f"  move={s['move_fraction']} noop={s['noop_fraction']} pass={s['pass_fraction']}")
    print(f"  endgame_leak$={leak} wasted_lastday={wasted}")
    if "--melon" in sys.argv:
        rows = [(d["day"], d["prices"]["MELON"],
                 d["cum_sales"][0]["MELON"] + d["cum_sales"][1]["MELON"]) for d in s["daily"]]
        print("  melon (day, price, cum_sold_both): " + " ".join(f"{r[0]}:{r[1]}/{r[2]}" for r in rows))
    if "--fert" in sys.argv:
        rows = [(d["day"], d["prices"]["FERTILIZER"],
                 d["cum_sales"][0]["FERTILIZER"] + d["cum_sales"][1]["FERTILIZER"]) for d in s["daily"]]
        print("  fert (day, price, cum_sold_both): " + " ".join(f"{r[0]}:{r[1]}/{r[2]}" for r in rows))
