"""Paired Delta/bank for one arena run, with the seat-swap pairing done explicitly.

`evaluate.py` reports the two banks' unpaired means and a Wilson CI on the win rate. The
integration ruling's primary gate is on the PAIRED difference and its own SE ("change in
paired Delta/bank >= 0, with the 95% CI on the change excluding a decline of 1 SE"), so the
pairing has to be reconstructed from the run's rows: one seed is played twice with the seats
swapped, and the seed's paired difference is the mean of (our bank - their bank) over the
two seats. sigma comes from that matchup's own paired differences, per the standing
conventions.

Usage:
    python scripts/exp022a_paired.py --run 20260822-091440-d3f51c --agent <id> --opponent <id>
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "data" / "arena" / "arena.sqlite"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--opponent", required=True)
    args = ap.parse_args(argv)

    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    cols = {r["name"] for r in con.execute("PRAGMA table_info(games)")}
    rows = [dict(r) for r in con.execute("SELECT * FROM games WHERE run_id=?", (args.run,))]
    rows = [r for r in rows if {r["p0"], r["p1"]} == {args.agent, args.opponent}]
    if not rows:
        print(f"no rows for run {args.run}; columns are {sorted(cols)}")
        return 2

    by_seed = {}
    wins = 0.0
    for r in rows:
        us_is_p0 = r["p0"] == args.agent
        us = r["money0"] if us_is_p0 else r["money1"]
        them = r["money1"] if us_is_p0 else r["money0"]
        res = float(r["result0"]) if us_is_p0 else 1.0 - float(r["result0"])
        wins += res
        by_seed.setdefault(r["seed"], []).append(float(us) - float(them))

    per = [sum(v) / len(v) for v in by_seed.values()]
    unpaired = [d for v in by_seed.values() for d in v]
    n = len(per)
    m = sum(per) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in per) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    both = sum(1 for v in by_seed.values() if len(v) == 2)

    print(f"run            : {args.run}")
    print(f"agent          : {args.agent}")
    print(f"opponent       : {args.opponent}")
    print(f"games          : {len(rows)}   seeds {min(by_seed)}-{max(by_seed)}   "
          f"seed-pairs with BOTH seats: {both}/{n}")
    print(f"win rate       : {100*wins/len(rows):.1f}%  ({wins:g}/{len(rows)})")
    print(f"paired D/bank  : {m:+.1f}   sd {sd:.1f}   SE {se:.1f}   n {n} seed-pairs")
    print(f"  95% CI       : [{m-1.96*se:+.1f}, {m+1.96*se:+.1f}]")
    print(f"  t            : {m/se if se else float('nan'):+.2f}")
    print()
    print("PRIMARY GATE  change in paired D/bank >= 0, CI excluding a decline of 1 SE")
    print(f"  baseline D   : +0.0  (the frozen base against itself is an exact mirror)")
    print(f"  change       : {m:+.1f}   1 SE = {se:.1f}")
    lo = m - 1.96 * se
    ge0 = m >= 0
    excl = lo > -se
    print(f"  D >= 0                              : {'PASS' if ge0 else 'FAIL'}")
    print(f"  CI lower {lo:+.1f} > -1 SE ({-se:+.1f})   : {'PASS' if excl else 'FAIL'}")
    print(f"  VERDICT                             : {'PASS' if (ge0 and excl) else 'FAIL'}")
    print()
    print(f"(unpaired per-game diff mean {sum(unpaired)/len(unpaired):+.1f}, "
          f"sd {math.sqrt(sum((x-sum(unpaired)/len(unpaired))**2 for x in unpaired)/(len(unpaired)-1)):.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
