"""Compare behavioural profiles produced by `scripts/behavior_profile.py`.

Cohorts are given as `name=<dir>[:seatspec]` where seatspec is
    `corpus`  -> the seat named in an index.csv `corpus_seat` column (their seat)
    `other`   -> the other seat of a replay (their opponent)
    `0` / `1` -> a literal seat
    `both`    -> both seats (default for our own mirror runs)

Every statistic is a per-episode mean over the cohort with the paired standard error, so a
difference is reportable rather than suggestive. Seeds are matched across cohorts wherever the
same seed appears in both, and the matched subset is used for the paired columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
            "FERTILIZER"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
NPH = 6
PHNAME = ["d0-4", "d5-9", "d10-14", "d15-19", "d20-24", "d25-29"]


def load_cohort(spec, seat_map):
    name, rest = spec.split("=", 1)
    if ":" in rest:
        d, sel = rest.rsplit(":", 1)
    else:
        d, sel = rest, "both"
    eps = []
    for f in sorted(Path(d).glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        eid = r.get("episode_id")
        if sel == "corpus":
            s = seat_map.get(int(eid)) if eid is not None else None
            if s is None:
                continue
            seats = [str(s)]
        elif sel == "other":
            s = seat_map.get(int(eid)) if eid is not None else None
            if s is None:
                continue
            seats = [str(1 - s)]
        elif sel == "both":
            seats = ["0", "1"]
        else:
            seats = [sel]
        for s in seats:
            if s not in r["seats"]:
                continue
            eps.append({"seed": r["seed"], "episode_id": eid, "seat": int(s),
                        "bank": r["local_rewards"][int(s)],
                        "opp_bank": r["local_rewards"][1 - int(s)],
                        "p": r["seats"][s]})
    return name, eps


def ms(vals):
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    m = sum(vals) / n
    if n < 2:
        return (m, 0.0)
    v = sum((x - m) ** 2 for x in vals) / (n - 1)
    return (m, math.sqrt(v / n))


def per_ep(eps, fn):
    return [fn(e) for e in eps]


def ckey(c, *want):
    """Sum a flattened Counter dict over keys matching a pattern of parts; None = wildcard."""
    tot = 0.0
    for k, v in c.items():
        parts = k.split("|")
        if len(parts) != len(want):
            continue
        if all(w is None or str(parts[i]) == str(w) for i, w in enumerate(want)):
            tot += v
    return tot


def table(title, rows, cohorts, fmt="{:.1f}"):
    print(f"\n### {title}")
    w = max(len(r[0]) for r in rows) + 2
    head = " " * w + "".join(f"{c:>22}" for c in cohorts)
    print(head)
    for label, cells in rows:
        line = f"{label:<{w}}"
        for m, s in cells:
            if isinstance(m, float) and math.isnan(m):
                line += f"{'-':>22}"
            else:
                line += f"{(fmt.format(m) + ' ±' + fmt.format(s)):>22}"
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cohorts", nargs="+")
    ap.add_argument("--index", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    seat_map = {}
    if args.index:
        for r in csv.DictReader(open(args.index, encoding="utf-8")):
            if r.get("tier") == "rank1":
                seat_map[int(r["episode_id"])] = int(r["seat"])

    C = [load_cohort(s, seat_map) for s in args.cohorts]
    names = [n for n, _ in C]
    data = {n: e for n, e in C}
    for n in names:
        b = per_ep(data[n], lambda e: e["bank"])
        m, s = ms(b)
        print(f"{n}: {len(data[n])} seats, bank {m:,.0f} +-{s:,.0f}")

    def row(label, fn, fmt=None):
        return (label, [ms(per_ep(data[n], fn)) for n in names])

    # ---------- 1. outcome ------------------------------------------------
    table("1. OUTCOME", [
        row("final bank", lambda e: e["bank"]),
        row("opponent bank", lambda e: e["opp_bank"]),
        row("margin", lambda e: e["bank"] - e["opp_bank"]),
        row("pie (both banks)", lambda e: e["bank"] + e["opp_bank"]),
        row("extraction share %", lambda e: 100 * e["bank"] / max(1.0, e["bank"] + e["opp_bank"])),
    ], names, "{:,.0f}")

    # ---------- 2. build-out ----------------------------------------------
    def day_stat(e, day, key, default=0.0):
        for d in e["p"]["daily"]:
            if d["day"] == day:
                return float(d.get(key, default))
        return default

    def tiles_at(e, day, prefix):
        for d in e["p"]["daily"]:
            if d["day"] == day:
                return float(sum(v for k, v in d["tiles"].items() if k.startswith(prefix)))
        return 0.0

    def mid_stat(e, day, key, default=0.0):
        for d in e["p"].get("midday", []):
            if d["day"] == day:
                return float(d.get(key, default))
        return default

    def mid_tiles(e, day, prefix):
        for d in e["p"].get("midday", []):
            if d["day"] == day:
                return float(sum(v for k, v in d["tiles"].items() if k.startswith(prefix)))
        return 0.0

    rows = []
    for day in (3, 7, 12, 18, 24, 29):
        rows.append(row(f"hands @ d{day} h12", lambda e, d=day: mid_stat(e, d, "hands")))
    rows.append(row("hires bought (season)",
                    lambda e: ckey(e["p"]["buy"], None, "HIRE", "-")))
    rows.append(row("hire spend $",
                    lambda e: ckey(e["p"]["buy_cost"], None, "HIRE", "-")))
    for day in (5, 10, 15, 20, 29):
        rows.append(row(f"quadrants @ d{day}",
                        lambda e, d=day: day_stat(e, d, "quadrants", 1.0)))
    rows.append(row("land spend $", lambda e: ckey(e["p"]["buy_cost"], None, "BUY_LAND", "-")))
    table("2. BUILD-OUT (labour + land)", rows, names, "{:.2f}")

    # ---------- 3. tile allocation ----------------------------------------
    rows = []
    for day in (5, 12, 20, 29):
        rows.append(row(f"planted tiles @ d{day}", lambda e, d=day: tiles_at(e, d, "plant_")))
        rows.append(row(f"animal tiles  @ d{day}", lambda e, d=day: tiles_at(e, d, "animal_")))
        rows.append(row(f"empty owned   @ d{day}", lambda e, d=day: tiles_at(e, d, "EMPTY")))
    table("3. TILE ALLOCATION", rows, names, "{:.2f}")

    rows = []
    for c in CROPS:
        rows.append(row(f"plant_{c} @ d20",
                        lambda e, c=c: next((float(d["tiles"].get("plant_" + c, 0))
                                             for d in e["p"]["daily"] if d["day"] == 20), 0.0)))
    for a in ANIMALS:
        rows.append(row(f"animal_{a} @ d20",
                        lambda e, a=a: next((float(d["tiles"].get("animal_" + a, 0))
                                             for d in e["p"]["daily"] if d["day"] == 20), 0.0)))
    table("3b. PORTFOLIO AT DAY 20", rows, names, "{:.2f}")

    # ---------- 4. cadence -------------------------------------------------
    ops = ["MOVE", "PASS", "WATER", "HARVEST", "FEED", "CARE", "COLLECT_FERTILIZER",
           "PICKUP", "DROP", "PLANT", "PLACE", "DIG", "FERTILIZE", "BUILD_COOP",
           "BUILD_PASTURE"]
    rows = []
    rows.append(row("unit-turns (season)", lambda e: ckey(e["p"]["acts"], None, None)))
    for o in ops:
        rows.append(row(f"{o} %", lambda e, o=o: 100 * ckey(e["p"]["acts"], None, o)
                        / max(1.0, ckey(e["p"]["acts"], None, None))))
    rows.append(row("silent no-op %", lambda e: 100 * ckey(e["p"]["acts_noop"], None, None)
                    / max(1.0, ckey(e["p"]["acts"], None, None))))
    table("4. UNIT-TURN CADENCE (share of all unit actions)", rows, names, "{:.2f}")

    rows = []
    for ph in range(NPH):
        rows.append(row(f"PASS % {PHNAME[ph]}",
                        lambda e, ph=ph: 100 * ckey(e["p"]["acts"], ph, "PASS")
                        / max(1.0, ckey(e["p"]["acts"], ph, None))))
    for ph in range(NPH):
        rows.append(row(f"HARVEST+COLLECT % {PHNAME[ph]}",
                        lambda e, ph=ph: 100 * (ckey(e["p"]["acts"], ph, "HARVEST")
                                                + ckey(e["p"]["acts"], ph, "COLLECT_FERTILIZER"))
                        / max(1.0, ckey(e["p"]["acts"], ph, None))))
    table("4b. IDLE AND HARVEST CADENCE BY PHASE", rows, names, "{:.2f}")

    # ---------- 5. market --------------------------------------------------
    rows = []
    for p in PRODUCTS:
        rows.append(row(f"{p} units sold", lambda e, p=p: ckey(e["p"]["sell"], None, p)))
    rows.append(row("TOTAL units sold", lambda e: ckey(e["p"]["sell"], None, None)))
    table("5. SELL VOLUME (units/season)", rows, names, "{:.1f}")

    rows = []
    for p in PRODUCTS:
        rows.append(row(f"{p} revenue $", lambda e, p=p: ckey(e["p"]["sell_rev"], None, p)))
    rows.append(row("TOTAL revenue $", lambda e: ckey(e["p"]["sell_rev"], None, None)))
    table("5b. SELL REVENUE ($/season)", rows, names, "{:,.0f}")

    rows = []
    for p in PRODUCTS:
        rows.append(row(f"{p} $/unit",
                        lambda e, p=p: (ckey(e["p"]["sell_rev"], None, p)
                                        / ckey(e["p"]["sell"], None, p))
                        if ckey(e["p"]["sell"], None, p) else float("nan")))
    rows.append(row("blended $/unit",
                    lambda e: ckey(e["p"]["sell_rev"], None, None)
                    / max(1.0, ckey(e["p"]["sell"], None, None))))
    table("5c. REALISED PRICE PER UNIT (the metering test)", rows, names, "{:.1f}")

    rows = []
    for ph in range(NPH):
        rows.append(row(f"revenue {PHNAME[ph]} $",
                        lambda e, ph=ph: ckey(e["p"]["sell_rev"], ph, None)))
    for ph in range(NPH):
        rows.append(row(f"units {PHNAME[ph]}",
                        lambda e, ph=ph: ckey(e["p"]["sell"], ph, None)))
    rows.append(row("day-29 units",
                    lambda e: ckey(e["p"]["sell_day"], 29, None)))
    rows.append(row("day-29 revenue $",
                    lambda e: ckey(e["p"]["sell_rev_day"], 29, None)))
    rows.append(row("day-29 $/unit",
                    lambda e: (ckey(e["p"]["sell_rev_day"], 29, None)
                               / ckey(e["p"]["sell_day"], 29, None))
                    if ckey(e["p"]["sell_day"], 29, None) else float("nan")))
    table("6. SELL TIMING", rows, names, "{:,.1f}")

    rows = []
    rows.append(row("SELL orders/season",
                    lambda e: ckey(e["p"]["orders"], None, "SELL")))
    rows.append(row("mean SELL order size",
                    lambda e: _mean_order(e["p"]["order_size"], "SELL")))
    rows.append(row("max SELL order size",
                    lambda e: _max_order(e["p"]["order_size"], "SELL")))
    rows.append(row("units per SELL order",
                    lambda e: ckey(e["p"]["sell"], None, None)
                    / max(1.0, ckey(e["p"]["orders"], None, "SELL"))))
    rows.append(row("BUY_PRODUCT orders", lambda e: ckey(e["p"]["orders"], None, "BUY_PRODUCT")))
    rows.append(row("BUY_SEED orders", lambda e: ckey(e["p"]["orders"], None, "BUY_SEED")))
    rows.append(row("BUY_ANIMAL orders", lambda e: ckey(e["p"]["orders"], None, "BUY_ANIMAL")))
    rows.append(row("HIRE orders", lambda e: ckey(e["p"]["orders"], None, "HIRE")))
    rows.append(row("total orders emitted", lambda e: ckey(e["p"]["orders"], None, None)))
    rows.append(row("orders/turn", lambda e: ckey(e["p"]["orders"], None, None) / 719.0))
    table("7. ORDER BOOK USE", rows, names, "{:.2f}")

    # ---------- 8. inputs ---------------------------------------------------
    rows = []
    for c in CROPS:
        rows.append(row(f"seed {c}", lambda e, c=c: ckey(e["p"]["buy"], None, "BUY_SEED", c)))
    for a in ANIMALS:
        rows.append(row(f"animal {a}", lambda e, a=a: ckey(e["p"]["buy"], None, "BUY_ANIMAL", a)))
    rows.append(row("WHEAT bought", lambda e: ckey(e["p"]["buy"], None, "BUY_PRODUCT", "WHEAT")))
    rows.append(row("FERTILIZER bought",
                    lambda e: ckey(e["p"]["buy"], None, "BUY_PRODUCT", "FERTILIZER")))
    rows.append(row("total buy spend $", lambda e: ckey(e["p"]["buy_cost"], None, None, None)))
    table("8. INPUT PURCHASES (units/season)", rows, names, "{:.2f}")

    # ---------- 9. terminal -------------------------------------------------
    rows = []
    rows.append(row("shed units @ d29",
                    lambda e: next((float(sum(d["shed"].values()))
                                    for d in e["p"]["daily"] if d["day"] == 29), 0.0)))
    rows.append(row("carried units @ d29",
                    lambda e: next((float(d["carried"]) for d in e["p"]["daily"]
                                    if d["day"] == 29), 0.0)))
    rows.append(row("seeds left @ d29",
                    lambda e: next((float(sum(d["seeds"].values())) for d in e["p"]["daily"]
                                    if d["day"] == 29), 0.0)))
    rows.append(row("bank @ d20", lambda e: day_stat(e, 20, "money")))
    rows.append(row("bank @ d25", lambda e: day_stat(e, 25, "money")))
    rows.append(row("bank @ d29", lambda e: day_stat(e, 29, "money")))
    rows.append(row("final - d29 bank",
                    lambda e: e["bank"] - day_stat(e, 29, "money")))
    table("9. TERMINAL / LIQUIDATION", rows, names, "{:,.1f}")

    # ---------- 10. shared market state ------------------------------------
    rows = []
    for p in ("EGG", "TOMATO", "CARROT", "MELON", "STRAWBERRY", "MILK", "FERTILIZER"):
        rows.append(row(f"{p} quote @ d29",
                        lambda e, p=p: next((float(m["price"][p]) for m in e["p"]["market_daily"]
                                             if m["day"] == 29), float("nan"))))
    for p in ("EGG", "TOMATO", "MELON", "STRAWBERRY"):
        rows.append(row(f"{p} pool inv @ d29",
                        lambda e, p=p: next((float(m["inv"][p]) for m in e["p"]["market_daily"]
                                             if m["day"] == 29), float("nan"))))
    rows.append(row("shops unlocked @ d29",
                    lambda e: next((float(len(m["shops"])) for m in e["p"]["town_daily"]
                                    if m["day"] == 29), float("nan"))))
    table("10. SHARED MARKET AT THE BELL", rows, names, "{:.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {n: {"n": len(data[n]),
                 "bank": ms(per_ep(data[n], lambda e: e["bank"]))} for n in names}, indent=1),
            encoding="utf-8")
    return 0


def _mean_order(os_, kind):
    tot = cnt = 0.0
    for k, (n, mean, mx) in os_.items():
        if k.split("|")[0] == kind:
            tot += n * mean
            cnt += n
    return tot / cnt if cnt else float("nan")


def _max_order(os_, kind):
    return max((mx for k, (n, mean, mx) in os_.items() if k.split("|")[0] == kind),
               default=float("nan"))


if __name__ == "__main__":
    raise SystemExit(main())
