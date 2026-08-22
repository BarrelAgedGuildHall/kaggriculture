"""Roll up `exp021_census.py` shards into the section-3 replacement table.

Every number is a per-episode mean over the shard's paired seat-pairs unless it is explicitly
a rate. Rates are pooled (sum numerator / sum denominator), not the mean of per-episode rates.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

PHASES = ("early", "mid", "late", "endgame")


def load(paths):
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for r in d["rows"]:
            r["_tag"] = d["tag"]
            out.append(r)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def pool(rows, path, key=None):
    c = Counter()
    for r in rows:
        d = r
        for step in path:
            d = d.get(step, {})
        if key is None:
            for k, v in d.items():
                c[k] += v
        else:
            c[key] += d.get(key, 0)
    return c


def per_ep(rows, path, key):
    tot = 0.0
    for r in rows:
        d = r
        for s in path:
            d = d.get(s, {})
        tot += d.get(key, 0)
    return tot / max(1, len(rows))


def report(tag, rows):
    n = len(rows)
    print(f"\n{'='*78}\n{tag}  --  {n} episodes  "
          f"(us {sum(r['us'] for r in rows)/n:,.0f} / them {sum(r['them'] for r in rows)/n:,.0f})\n{'='*78}")

    # ---- 1. unit-op stream -----------------------------------------------------
    print("\n[1] UNIT-OP STREAM  (who decides the op)")
    tot_ut = tot_ov = 0
    print(f"  {'phase':9} {'unit-turns':>11} {'net kept':>10} {'overridden':>11} {'rate%':>7} "
          f"{'endgame':>8} {'keepalive':>10} {'sanitise':>9}")
    for ph in PHASES:
        c = pool(rows, ["unit"]) if False else Counter()
        for r in rows:
            for k, v in r["unit"].get(ph, {}).items():
                c[k] += v
        ut, ov = c["unit_turns"], c["overridden"]
        tot_ut += ut
        tot_ov += ov
        print(f"  {ph:9} {ut:11,} {c['net_kept']:10,} {ov:11,} {100*ov/max(1,ut):7.2f} "
              f"{c['by_endgame']:8,} {c['by_keepalive']:10,} {c['by_sanitise']:9,}")
    lo, hi = wilson(tot_ut - tot_ov, tot_ut)
    print(f"  {'ALL':9} {tot_ut:11,} {tot_ut-tot_ov:10,} {tot_ov:11,} {100*tot_ov/tot_ut:7.2f}")
    print(f"  -> net-decided share of unit-turns: {100*(tot_ut-tot_ov)/tot_ut:.2f}% "
          f"[{lo:.2f}, {hi:.2f}]   (review estimate: 99.8%)")

    # per-head
    print("\n[2] OVERRIDE BY HEAD  (pooled over phases)")
    h = Counter()
    for r in rows:
        for ph in PHASES:
            for k, v in r["head_override"].get(ph, {}).items():
                h[k] += v
    for head, nk, ck in (("op", "op_n", "op_changed"), ("crop", "crop_n", "crop_changed"),
                         ("item", "item_n", "item_changed"), ("qty(unit)", "uqty_n", "uqty_changed")):
        d, c = h[nk], h[ck]
        print(f"  {head:10} decisions {d:9,}   overridden {c:7,}   {100*c/max(1,d):6.2f}%")

    # ---- 3. execution ----------------------------------------------------------
    print("\n[3] EXECUTED vs EMITTED  --  unit ops")
    ex = Counter()
    no = Counter()
    for r in rows:
        ex.update(r["exec_by_op"])
        no.update(r["noop_by_op"])
    allops = sorted(set(ex) | set(no), key=lambda k: -(ex[k] + no[k]))
    tot_e = sum(ex.values())
    tot_n = sum(no.values())
    tot_np = no.get("PASS", 0)
    print(f"  {'op':20} {'emitted':>10} {'executed':>10} {'no-op':>9} {'fail%':>7}")
    for k in allops:
        e, nn = ex[k], no[k]
        t = e + nn
        note = "  (PASS is a deliberate no-op)" if k == "PASS" else ""
        print(f"  {k:20} {t:10,} {e:10,} {nn:9,} {100*nn/max(1,t):7.2f}{note}")
    silent = tot_n - tot_np
    print(f"  -> {tot_e+tot_n:,} unit actions: {tot_e:,} executed, {tot_np:,} PASS, "
          f"{silent:,} silently failed ({100*silent/(tot_e+tot_n):.2f}%)")
    er = Counter()
    for r in rows:
        er.update(r["engine_rewrote"])
    print(f"  -> engine PLANT-block rewrites that `_sanitise` did not pre-empt: "
          f"{sum(er.values()):,} {dict(er) if er else ''}")

    # ---- 4. market orders ------------------------------------------------------
    print("\n[4] EXECUTED vs EMITTED  --  market orders (units)")
    rq, xq = Counter(), Counter()
    ofull, opart, ozero = Counter(), Counter(), Counter()
    fails = Counter()
    for r in rows:
        rq.update(r["order_req"]); xq.update(r["order_ex"])
        ofull.update(r["orders_full"]); opart.update(r["orders_partial"]); ozero.update(r["orders_zero"])
        fails.update(r["order_fails"])
    print(f"  {'order':24} {'req u':>9} {'exec u':>9} {'u fail%':>8} "
          f"{'orders':>8} {'full':>7} {'part':>6} {'zero':>6} {'ord fail%':>10}")
    T = [0, 0, 0, 0]
    for k in sorted(rq, key=lambda k: -rq[k]):
        f, p, z = ofull[k], opart[k], ozero[k]
        o = f + p + z
        T[0] += rq[k]; T[1] += xq[k]; T[2] += o; T[3] += p + z
        print(f"  {k:24} {rq[k]:9,} {xq[k]:9,} {100*(rq[k]-xq[k])/max(1,rq[k]):8.2f} "
              f"{o:8,} {f:7,} {p:6,} {z:6,} {100*(p+z)/max(1,o):10.2f}")
    lo, hi = wilson(T[3], T[2])
    print(f"  {'ALL':24} {T[0]:9,} {T[1]:9,} {100*(T[0]-T[1])/T[0]:8.2f} "
          f"{T[2]:8,} {T[2]-T[3]:7,} {'':6} {'':6} {100*T[3]/T[2]:10.2f}")
    print(f"  -> orders failing to execute in full: {100*T[3]/T[2]:.2f}% [{lo:.2f}, {hi:.2f}]"
          f"   (pre-registered defect threshold: >5%)")
    print(f"  -> failure reasons: {dict(fails)}")

    # ---- 5. origin of money ----------------------------------------------------
    print("\n[5] WHO DECIDES THE MONEY  (dollars committed, per episode)")
    mo = Counter(); uo = Counter()
    for r in rows:
        mo.update(r["money_by_origin"]); uo.update(r["units_by_origin"])
    inflow = defaultdict(float); outflow = defaultdict(float)
    for k, v in mo.items():
        origin, op = k.split("|")
        (inflow if op == "SELL" else outflow)[origin] += v
    tin = sum(inflow.values()); tout = sum(outflow.values())
    print("  INFLOW (SELL revenue) -- structurally 100% rule-issued: `_liquidate` deletes")
    print("  every net SELL each turn. The split below is COINCIDENCE with a net order,")
    print("  i.e. how often the rule happens to reproduce what the net asked for.")
    for o in ("net_exact", "net_requantified", "rule", "unmatched"):
        if inflow.get(o):
            print(f"    {o:18} ${inflow[o]/len(rows):11,.0f}/ep   {100*inflow[o]/tin:6.2f}%   "
                  f"{uo[o+'|SELL']/len(rows):8.1f} units/ep")
    print(f"    {'TOTAL':18} ${tin/len(rows):11,.0f}/ep")
    print("  OUTFLOW (buys / hires / land):")
    for o in ("net_exact", "net_requantified", "rule", "unmatched"):
        if outflow.get(o):
            print(f"    {o:18} ${outflow[o]/len(rows):11,.0f}/ep   {100*outflow[o]/tout:6.2f}%")
    print(f"    {'TOTAL':18} ${tout/len(rows):11,.0f}/ep")
    for k in sorted(mo, key=lambda k: -abs(mo[k])):
        print(f"      {k:32} ${mo[k]/len(rows):11,.0f}/ep  {uo[k]/len(rows):8.1f} u/ep")

    # ---- 6. the slot budget ----------------------------------------------------
    print("\n[6] `_liquidate` SLOT BUDGET  (MAX_MARKET_ORDERS = 10; buys+hires get priority)")
    print(f"  {'phase':9} {'turns':>7} {'kept/turn':>10} {'hires/turn':>11} {'room':>7} "
          f"{'cands':>7} {'trunc turns':>12} {'room=0':>8} {'$ truncated/ep':>15}")
    tt = Counter()
    for ph in PHASES:
        c = Counter()
        for r in rows:
            for k, v in r["liq"].get(ph, {}).items():
                c[k] += v
        tt.update(c)
        t = max(1, c["turns"])
        print(f"  {ph:9} {c['turns']/len(rows):7.0f} {c['kept']/t:10.2f} {c['hires_kept']/t:11.2f} "
              f"{c['room']/t:7.2f} {c['cand']/t:7.2f} {c['truncating_turns']/len(rows):12.1f} "
              f"{c['room_zero_turns']/len(rows):8.1f} {c['truncated_value']/len(rows):15,.0f}")
    t = max(1, tt["turns"])
    print(f"  {'ALL':9} {tt['turns']/len(rows):7.0f} {tt['kept']/t:10.2f} {tt['hires_kept']/t:11.2f} "
          f"{tt['room']/t:7.2f} {tt['cand']/t:7.2f} {tt['truncating_turns']/len(rows):12.1f} "
          f"{tt['room_zero_turns']/len(rows):8.1f} {tt['truncated_value']/len(rows):15,.0f}")
    print(f"  -> share of turns where the sell side got ZERO order slots: "
          f"{100*tt['room_zero_turns']/tt['turns']:.2f}%")
    print(f"  -> share of turns where at least one sellable product was truncated away: "
          f"{100*tt['truncating_turns']/tt['turns']:.2f}%")

    # ---- 7. what the net wanted to sell vs what went out ------------------------
    print("\n[7] SELL VOLUME: net's own emission vs `_liquidate`'s (units/episode)")
    nsu, fsu = Counter(), Counter()
    for r in rows:
        nsu.update(r["net_sell_units"]); fsu.update(r["final_sell_units"])
    print(f"  {'product':14} {'net asked':>10} {'rule sent':>10} {'ratio':>8}")
    for k in sorted(set(nsu) | set(fsu), key=lambda k: -(fsu[k])):
        a, b = nsu[k] / len(rows), fsu[k] / len(rows)
        print(f"  {k:14} {a:10.1f} {b:10.1f} {(b/a if a else float('nan')):8.2f}")
    print(f"  {'TOTAL':14} {sum(nsu.values())/len(rows):10.1f} {sum(fsu.values())/len(rows):10.1f}")

    # ---- 8. emitted order composition -----------------------------------------
    print("\n[8] EMITTED ORDER COMPOSITION (stream B, share of order slots used)")
    comp = Counter()
    for r in rows:
        for ph in PHASES:
            for k, v in r["order_stage"].get(ph, {}).items():
                if k.startswith("origin_") and k.count("_") >= 2:
                    parts = k.split("_")
                    comp[parts[-1] if parts[-1] not in ("exact", "requantified") else None] += 0
    ordtypes = Counter()
    for r in rows:
        for k, v in r["order_req"].items():
            ordtypes[k.split("|")[0]] += 0
    # simpler: count final orders per type from orders_full/partial/zero
    per_type = Counter()
    for r in rows:
        for d in (r["orders_full"], r["orders_partial"], r["orders_zero"]):
            for k, v in d.items():
                per_type[k.split("|")[0]] += v
    tot = sum(per_type.values())
    for k, v in per_type.most_common():
        print(f"  {k:16} {v/len(rows):8.1f} orders/ep   {100*v/tot:6.2f}% of slots used")
    print(f"  {'TOTAL':16} {tot/len(rows):8.1f} orders/ep")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--combined", action="store_true")
    args = ap.parse_args(argv)
    per = defaultdict(list)
    allrows = []
    for p in args.shards:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for r in d["rows"]:
            per[d["tag"]].append(r)
            allrows.append(r)
    for tag, rows in per.items():
        report(tag, rows)
    if args.combined and len(per) > 1:
        report("COMBINED", allrows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
