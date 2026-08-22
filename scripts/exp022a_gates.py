"""exp-022a -- the paired before/after census gate table.

Reads two `exp021_census.py` shards recorded on the SAME seed band with the same opponent
and the same weights, differing only in `liquidate_order`, and prints the gate table from
`workshop/deliverables/2026-08-22-bc-representation-review/integration-ruling.md` I-1.

Gates (point estimate, Wilson 95% CI reported beside it):

    BUY_LAND orders failing entirely     42.9% -> <= 10%   (min-n 60 orders)
    BUY_ANIMAL|COW orders failing entirely 44.0% -> <= 15%  (min-n 150 orders)
    all orders failing in full            3.08% -> <= 2.0%  (reported, not blocking)

Tripwire: BUY_LAND failure still > 20% after the reorder => the cash diagnosis is wrong;
stop the line and take another census, do NOT open a tuning arm.

An "order" is one (step, type, item) request in the post-wrapper stream; it fails ENTIRELY
when the engine committed zero of its units (`orders_zero`), partially when it committed
some but not all (`orders_partial`).

Usage:
    python scripts/exp022a_gates.py --ctl data/results/exp-022a/ctl.json \
        --arm data/results/exp-022a/arm.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

GATES = [
    # key in orders_*, label, baseline %, target %, min-n, blocking
    ("BUY_LAND|-", "BUY_LAND failing entirely", 42.9, 10.0, 60, True),
    ("BUY_ANIMAL|COW", "BUY_ANIMAL|COW failing entirely", 44.0, 15.0, 150, True),
]
TRIPWIRE = ("BUY_LAND|-", 20.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(paths):
    """Union of one or more shards; the ruling's band extension is quoted on the union."""
    shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    orders = {s.get("liquidate_order") for s in shards}
    assert len(orders) == 1, f"shards disagree on liquidate_order: {orders}"
    rows = [r for s in shards for r in s["rows"]]
    par = {"checked": sum((s.get("parity") or {}).get("checked", 0) for s in shards),
           "mismatches": [m for s in shards for m in (s.get("parity") or {}).get("mismatches", [])]}
    return {"tag": "+".join(s["tag"] for s in shards),
            "seeds": [s["seeds"] for s in shards],
            "episodes": sum(s["episodes"] for s in shards),
            "liquidate_order": orders.pop(), "opponent": shards[0]["opponent"],
            "parity": par, "rows": rows}


def counts(shard):
    """key -> (zero, partial, full)."""
    z, p, f = Counter(), Counter(), Counter()
    for r in shard["rows"]:
        z.update(r["orders_zero"])
        p.update(r["orders_partial"])
        f.update(r["orders_full"])
    return z, p, f


def rate(z, p, f, key):
    n = z.get(key, 0) + p.get(key, 0) + f.get(key, 0)
    return z.get(key, 0), n


def fail_reasons(shard, prefix):
    c = Counter()
    for r in shard["rows"]:
        for k, v in r["order_fails"].items():
            if k.startswith(prefix):
                c[k] += v
    return c


def paired_delta(shard):
    """Mean over seed-pairs of (us - them), averaged over the two seats of the seed.

    A mirror shard (agent vs a byte-identical opponent) gives exactly 0.0, which is what
    makes the control's Delta the natural zero for `change in paired Delta/bank`.
    """
    by_seed = {}
    for r in shard["rows"]:
        by_seed.setdefault(r["seed"], []).append(r["us"] - r["them"])
    per = [sum(v) / len(v) for v in by_seed.values()]
    m = sum(per) / len(per)
    if len(per) > 1:
        sd = math.sqrt(sum((x - m) ** 2 for x in per) / (len(per) - 1))
        se = sd / math.sqrt(len(per))
    else:
        sd = se = 0.0
    return m, sd, se, len(per)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctl", required=True, nargs="+")
    ap.add_argument("--arm", required=True, nargs="+")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="further (key,label) order classes to report, e.g. BUY_SEED|WHEAT")
    args = ap.parse_args(argv)

    ctl, arm = load(args.ctl), load(args.arm)
    for name, sh in (("ctl", ctl), ("arm", arm)):
        par = sh.get("parity") or {}
        print(f"{name}: tag={sh['tag']} seeds={sh['seeds']} episodes={sh['episodes']} "
              f"liquidate_order={sh.get('liquidate_order')} opponent={sh['opponent']} "
              f"parity={par.get('checked')}/{sh['episodes']} "
              f"mismatches={len(par.get('mismatches', []))}")
    print()

    cz, cp, cf = counts(ctl)
    az, ap_, af = counts(arm)

    print(f"{'gate':34s} {'baseline':>22s} {'after':>22s} {'target':>8s} {'min-n':>6s}  verdict")
    verdicts = {}
    for key, label, base_pct, target, min_n, blocking in GATES:
        bk, bn = rate(cz, cp, cf, key)
        ak, an = rate(az, ap_, af, key)
        blo, bhi = wilson(bk, bn)
        alo, ahi = wilson(ak, an)
        bp = 100 * bk / bn if bn else float("nan")
        appct = 100 * ak / an if an else float("nan")
        ok = an >= min_n and appct <= target
        why = "PASS" if ok else ("MIN-N FAIL" if an < min_n else "FAIL")
        verdicts[key] = (ok, appct, an)
        print(f"{label:34s} {bp:6.1f}% ({bk:4d}/{bn:4d}) {appct:6.1f}% ({ak:4d}/{an:4d}) "
              f"{target:7.1f}% {min_n:6d}  {why}")
        print(f"{'':34s} {'CI ' + f'[{blo:.1f},{bhi:.1f}]':>22s} "
              f"{'CI ' + f'[{alo:.1f},{ahi:.1f}]':>22s}")

    # all orders failing in full (reported, not blocking)
    def total(z, p, f):
        n = sum(z.values()) + sum(p.values()) + sum(f.values())
        return sum(z.values()), n
    bk, bn = total(cz, cp, cf)
    ak, an = total(az, ap_, af)
    blo, bhi = wilson(bk, bn)
    alo, ahi = wilson(ak, an)
    print(f"{'ALL orders failing in full':34s} {100*bk/bn:6.2f}% ({bk:4d}/{bn:4d}) "
          f"{100*ak/an:6.2f}% ({ak:4d}/{an:4d}) {2.0:7.1f}% {'-':>6s}  "
          f"{'pass' if 100*ak/an <= 2.0 else 'miss'} (reported)")
    print(f"{'':34s} {'CI ' + f'[{blo:.2f},{bhi:.2f}]':>22s} "
          f"{'CI ' + f'[{alo:.2f},{ahi:.2f}]':>22s}")

    key, thresh = TRIPWIRE
    ok, pct, n = verdicts[key]
    print()
    print(f"TRIPWIRE  BUY_LAND failing entirely > {thresh:.0f}%  ->  "
          f"{pct:.1f}%  {'FIRED (cash diagnosis wrong; census, not tuning)' if pct > thresh else 'not fired'}")

    # order-class detail: every class that spends money
    print()
    print(f"{'class':22s} {'ctl fail%':>12s} {'arm fail%':>12s} {'ctl n':>7s} {'arm n':>7s}")
    keys = sorted(set(list(cz) + list(cp) + list(cf) + list(az) + list(ap_) + list(af)))
    for k in keys:
        bk, bn = rate(cz, cp, cf, k)
        ak2, an2 = rate(az, ap_, af, k)
        if bn == 0 and an2 == 0:
            continue
        print(f"{k:22s} {(100*bk/bn if bn else 0):11.1f}% {(100*ak2/an2 if an2 else 0):11.1f}% "
              f"{bn:7d} {an2:7d}")

    # failure reasons for the two gated classes
    print()
    for prefix in ("BUY_LAND", "BUY_ANIMAL", "BUY_SEED", "HIRE", "BUY_PRODUCT", "SELL"):
        b = fail_reasons(ctl, prefix)
        a = fail_reasons(arm, prefix)
        if not b and not a:
            continue
        print(f"fail reasons {prefix:12s} ctl={dict(b)}  arm={dict(a)}")

    # paired bank
    print()
    for name, sh in (("ctl", ctl), ("arm", arm)):
        m, sd, se, n = paired_delta(sh)
        print(f"paired Delta/bank {name}: {m:+9.1f}  sd {sd:8.1f}  se {se:7.1f}  "
              f"n {n} seed-pairs   95% CI [{m-1.96*se:+.1f}, {m+1.96*se:+.1f}]")
    mc, _, _, _ = paired_delta(ctl)
    ma, _, sea, na = paired_delta(arm)
    print(f"change in paired Delta/bank (arm - ctl): {ma-mc:+.1f}  "
          f"95% CI [{ma-mc-1.96*sea:+.1f}, {ma-mc+1.96*sea:+.1f}]  (1 SE = {sea:.1f})")

    # units sold / season, both arms
    print()
    for name, sh in (("ctl", ctl), ("arm", arm)):
        tot = Counter()
        for r in sh["rows"]:
            tot.update(r["final_sell_units"])
        n = len(sh["rows"])
        print(f"units EMITTED to sell/season {name}: {sum(tot.values())/n:8.1f}  "
              + " ".join(f"{k}={v/n:.1f}" for k, v in sorted(tot.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
