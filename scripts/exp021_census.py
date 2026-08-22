"""exp-021 -- override / executed-vs-emitted census for the Team A BC policy.

Ruling: `workshop/deliverables/2026-08-22-bc-representation-review/review.md` section 7,
row exp-021. The review's section 3 table ("the net makes ~99.8% of the decisions and none of
the revenue decisions") is *estimated from the record*; this script replaces every cell of it
with a measurement.

THREE STREAMS, all recorded on the same episode, all observational:

  A  pre-wrapper emission   what the net emits  -- `_unit_op` per unit, `_market` for orders,
                            captured before `_apply_safety` runs.
  B  post-wrapper action    what `act` returns  -- captured after `_endgame` / `_keep_alive` /
                            `_liquidate` / `_sanitise`, each stage recorded separately so an
                            override is attributed to the rule that made it.
  C  engine-executed        what the engine actually did with B.

Stream C is the part that needs engine knowledge, and the engine makes it exact:

  * `_apply_unit_action` (kaggriculture.py:312) documents itself as "Invalid / illegal actions
    are silent no-ops", and *every* failure path in it returns before mutating anything. So a
    fingerprint of the mutable state a unit action can touch -- the unit's position, its
    inventory, the private shed, the private seed counts and the tile it stands on -- taken
    either side of the call is an exact executed/no-op oracle. (`PASS` is a deliberate no-op
    and is counted in its own bucket, never as a failure.)
  * `interpreter` (line 920) additionally drops *all* `PLANT` requests for a crop whose demand
    in one turn exceeds the seeds held. It rewrites those actions to `["PASS"]` before calling
    `_apply_unit_action`, so comparing the action the engine saw with stream B measures how
    often `policy._sanitise` failed to pre-empt it.
  * a market order for n units is committed one unit at a time by `_commit_unit`
    (line 652); the first unit that returns False aborts the rest of that order
    (`_process_market` line 623). So executed/requested per order is a count of True returns,
    and the failure reason is recoverable from the same pre-conditions the engine tests.
  * `_do_hire` / `_do_buy_land` fail silently on insufficient funds; a zero money delta is the
    tell (the `d1_probe` recipe).

Identity map: `env.interpreter` is wrapped so the farm-object -> player-index map is rebuilt at
the top of every step, exactly as `tools/pnl_census.py` and `scripts/d1_probe.py` do. Wrapping
`_process_market` instead would be wrong -- the farm dicts are rebuilt per step.

Nothing here takes a decision. `--check-parity` replays each seed uninstrumented and asserts
byte-identical banks.

Usage:
    python scripts/exp021_census.py --seeds 4256-4271 --opponent base --tag base \
        --out data/results/exp-021/base.json --workers 6
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Day buckets. Day 29 is its own phase because `_endgame` only exists there.
PHASES = ("early", "mid", "late", "endgame")


def phase_of(day):
    if day >= 29:
        return "endgame"
    if day >= 20:
        return "late"
    if day >= 10:
        return "mid"
    return "early"


def _order_key(o):
    """(type, item) -- the granularity at which the engine and the policy both address an
    order. `BUY_LAND` and `HIRE` carry no item."""
    if not isinstance(o, list) or not o:
        return None
    if o[0] in ("BUY_LAND", "HIRE"):
        return (o[0], "-")
    if len(o) < 2:
        return (o[0], "-")
    return (o[0], o[1])


def _order_qty(o):
    if not isinstance(o, list) or not o:
        return 0
    if o[0] in ("BUY_LAND", "HIRE"):
        return 1
    return int(o[2]) if len(o) >= 3 else 1


# --------------------------------------------------------------------------- policy side
class PolicyRecorder:
    """Instance-level wrappers on one `BCPolicy`. No class is patched, so the opponent -- which
    may be the same class -- is untouched, and no method body is duplicated."""

    def __init__(self, pol, F):
        self.pol = pol
        self.turns = []          # one dict per act()
        self._pre_ops = []
        self._cur = None

        orig_unit_op = pol._unit_op
        orig_market = pol._market
        orig_endgame = pol._endgame
        orig_keep = pol._keep_alive
        orig_liq = pol._liquidate
        orig_san = pol._sanitise
        orig_act = pol.act

        def unit_op(ctx, u, claimed):
            op, claims = orig_unit_op(ctx, u, claimed)
            self._pre_ops.append(copy.deepcopy(op))
            return op, claims

        def market(ctx):
            out = orig_market(ctx)
            self._cur["orders_net"] = copy.deepcopy(out)
            return out

        def endgame(ctx, ops, orders):
            a, b = orig_endgame(ctx, copy.deepcopy(ops), copy.deepcopy(orders))
            self._cur["ops_endgame"] = copy.deepcopy(a)
            self._cur["orders_endgame"] = copy.deepcopy(b)
            return a, b

        def keep_alive(ctx, ops, orders):
            a, b = orig_keep(ctx, copy.deepcopy(ops), copy.deepcopy(orders))
            self._cur["ops_keepalive"] = copy.deepcopy(a)
            self._cur["orders_keepalive"] = copy.deepcopy(b)
            return a, b

        def liquidate(ctx, orders):
            # `_liquidate` is `kept + sells[:room]` with `room = MAX - len(kept)`, and `kept`
            # holds every non-SELL order -- HIRE included, one entry per hand. So the buy/hire
            # side has absolute priority over the sell side for the 10 order slots. Record the
            # candidate set and what the truncation costs, because that price is invisible
            # anywhere else.
            n_animals = sum(1 for row in ctx.tiles for t in row
                            if isinstance(t, dict) and t.get("animal"))
            reserve = {"WHEAT": 3 * n_animals}
            cand = []
            for item in F.PRODUCTS:
                have = int(ctx.shed.get(item, 0)) - reserve.get(item, 0)
                if have > 0:
                    cand.append((int(ctx.prices.get(item, 1)) * have, item, have))
            cand.sort(key=lambda t: -t[0])
            kept = [o for o in orders if o[0] != "SELL"]
            room = max(0, F.MAX_MARKET_ORDERS - len(kept))
            self._cur["liq"] = {
                "kept": len(kept),
                "hires_kept": sum(1 for o in kept if o[0] == "HIRE"),
                "room": room,
                "cand": len(cand),
                "truncated": max(0, len(cand) - room),
                "truncated_value": float(sum(v for v, _, _ in cand[room:])),
                "cand_value": float(sum(v for v, _, _ in cand)),
                "shed_units": int(sum(int(v) for v in ctx.shed.values())),
            }
            b = orig_liq(ctx, copy.deepcopy(orders))
            self._cur["orders_liquidate"] = copy.deepcopy(b)
            return b

        def sanitise(ctx, ops, orders):
            a, b = orig_san(ctx, copy.deepcopy(ops), copy.deepcopy(orders))
            self._cur["ops_final"] = copy.deepcopy(a)
            self._cur["orders_final"] = copy.deepcopy(b)
            return a, b

        def act(obs):
            self._pre_ops = []
            self._cur = {"step": int(obs.get("step", -1)),
                         "day": int(obs.get("day", 0)),
                         "hour": int(obs.get("hour", 0)),
                         "money": float(obs["farms"][obs["player"]]["money"])}
            out = orig_act(obs)
            self._cur["ops_net"] = self._pre_ops
            self._cur.setdefault("ops_endgame", self._cur["ops_net"])
            self._cur.setdefault("orders_endgame", self._cur.get("orders_net", []))
            self._cur.setdefault("ops_keepalive", self._cur["ops_endgame"])
            self._cur.setdefault("orders_keepalive", self._cur["orders_endgame"])
            self._cur.setdefault("orders_liquidate", self._cur["orders_keepalive"])
            self.turns.append(self._cur)
            return out

        pol._unit_op = unit_op
        pol._market = market
        pol._endgame = endgame
        pol._keep_alive = keep_alive
        pol._liquidate = liquidate
        pol._sanitise = sanitise
        pol.act = act


# --------------------------------------------------------------------------- engine side
def _fingerprint(farm, private, idx):
    """Everything `_apply_unit_action` can mutate for unit `idx`, cheaply."""
    pos = tuple(farm["farmer"]) if idx == 0 else tuple(farm["hands"][idx - 1])
    invs = private["inventories"]
    inv = invs[idx] if idx < len(invs) else {}
    x, y = pos
    tile = farm["tiles"][y][x]
    return (pos, repr(sorted(inv.items())), repr(sorted(private["shed"].items())),
            repr(sorted(private["seeds"].items())), repr(tile), float(farm["money"]),
            len(farm["hands"]))


def _fail_reason(K, op, item, price, farm, private, shed_capacity):
    if op == "SELL":
        return "no_stock" if private["shed"].get(item, 0) <= 0 else "other"
    if op == "BUY_PRODUCT":
        if farm["money"] < price:
            return "no_cash"
        if sum(private["shed"].values()) >= shed_capacity:
            return "shed_full"
        return "other"
    if op == "BUY_SEED":
        return "no_cash" if farm["money"] < price else "other"
    if op == "BUY_ANIMAL":
        if farm["money"] < price:
            return "no_cash"
        if sum(private["shed"].values()) >= shed_capacity:
            return "shed_full"
        return "other"
    return "malformed"


def _one(job):
    from arena import engine as arena_engine          # noqa: E402
    ke = arena_engine.import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as K  # noqa: E402
    sys.path.insert(0, str(REPO_ROOT / "agents" / "team_a" / "bc"))
    import policy as P                                # noqa: E402

    seed, seat = int(job["seed"]), int(job["seat"])
    steps = int(job.get("steps", 720))
    shed_capacity = 100

    fmap = {}
    state = {"step": -1}
    unit_rows = []       # (step, idx, op_name, executed)  -- our seat only
    engine_ops = defaultdict(dict)   # step -> idx -> action the engine saw
    commits = []         # (step, op, item, price, ok, reason) -- our seat only
    atomic = []          # (step, kind, spent)

    orig_unit = K._apply_unit_action
    orig_commit = K._commit_unit
    orig_hire = K._do_hire
    orig_land = K._do_buy_land

    def apply_unit(farm, private, idx, action, board_size, day, turns_per_day,
                   shed_capacity_=100):
        if fmap.get(id(farm), -1) != seat:
            return orig_unit(farm, private, idx, action, board_size, day, turns_per_day,
                             shed_capacity_)
        before = _fingerprint(farm, private, idx)
        out = orig_unit(farm, private, idx, action, board_size, day, turns_per_day,
                        shed_capacity_)
        after = _fingerprint(farm, private, idx)
        name = action[0] if isinstance(action, list) and action else "??"
        engine_ops[state["step"]][idx] = copy.deepcopy(action)
        unit_rows.append((state["step"], idx, name, before != after))
        return out

    def commit(op, item, price, farm, private, market, shed_capacity_=100):
        mine = fmap.get(id(farm), -1) == seat
        if not mine:
            return orig_commit(op, item, price, farm, private, market, shed_capacity_)
        reason = None
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity_)
        if not ok:
            reason = _fail_reason(K, op, item, price, farm, private, shed_capacity_)
        commits.append((state["step"], op, item, float(price), bool(ok), reason))
        return ok

    def do_hire(farm, private, board_size, mult=K.FARM_HAND_COST_MULT):
        mine = fmap.get(id(farm), -1) == seat
        before = float(farm["money"])
        out = orig_hire(farm, private, board_size, mult)
        if mine:
            atomic.append((state["step"], "HIRE", before - float(farm["money"])))
        return out

    def do_land(farm, board_size):
        mine = fmap.get(id(farm), -1) == seat
        before = float(farm["money"])
        out = orig_land(farm, board_size)
        if mine:
            atomic.append((state["step"], "BUY_LAND", before - float(farm["money"])))
        return out

    K._apply_unit_action = apply_unit
    K._commit_unit = commit
    K._do_hire = do_hire
    K._do_buy_land = do_land
    try:
        env = ke.make("kaggriculture",
                      configuration={"seed": seed, "episodeSteps": steps}, debug=False)
        shed_capacity = int(env.configuration.get("shedCapacity", 100))
        inner = env.interpreter

        def interpreter(st, env_):
            obs0 = st[0].observation
            farms = obs0["farms"] if "farms" in obs0 else None
            fmap.clear()
            if farms:
                for i, f in enumerate(farms):
                    fmap[id(f)] = i
                state["step"] = int(obs0.get("step", state["step"] + 1))
            return inner(st, env_)

        env.interpreter = interpreter

        actor = P.BCPolicy(job["members"], safety=int(job.get("safety", 3)))
        import features as FMOD
        rec = PolicyRecorder(actor, FMOD)
        opp = _make_opponent(job["opponent"], K)
        agents = [None, None]
        agents[seat] = lambda obs, config=None: actor.act(obs)
        agents[1 - seat] = opp
        env.run(agents)
        last = env.steps[-1]
        if [last[i]["status"] for i in (0, 1)] != ["DONE", "DONE"]:
            raise RuntimeError(f"seed={seed} seat={seat} ended "
                               f"{[last[i]['status'] for i in (0, 1)]}")
        money = [float(last[0]["observation"]["farms"][i]["money"]) for i in (0, 1)]
    finally:
        K._apply_unit_action = orig_unit
        K._commit_unit = orig_commit
        K._do_hire = orig_hire
        K._do_buy_land = orig_land

    out = _summarise(rec.turns, unit_rows, dict(engine_ops), commits, atomic)
    out.update({"seed": seed, "seat": seat, "us": money[seat], "them": money[1 - seat],
                "opponent": str(job["opponent"]), "shed_capacity": shed_capacity})
    return out


def _make_opponent(spec, K):
    if spec == "base":
        members = sorted(str(p) for p in
                         (REPO_ROOT / "agents" / "team_a" / "bc" / "weights").glob("*.npz"))
        sys.path.insert(0, str(REPO_ROOT / "agents" / "team_a" / "bc"))
        import policy as P
        pol = P.BCPolicy(members, safety=3)
        return lambda obs, config=None: pol.act(obs)
    if spec == "starter":
        return K.starter_agent
    if spec == "random":
        return K.random_agent
    # directory agent (arena layout): import its main.py under a private module name
    d = Path(spec)
    if not d.is_absolute():
        d = REPO_ROOT / spec
    sys.path.insert(0, str(d))
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("_opp_main", d / "main.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    fn = getattr(mod, "agent", None) or getattr(mod, "act", None)
    return lambda obs, config=None: fn(obs)


# --------------------------------------------------------------------------- aggregation
UNIT_MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def _summarise(turns, unit_rows, engine_ops, commits, atomic):
    """Everything below is per-episode; the cross-episode roll-up lives in the analyser."""
    # ---- unit-op stream ------------------------------------------------------------
    unit = {p: Counter() for p in PHASES}
    op_transitions = Counter()          # (net_op -> final_op) when they differ
    head_override = {p: Counter() for p in PHASES}
    step_day = {t["step"]: t["day"] for t in turns}

    for t in turns:
        ph = phase_of(t["day"])
        net, eg, ka, fin = (t["ops_net"], t["ops_endgame"], t["ops_keepalive"],
                            t["ops_final"])
        n = len(fin)
        unit[ph]["unit_turns"] += n
        for u in range(n):
            a = net[u] if u < len(net) else None
            b = fin[u] if u < len(fin) else None
            e = eg[u] if u < len(eg) else a
            k = ka[u] if u < len(ka) else e
            if a == b:
                unit[ph]["net_kept"] += 1
            else:
                unit[ph]["overridden"] += 1
                op_transitions[(a[0] if a else "?", b[0] if b else "?")] += 1
                if a != e:
                    unit[ph]["by_endgame"] += 1
                elif e != k:
                    unit[ph]["by_keepalive"] += 1
                else:
                    unit[ph]["by_sanitise"] += 1
            # per-head override: op / crop(PLANT) / item(PICKUP,PLACE) / qty(PICKUP)
            if a and b:
                head_override[ph]["op_n"] += 1
                if a[0] != b[0]:
                    head_override[ph]["op_changed"] += 1
                if a[0] == "PLANT":
                    head_override[ph]["crop_n"] += 1
                    if b[0] != "PLANT" or (len(b) > 1 and a[1] != b[1]):
                        head_override[ph]["crop_changed"] += 1
                if a[0] in ("PICKUP", "PLACE"):
                    head_override[ph]["item_n"] += 1
                    if b[0] != a[0] or (len(b) > 1 and a[1] != b[1]):
                        head_override[ph]["item_changed"] += 1
                if a[0] == "PICKUP":
                    head_override[ph]["uqty_n"] += 1
                    if b[0] != "PICKUP" or (len(b) > 2 and len(a) > 2 and a[2] != b[2]):
                        head_override[ph]["uqty_changed"] += 1

    # engine-seen vs stream B (the interpreter's PLANT block) + executed flags
    exec_by_op = Counter()
    noop_by_op = Counter()
    exec_phase = {p: Counter() for p in PHASES}
    engine_rewrote = Counter()
    final_ops = {t["step"]: t["ops_final"] for t in turns}
    for step, idx, name, executed in unit_rows:
        ph = phase_of(step_day.get(step, step // 24))
        key = "PASS" if name == "PASS" else ("MOVE" if name in UNIT_MOVES else name)
        if executed:
            exec_by_op[key] += 1
            exec_phase[ph]["executed"] += 1
        else:
            noop_by_op[key] += 1
            exec_phase[ph]["noop"] += 1
            if key == "PASS":
                exec_phase[ph]["noop_pass"] += 1
        b = final_ops.get(step, [])
        seen = engine_ops.get(step, {}).get(idx)
        if idx < len(b) and seen is not None and list(b[idx]) != list(seen):
            engine_rewrote[(b[idx][0], seen[0] if seen else "?")] += 1

    # ---- order stream ---------------------------------------------------------------
    # requested per (step, type, item) from stream B; executed from stream C.
    req = defaultdict(int)
    dup_keys = 0
    order_stage = {p: Counter() for p in PHASES}
    origin_of = {}       # (step, key) -> origin tag
    liq = {p: Counter() for p in PHASES}
    net_sell_units = Counter()
    final_sell_units = Counter()
    for t in turns:
        ph = phase_of(t["day"])
        L = t.get("liq")
        if L:
            liq[ph]["turns"] += 1
            for k, v in L.items():
                liq[ph][k] += v
            if L["room"] == 0:
                liq[ph]["room_zero_turns"] += 1
            if L["truncated"] > 0:
                liq[ph]["truncating_turns"] += 1
        for o in t.get("orders_net", []):
            if o and o[0] == "SELL":
                net_sell_units[o[1]] += _order_qty(o)
        for o in t.get("orders_final", []):
            if o and o[0] == "SELL":
                final_sell_units[o[1]] += _order_qty(o)
        net = t.get("orders_net", [])
        eg = t.get("orders_endgame", net)
        ka = t.get("orders_keepalive", eg)
        lq = t.get("orders_liquidate", ka)
        fin = t.get("orders_final", lq)
        net_map = {}
        for o in net:
            k = _order_key(o)
            if k:
                net_map.setdefault(k, []).append(_order_qty(o))
        order_stage[ph]["net_emitted"] += len(net)
        order_stage[ph]["final_emitted"] += len(fin)
        seen_keys = set()
        for o in fin:
            k = _order_key(o)
            if k is None:
                continue
            q = _order_qty(o)
            # HIRE legitimately repeats within a turn (one order per hand); anything else
            # repeating would make the (type,item) join ambiguous, so it is counted.
            if k in seen_keys and k[0] != "HIRE":
                dup_keys += 1
            req[(t["step"], k)] += q
            seen_keys.add(k)
            # origin: identical order in the net stream?
            cand = net_map.get(k)
            if cand and q in cand:
                origin = "net_exact"
            elif cand:
                origin = "net_requantified"
            else:
                origin = "rule"
            origin_of[(t["step"], k)] = origin
            order_stage[ph]["origin_" + origin] += 1
            order_stage[ph]["origin_%s_%s" % (origin, k[0])] += 1
        # which stage introduced / removed
        def keys(lst):
            return Counter(_order_key(o) for o in lst if _order_key(o))
        cn, ce, ck, cl, cf = keys(net), keys(eg), keys(ka), keys(lq), keys(fin)
        order_stage[ph]["endgame_delta"] += sum((ce - cn).values()) + sum((cn - ce).values())
        order_stage[ph]["keepalive_delta"] += sum((ck - ce).values()) + sum((ce - ck).values())
        order_stage[ph]["liquidate_added"] += sum((cl - ck).values())
        order_stage[ph]["liquidate_removed"] += sum((ck - cl).values())
        order_stage[ph]["sanitise_removed"] += sum((cl - cf).values())
        order_stage[ph]["net_sell_emitted"] += sum(v for k, v in cn.items() if k[0] == "SELL")
        order_stage[ph]["final_sell_emitted"] += sum(v for k, v in cf.items() if k[0] == "SELL")

    # executed units + dollars, joined to requested + origin
    ex = defaultdict(int)
    dollars = defaultdict(float)
    fails = Counter()
    money_by_origin = defaultdict(float)
    units_by_origin = defaultdict(int)
    for step, op, item, price, ok, reason in commits:
        k = (op, item if op not in ("HIRE", "BUY_LAND") else "-")
        origin = origin_of.get((step, k), "unmatched")
        if ok:
            ex[(step, k)] += 1
            dollars[k] += price if op == "SELL" else -price
            money_by_origin[(origin, op)] += price if op == "SELL" else -price
            units_by_origin[(origin, op)] += 1
        else:
            fails[(op, reason)] += 1
    for step, kind, spent in atomic:
        k = (kind, "-")
        origin = origin_of.get((step, k), "unmatched")
        if spent > 0:
            ex[(step, k)] += 1
            dollars[k] += -spent
            money_by_origin[(origin, kind)] += -spent
            units_by_origin[(origin, kind)] += 1
        else:
            fails[(kind, "no_cash")] += 1

    # per-order-type executed/requested
    per_type_req = Counter()
    per_type_ex = Counter()
    orders_full = Counter()
    orders_partial = Counter()
    orders_zero = Counter()
    for (step, k), q in req.items():
        got = ex.get((step, k), 0)
        per_type_req[k] += q
        per_type_ex[k] += got
        if got >= q:
            orders_full[k] += 1
        elif got > 0:
            orders_partial[k] += 1
        else:
            orders_zero[k] += 1

    def flat(d):
        return {"|".join(map(str, k)) if isinstance(k, tuple) else str(k): v
                for k, v in d.items()}

    return {
        "turns": len(turns),
        "unit": {p: dict(unit[p]) for p in PHASES},
        "head_override": {p: dict(head_override[p]) for p in PHASES},
        "op_transitions": flat(op_transitions),
        "exec_by_op": dict(exec_by_op),
        "noop_by_op": dict(noop_by_op),
        "exec_phase": {p: dict(exec_phase[p]) for p in PHASES},
        "engine_rewrote": flat(engine_rewrote),
        "order_stage": {p: dict(order_stage[p]) for p in PHASES},
        "liq": {p: dict(liq[p]) for p in PHASES},
        "net_sell_units": dict(net_sell_units),
        "final_sell_units": dict(final_sell_units),
        "order_req": flat(per_type_req),
        "order_ex": flat(per_type_ex),
        "orders_full": flat(orders_full),
        "orders_partial": flat(orders_partial),
        "orders_zero": flat(orders_zero),
        "order_fails": flat(fails),
        "dollars": flat(dollars),
        "money_by_origin": flat(money_by_origin),
        "units_by_origin": flat(units_by_origin),
        "dup_order_keys": dup_keys,
    }


def _clean(job):
    """Same episode with no instrumentation -- the parity control."""
    from arena import engine as arena_engine
    ke = arena_engine.import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as K  # noqa: F401
    sys.path.insert(0, str(REPO_ROOT / "agents" / "team_a" / "bc"))
    import policy as P
    seed, seat = int(job["seed"]), int(job["seat"])
    env = ke.make("kaggriculture",
                  configuration={"seed": seed, "episodeSteps": int(job.get("steps", 720))},
                  debug=False)
    actor = P.BCPolicy(job["members"], safety=int(job.get("safety", 3)))
    opp = _make_opponent(job["opponent"], K)
    agents = [None, None]
    agents[seat] = lambda obs, config=None: actor.act(obs)
    agents[1 - seat] = opp
    env.run(agents)
    last = env.steps[-1]
    money = [float(last[0]["observation"]["farms"][i]["money"]) for i in (0, 1)]
    return {"seed": seed, "seat": seat, "us": money[seat], "them": money[1 - seat]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="4256-4271")
    ap.add_argument("--opponent", default="base")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--members", default="", help="pathsep list; default = champion weights")
    ap.add_argument("--safety", type=int, default=3)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--check-parity", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import os
    if args.members:
        members = [p for p in args.members.split(os.pathsep) if p]
    else:
        members = sorted(str(p) for p in
                         (REPO_ROOT / "agents" / "team_a" / "bc" / "weights").glob("*.npz"))
    lo, hi = (int(x) for x in args.seeds.split("-"))
    jobs = [{"seed": s, "seat": seat, "members": members, "opponent": args.opponent,
             "steps": args.steps, "safety": args.safety}
            for s in range(lo, hi + 1) for seat in (0, 1)]

    def run(fn, js):
        if args.workers <= 1:
            return [fn(j) for j in js]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            return pool.map(fn, js, chunksize=1)

    rows = run(_one, jobs)
    parity = None
    if args.check_parity:
        clean = run(_clean, jobs)
        bad = [(a["seed"], a["seat"], a["us"], b["us"])
               for a, b in zip(rows, clean) if abs(a["us"] - b["us"]) > 1e-6
               or abs(a["them"] - b["them"]) > 1e-6]
        parity = {"checked": len(clean), "mismatches": bad}
        if bad:
            raise RuntimeError(f"instrumentation changed the episode: {bad[:3]}")

    out = {"tag": args.tag, "opponent": args.opponent, "seeds": [lo, hi],
           "episodes": len(rows), "members": members, "safety": args.safety,
           "parity": parity, "rows": rows}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({"tag": args.tag, "episodes": len(rows),
                      "us_mean": round(sum(r["us"] for r in rows) / len(rows), 1),
                      "them_mean": round(sum(r["them"] for r in rows) / len(rows), 1),
                      "parity": parity}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
