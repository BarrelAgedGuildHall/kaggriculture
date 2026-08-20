"""expB-201 arm: M3a/M3b routing dropped onto the FROZEN Mixed probe chassis.

Portfolio held fixed: the market/economy section below (orders: sell, feed,
hire, land, animals, seeds) and the task GENERATION are copied verbatim from
`agents/team_b/probes/probe_core.py::_act` at main@31ff526 (PR #8 merge) with
the Mixed preset. The ONLY changes are:

  (a) siting — build/plant tiles come from layout.SitePlan (contiguous rings
      from each quadrant's shed-access anchor; animals near, melon far) instead
      of the global dist-to-nearest-shed sort;
  (b) unit assignment — routing.Router (persistent per-day cluster binding +
      target locks + batched pickups) instead of per-turn greedy nearest-unit
      dispatch. The courier, endgame and on-tile-service passes are unchanged.

Everything runs inside the M0 chassis (deadline + sanitiser + fallback).

A/B: millrace:routed_mixed vs probe:Mixed — run_ab.py, paired seeds, side swap.
Hypothesis (spec M3b gate): MOVE <= 30% and Wilson LB > 50%.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import probe_core as pc  # frozen chassis: constants + helpers + PRESETS

from .layout import SitePlan
from .main import make_chassis
from .routing import Router

BATCH = {"WHEAT": 8, "FERTILIZER": 4, "GOOSE": 3, "COW": 3, "SHEEP": 3}


def _requirement(act):
    if act[0] == "FEED":
        return "WHEAT"
    if act[0] == "FERTILIZE":
        return "FERTILIZER"
    if act[0] == "PLACE":
        return act[1]
    return None


def _plan(obs, S, deadline):
    P, st = S["P"], S["st"]
    router = S["router"]

    pid = obs["player"]
    me = obs["farms"][pid]
    priv = obs["private"]
    market = obs["market"]
    day, hour = obs["day"], obs["hour"]
    tiles = me["tiles"]
    board = len(tiles)
    money = me["money"]
    shed = dict(priv["shed"])
    seeds = dict(priv["seeds"])
    prices = market["prices"]
    last_day = P["season_days"] - 1
    endgame = day >= last_day

    if st["day"] != day:
        st["day"] = day
        st["sold_today"] = {}

    units = [tuple(me["farmer"])] + [tuple(h) for h in me["hands"]]
    invs = [dict(priv["inventories"][i]) if i < len(priv["inventories"]) else {} for i in range(len(units))]
    carried_animals = {a: sum(iv.get(a, 0) for iv in invs) for a in pc.ANIMALS}

    # ---- census (verbatim from probe_core._act @31ff526) -------------------
    animals, plants, empties, empty_structs = [], [], [], []
    for y in range(board):
        for x in range(board):
            t = tiles[y][x]
            if t is None:
                empties.append((x, y))
            elif isinstance(t, dict):
                if "animal" in t:
                    animals.append(((x, y), t))
                elif t.get("kind") in ("COOP", "PASTURE"):
                    empty_structs.append(((x, y), t["kind"]))
                elif t.get("kind") == "PLANT":
                    plants.append(((x, y), t))
    # M3a: ring siting replaces the global dist-to-nearest-shed sort
    site = SitePlan(empties, board)
    n_animals = len(animals)
    n_geese = sum(1 for _, t in animals if t["animal"] == "GOOSE")
    n_cows = sum(1 for _, t in animals if t["animal"] == "COW")
    n_sheep = sum(1 for _, t in animals if t["animal"] == "SHEEP")
    crop_counts = {}
    for _, t in plants:
        crop_counts[t["crop"]] = crop_counts.get(t["crop"], 0) + 1
    wheat_stock = shed.get("WHEAT", 0) + sum(iv.get("WHEAT", 0) for iv in invs)

    # ---- market orders (verbatim; portfolio held fixed) ---------------------
    orders = []
    herd = n_animals + sum(shed.get(a, 0) + carried_animals[a] for a in pc.ANIMALS)
    reserve_wheat = 0 if endgame else int(2.5 * max(1, herd))
    sellable = {}
    for item, n in shed.items():
        if item in pc.ANIMALS or n <= 0:
            continue
        if item == "WHEAT":
            n = max(0, n - reserve_wheat)
        if n > 0:
            sellable[item] = n
    # (sell_policy == "onsight" for the Mixed preset: no metering branch)
    expected_rev = 0
    for item, n in sorted(sellable.items(), key=lambda kv: -prices.get(kv[0], 0) * kv[1]):
        orders.append(["SELL", item, n])
        st["sold_today"][item] = st["sold_today"].get(item, 0) + n
        expected_rev += prices.get(item, 0) * n

    budget = money - P["cash_reserve"] + int(0.8 * expected_rev)
    capex_ok = day < P["capex_stop_day"] and not endgame
    shed_room = 100 - sum(shed.values())

    def buy_animals():
        nonlocal budget
        if not (capex_ok and shed_room > 2):
            return
        shed_animals = {a: shed.get(a, 0) for a in pc.ANIMALS}
        pending = sum(shed_animals.values())
        wheat_q = prices.get("WHEAT", 25) + 3
        if len(empties) + len(empty_structs) <= pending:
            return
        n_extra = len(me["unlocked_quadrants"]) - 1
        n_owned = 25 * len(me["unlocked_quadrants"])
        if day > 0 and P["buy_land"] and n_extra < 3 and (n_owned - len(empties)) >= n_owned - 6:
            return
        feed_cover = max(0, 2 * (herd + 1) - wheat_stock) * wheat_q
        if day > 0 and budget < 300 + feed_cover:
            return
        want = []
        if day <= 11 and n_cows + shed_animals["COW"] < P["cows"]:
            want.append(("COW", P["cows"] - n_cows - shed_animals["COW"]))
        if day <= 13 and n_sheep + shed_animals["SHEEP"] < P["sheep"]:
            want.append(("SHEEP", P["sheep"] - n_sheep - shed_animals["SHEEP"]))
        goose_cap = P["geese_day0"] if day == 0 else P["goose_target"]
        if day <= 23 and n_geese + shed_animals["GOOSE"] < goose_cap:
            want.append(("GOOSE", goose_cap - n_geese - shed_animals["GOOSE"]))
        pend_cap = 12 if day == 0 else 6
        for animal, k in want:
            if pending >= pend_cap:
                break
            cost = pc.ANIMALS[animal]["cost"]
            k = min(k, max(0, budget) // cost, pend_cap - pending, shed_room)
            if day > 0:
                k = min(k, max(0, (budget - feed_cover)) // cost)
            if k > 0 and len(orders) < 10:
                orders.append(["BUY_ANIMAL", animal, int(k)])
                budget -= k * cost
                pending += k

    def buy_feed(keepalive_only=False):
        nonlocal budget
        if endgame or herd == 0:
            return
        target = herd if keepalive_only else int(P["wheat_reserve"] * herd)
        short = target - wheat_stock
        wheat_q = prices.get("WHEAT", 25)
        must = wheat_stock < 1.2 * herd
        if short > 0 and (must or wheat_q <= P["wheat_buy_cap"]) and len(orders) < 10:
            k = min(short, max(0, budget) // max(1, wheat_q + 3))
            if k > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", int(k)])
                budget -= k * (wheat_q + 3)

    def hire(count):
        nonlocal budget
        n_prev = me["hires_today"]
        done = 0
        while done < count and len(orders) < 10:
            c = pc._fib(n_prev + done)
            if budget < c:
                break
            budget -= c
            orders.append(["HIRE"])
            done += 1

    def buy_land():
        nonlocal budget
        if not (capex_ok and P["buy_land"] and len(orders) < 10):
            return
        n_extra = len(me["unlocked_quadrants"]) - 1
        n_owned = 25 * len(me["unlocked_quadrants"])
        used = n_owned - len(empties)
        want_more = (n_geese < P["goose_target"] or crop_counts.get("WHEAT", 0) < P["wheat_tiles"]
                     or (P["cows"] + P["sheep"]) > n_cows + n_sheep)
        if n_extra < 3 and want_more and used >= n_owned - 6 and day >= 2:
            cost = pc.LAND_PRICES[n_extra]
            need_feed = max(0, herd - wheat_stock) * (prices.get("WHEAT", 25) + 3)
            if budget >= cost + need_feed:
                orders.append(["BUY_LAND"])
                budget -= cost

    def buy_seeds():
        nonlocal budget
        if not capex_ok:
            return
        want_wheat_tiles = min(P["wheat_tiles"], 3 * len(units),
                               max(0, 25 * len(me["unlocked_quadrants"]) - n_animals - 2))
        need = {
            "WHEAT": 0 if seeds.get("WHEAT", 0) > 8 else max(0, want_wheat_tiles - crop_counts.get("WHEAT", 0) - seeds.get("WHEAT", 0)),
            "CARROT": max(0, pc._want_carrots(P, prices, day) - crop_counts.get("CARROT", 0) - seeds.get("CARROT", 0)),
            "MELON": max(0, (P["melon_tiles"] if day <= 18 else 0) - crop_counts.get("MELON", 0) - seeds.get("MELON", 0)),
        }
        for crop, k in need.items():
            k = min(k, 10, max(0, budget) // pc.CROPS[crop]["seed"])
            if k > 0 and len(orders) < 10:
                orders.append(["BUY_SEED", crop, int(k)])
                budget -= k * pc.CROPS[crop]["seed"]

    want_hands = P["hands_target"] + (4 if (endgame and P["smart_endgame"]) else 0)
    hires_needed = max(0, want_hands - me["hires_today"])
    if day == 0 and hour == 0:
        buy_animals()
        if P["wheat_buy_open"] > 0:
            wq = prices.get("WHEAT", 25)
            k = min(P["wheat_buy_open"], max(0, budget) // (wq + 3))
            if k > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", int(k)])
                budget -= k * (wq + 3)
        hire(min(4, hires_needed))
        buy_seeds()
    else:
        buy_feed(keepalive_only=True)
        if hour <= 1 and hires_needed:
            hire(hires_needed)
        buy_land()
        buy_animals()
        buy_feed()
        buy_seeds()
    orders = orders[:10]

    # ---- unit tasks (generation verbatim; SITING via SitePlan) --------------
    tasks = []

    if endgame and P["smart_endgame"]:
        for (x, y), t in animals:
            if t.get("yield_units", 0) > 0:
                tasks.append((0, (x, y), ["HARVEST"], ("H", x, y)))
            if t.get("fertilizer_available"):
                tasks.append((1, (x, y), ["COLLECT_FERTILIZER"], ("CF", x, y)))
        for (x, y), t in plants:
            age = day - t["planted_day"]
            if t.get("yield_units", 0) > 0 and age >= pc.CROPS.get(t["crop"], {"first": 2})["first"]:
                tasks.append((0, (x, y), ["HARVEST"], ("H", x, y)))
    else:
        for (x, y), t in animals:
            if not t["fed_today"] and wheat_stock > 0:
                tasks.append((1, (x, y), ["FEED"], ("F", x, y)))
            if t.get("fertilizer_available"):
                tasks.append((3, (x, y), ["COLLECT_FERTILIZER"], ("CF", x, y)))
            if P["care"] and not t["cared_today"]:
                tasks.append((6, (x, y), ["CARE"], ("C", x, y)))
            cap = 4 if t["animal"] == "GOOSE" else 6
            if t.get("yield_units", 0) >= cap - 2 or (t.get("yield_units", 0) > 0 and day >= last_day - 1):
                tasks.append((4, (x, y), ["HARVEST"], ("H", x, y)))
        for (x, y), t in plants:
            crop = t["crop"]
            cd = pc.CROPS.get(crop)
            if cd is None:
                continue
            age = day - t["planted_day"]
            if not t["watered_today"] and age < cd["maxday"]:
                prio = 2.5 if t["consecutive_unwatered"] >= 1 else 5
                tasks.append((prio, (x, y), ["WATER"], ("W", x, y)))
            ready_age = cd["first"] if crop == "MELON" else cd["maxday"]
            ready = age >= ready_age or (day >= last_day - 1 and age >= cd["first"])
            if ready and t.get("yield_units", 0) > 0:
                tasks.append((4, (x, y), ["HARVEST"], ("H", x, y)))
            if (P["fertilize_wheat"] and crop == "WHEAT" and age <= 2
                    and t.get("fertilized_until_day", -1) < day):
                tasks.append((7, (x, y), ["FERTILIZE"], ("FZ", x, y)))
        avail_geese = shed.get("GOOSE", 0) + carried_animals["GOOSE"]
        avail_pasture = (shed.get("COW", 0) + carried_animals["COW"]
                         + shed.get("SHEEP", 0) + carried_animals["SHEEP"])
        coop_needs = max(0, avail_geese - sum(1 for _, k in empty_structs if k == "COOP"))
        past_needs = max(0, avail_pasture - sum(1 for _, k in empty_structs if k == "PASTURE"))
        for _ in range(coop_needs):
            p = site.take_animal_site()
            if p is None:
                break
            tasks.append((2, p, ["BUILD_COOP"], ("B", p[0], p[1])))
        for _ in range(past_needs):
            p = site.take_animal_site()
            if p is None:
                break
            tasks.append((2, p, ["BUILD_PASTURE"], ("B", p[0], p[1])))
        for (x, y), kind in empty_structs:
            if kind == "COOP":
                animal = "GOOSE"
            else:
                animal = "COW" if (shed.get("COW", 0) + carried_animals["COW"]) > 0 else "SHEEP"
            if shed.get(animal, 0) + carried_animals[animal] > 0:
                tasks.append((0, (x, y), ["PLACE", animal], ("P", x, y)))
        if capex_ok:
            plan = []
            plan += ["MELON"] * max(0, P["melon_tiles"] - crop_counts.get("MELON", 0))
            plan += ["WHEAT"] * max(0, min(P["wheat_tiles"], 3 * len(units)) - crop_counts.get("WHEAT", 0))
            plan += ["CARROT"] * max(0, pc._want_carrots(P, prices, day) - crop_counts.get("CARROT", 0))
            avail_seeds = dict(seeds)
            for crop in plan:
                if avail_seeds.get(crop, 0) <= 0:
                    continue
                p = site.take_melon_site() if crop == "MELON" else site.take_crop_site()
                if p is None:
                    break
                avail_seeds[crop] -= 1
                tasks.append((8, p, ["PLANT", crop], ("PL", p[0], p[1])))

    tasks.sort(key=lambda t: t[0])

    # ---- assignment: courier/endgame/on-tile passes (verbatim), then Router --
    actions = [["PASS"] for _ in units]
    claimed = set()
    busy = [False] * len(units)

    if not endgame:
        for i, pos in enumerate(units):
            carry = sum(n for it, n in invs[i].items() if it not in ("WHEAT",) and it not in pc.ANIMALS)
            if carry >= 5:
                if pos in pc.SHED_TILES:
                    actions[i] = ["DROP"]
                else:
                    actions[i] = [pc._step_toward(pos, pc._nearest_shed(pos), board)]
                busy[i] = True

    if endgame and P["smart_endgame"]:
        for i, pos in enumerate(units):
            carry = sum(invs[i].values())
            if carry > 0 and hour >= 18:
                if pos in pc.SHED_TILES:
                    actions[i] = ["DROP"]
                else:
                    actions[i] = [pc._step_toward(pos, pc._nearest_shed(pos), board)]
                busy[i] = True

    if not endgame:
        for i, pos in enumerate(units):
            if busy[i]:
                continue
            x, y = pos
            t = tiles[y][x]
            if not isinstance(t, dict):
                continue
            act = None
            if "animal" in t:
                cap = 4 if t["animal"] == "GOOSE" else 6
                if not t["fed_today"] and invs[i].get("WHEAT", 0) > 0:
                    act = ["FEED"]
                elif t.get("fertilizer_available"):
                    act = ["COLLECT_FERTILIZER"]
                elif t.get("yield_units", 0) >= cap - 2:
                    act = ["HARVEST"]
                elif P["care"] and not t["cared_today"]:
                    act = ["CARE"]
            elif t.get("kind") == "PLANT":
                crop = t["crop"]
                cd = pc.CROPS.get(crop)
                age = day - t["planted_day"]
                r_age = cd["first"] if crop == "MELON" else cd["maxday"] if cd else 99
                if cd and not t["watered_today"] and age < cd["maxday"]:
                    act = ["WATER"]
                elif cd and age >= r_age and t.get("yield_units", 0) > 0:
                    act = ["HARVEST"]
            if act:
                actions[i] = act
                busy[i] = True
                claimed.add((act[0][0] if act[0] != "COLLECT_FERTILIZER" else "CF", x, y))

    # M3b: persistent cluster assignment + target locks + batched pickups
    shed_left = dict(shed)
    ctx = {
        "invs": invs,
        "shed": shed_left,
        "quads": list(me["unlocked_quadrants"]),
        "step_toward": lambda pos, tgt: pc._step_toward(pos, tgt, board),
        "shed_tiles": pc.SHED_TILES,
        "nearest_shed": pc._nearest_shed,
        "requirement": _requirement,
        "batch": BATCH,
        "claimed": claimed,
    }
    router.plan(day, units, busy, actions, tasks, ctx)

    # idle units: drop full inventories or drift home (verbatim)
    for i, pos in enumerate(units):
        if busy[i]:
            continue
        carry = sum(n for it, n in invs[i].items() if it != "WHEAT" and it not in pc.ANIMALS)
        if carry >= 3 or (carry > 0 and hour >= 21):
            if pos in pc.SHED_TILES:
                actions[i] = ["DROP"]
            else:
                actions[i] = [pc._step_toward(pos, pc._nearest_shed(pos), board)]

    return {"farmer": actions[0], "hands": actions[1:], "market": orders}


def make_routed_agent(params=None):
    """Routed Mixed chassis, wrapped in the M0 chassis. `params` overlays the
    frozen Mixed preset (for probes only — expB-201 runs it untouched)."""
    P = dict(pc.DEFAULTS)
    P.update(pc.PRESETS["Mixed"])
    if params:
        P.update(params)

    def state_factory():
        return {"P": P,
                "st": {"sold_today": {}, "day": -1, "planted_melon_days": {}},
                "router": Router()}

    return make_chassis(_plan, state_factory)
