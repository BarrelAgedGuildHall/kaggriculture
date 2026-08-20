"""Feed-chain arm: M4 service scheduler + M6 wheat sizing/siting + M5a lambda,
dropped onto the expB-201 routed Mixed chassis.

Lineage: `_plan` started as a verbatim copy of `routed_probe._plan` (the
frozen expB-201 champion, which itself carries the probe economy verbatim
from main@31ff526). The champion arm is NOT touched — it stays the A/B
control. Changes here, in spec order (section 5.3 step 2, internal order M4
first per expB-211's c=62.5% verdict):

M4 (service.py):
  - task generation for animals/plants moves to `service.animal_tasks` /
    `service.plant_tasks`: keep-alive FEED at prio 0.25, keep-alive WATER at
    0.75, harvest-or-lose at 1.5, fertilizer-collection escalation from hour
    16, mature-unwatered harvesting;
  - `service.FeedRouter`: cross-cluster borrowing for urgent tasks;
  - adjacent-feed passes: a wheat carrier standing on / next to an unfed
    animal feeds it instead of walking past (the expB-211 c bucket). Critical
    animals (consecutive_unfed >= 1) pre-empt even couriers; ordinary unfed
    only claim idle units.

M5a: lambda = fib(hands + 1) / productive_actions_per_unit_day, measured from
  our own emitted non-move actions (no-op rate is 0.0%, so emitted ~= executed)
  over a 3-day rolling window; consumed by the M6 fertilize rule.

M6 (flag "m6_wheat", default on):
  - wheat tile target sized from feed demand: ceil(herd / 0.8 yield-per-tile-
    day) instead of the fixed 35 (0.8 = 4 units per 5-day unfertilized cycle,
    structural, self-checkable against realised yield);
  - wheat sited INSIDE animal clusters (layout.take_wheat_site);
  - wheat shadow price (M1): shadow = marginal_revenue(EGG, 2) while
    feed-short (stock < 2 x herd) else quote(WHEAT); wheat is bought while
    quote < shadow (replaces the hardcoded $55 cap — the G4 assay showed
    absolute-dollar thresholds misread a price-level shift);
  - fertilize rule (flag "m6_fert"): "shadow" = fertilize wheat iff
    quote(FERTILIZER) + lambda < 2 x wheat_shadow (expB-206 arm);
    "sell" = never fertilize; "probe" = legacy young-wheat-always.

A/B specs (run_ab / feed_audit): millrace:feedchain, millrace:feedchain{json}.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import probe_core as pc  # frozen chassis: constants + helpers + PRESETS

from .layout import SitePlan
from .main import make_chassis
from .market import MarketModel
from .service import FeedRouter, animal_tasks, plant_tasks

BATCH = {"WHEAT": 8, "FERTILIZER": 4, "GOOSE": 3, "COW": 3, "SHEEP": 3}
COURIER_CARRY = 5
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
WHEAT_YIELD_PER_TILE_DAY = 0.8  # 4 units / 5-day unfertilized cycle (structural)


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
    model: MarketModel = S["model"]
    model.observe(obs, st.get("last_orders"))

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
        # ---- M5a: close the day's productive-action ledger ------------------
        if st["unit_turns_today"] > 0:
            per_unit_day = st["prod_today"] / (st["unit_turns_today"] / 24.0)
            st["prod_days"].append(per_unit_day)
            st["prod_days"] = st["prod_days"][-3:]
        st["prod_today"] = 0
        st["unit_turns_today"] = 0
        st["day"] = day
        st["sold_today"] = {}

    units = [tuple(me["farmer"])] + [tuple(h) for h in me["hands"]]
    invs = [dict(priv["inventories"][i]) if i < len(priv["inventories"]) else {} for i in range(len(units))]
    carried_animals = {a: sum(iv.get(a, 0) for iv in invs) for a in pc.ANIMALS}
    st["unit_turns_today"] += len(units)

    # ---- census (probe lineage) ---------------------------------------------
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
    site = SitePlan(empties, board)
    n_animals = len(animals)
    n_geese = sum(1 for _, t in animals if t["animal"] == "GOOSE")
    n_cows = sum(1 for _, t in animals if t["animal"] == "COW")
    n_sheep = sum(1 for _, t in animals if t["animal"] == "SHEEP")
    crop_counts = {}
    for _, t in plants:
        crop_counts[t["crop"]] = crop_counts.get(t["crop"], 0) + 1
    wheat_stock = shed.get("WHEAT", 0) + sum(iv.get("WHEAT", 0) for iv in invs)

    herd = n_animals + sum(shed.get(a, 0) + carried_animals[a] for a in pc.ANIMALS)

    # ---- M5a lambda + M6 wheat shadow ---------------------------------------
    prod_rate = (sum(st["prod_days"]) / len(st["prod_days"])) if st["prod_days"] else 10.0
    lam = pc._fib(len(me["hands"]) + 1) / max(1.0, prod_rate)
    feed_short = wheat_stock < 2 * herd
    if P["m6_band"] and herd > 0 and feed_short:
        wheat_shadow = model.marginal_revenue("EGG", 2)
    else:
        wheat_shadow = prices.get("WHEAT", 25)
    st["lambda"] = lam
    st["wheat_shadow"] = wheat_shadow

    # ---- M4 herd capacity clamp ----------------------------------------------
    # An animal-day costs ~4 productive actions (feed + collect + care +
    # amortised harvest); the probe grew the herd toward goose_target 55 while
    # 11 units can service ~35 — the expB-213 audits put the unfed mass in the
    # big-herd late season. Never buy an animal the measured action budget
    # cannot keep alive. (Generalises the spec's P1 placement clamp; M5b will
    # raise capacity by hiring — until then the herd must fit the crew.)
    crew = 1 + max(len(me["hands"]), P["hands_target"])
    crop_load = 1.3 * (crop_counts.get("WHEAT", 0) + crop_counts.get("CARROT", 0)
                       + crop_counts.get("MELON", 0))
    herd_cap = max(12, int((prod_rate * crew - crop_load) / P["actions_per_animal"]))

    # ---- M6 wheat tile target ------------------------------------------------
    if P["m6_size"]:
        demand_tiles = 0 if herd == 0 else math.ceil(herd / WHEAT_YIELD_PER_TILE_DAY)
        want_wheat = min(demand_tiles, 3 * len(units),
                         max(0, 25 * len(me["unlocked_quadrants"]) - n_animals - 2))
    else:
        want_wheat = min(P["wheat_tiles"], 3 * len(units),
                         max(0, 25 * len(me["unlocked_quadrants"]) - n_animals - 2))

    # ---- market orders (probe lineage; M6 touches wheat buying only) --------
    orders = []
    reserve_wheat = 0 if endgame else int(2.5 * max(1, herd))
    sellable = {}
    for item, n in shed.items():
        if item in pc.ANIMALS or n <= 0:
            continue
        if item == "WHEAT":
            n = max(0, n - reserve_wheat)
        if n > 0:
            sellable[item] = n
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
        cap_left = herd_cap - herd
        for animal, k in want:
            if pending >= pend_cap or cap_left <= 0:
                break
            cost = pc.ANIMALS[animal]["cost"]
            k = min(k, max(0, budget) // cost, pend_cap - pending, shed_room, cap_left)
            if day > 0:
                k = min(k, max(0, (budget - feed_cover)) // cost)
            if k > 0 and len(orders) < 10:
                orders.append(["BUY_ANIMAL", animal, int(k)])
                budget -= k * cost
                pending += k
                cap_left -= k

    n_unfed = sum(1 for _, t in animals if not t["fed_today"])

    def buy_feed(keepalive_only=False):
        nonlocal budget
        if endgame or herd == 0:
            return
        target = herd if keepalive_only else int(P["wheat_reserve"] * herd)
        short = target - wheat_stock
        wheat_q = prices.get("WHEAT", 25)
        must = wheat_stock < 1.2 * herd
        # shed float: pocket wheat is locked to its carrier's route, so an
        # empty shed starves every OTHER unit's feed round even when total
        # stock looks fine (expB-213 trace: shed at 0 from hour 2, one carrier
        # feeding serially, animals unfed at day end)
        if keepalive_only and shed.get("WHEAT", 0) < min(4, n_unfed):
            must = True
            short = max(short, min(n_unfed, 8) - shed.get("WHEAT", 0))
        if P["m6_band"]:
            in_band = wheat_q < wheat_shadow  # M6 no-trade band, not a $ cap
        else:
            in_band = wheat_q <= P["wheat_buy_cap"]
        if short > 0 and (must or in_band) and len(orders) < 10:
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
        want_more = (n_geese < P["goose_target"] or crop_counts.get("WHEAT", 0) < want_wheat
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
        need = {
            "WHEAT": 0 if seeds.get("WHEAT", 0) > 8 else max(0, want_wheat - crop_counts.get("WHEAT", 0) - seeds.get("WHEAT", 0)),
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
        # feed kitty: reserve one day of feed for ~8 animals BEFORE the animal
        # burst spends the whole $3000 (expB-213 v6: the residual escapes were
        # day-0/1 bucket (a) — broke, no production, nothing to sell yet)
        feed_kitty = 8 * (prices.get("WHEAT", 25) + 3)
        budget -= feed_kitty
        buy_animals()
        budget += feed_kitty
        # day-0 feed sized to the day-0 herd: the fixed wheat_buy_open=6 left
        # the opening 7-12 animals short (expB-213 audit_v1: early unfed days
        # were bucket (a) — no wheat existed in the system)
        bought_today = sum(o[2] for o in orders if o[0] == "BUY_ANIMAL")
        want_open_wheat = max(P["wheat_buy_open"], bought_today + 2)
        if want_open_wheat > 0:
            wq = prices.get("WHEAT", 25)
            k = min(want_open_wheat, max(0, budget) // (wq + 3))
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

    # ---- unit tasks: M4 service scheduler ------------------------------------
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
        tasks += animal_tasks(animals, wheat_stock, P, hour, day >= last_day - 1)

        if P["m6_fert"] == "shadow":
            def fert_rule(crop, t):
                if crop != "WHEAT":
                    return False
                age = day - t["planted_day"]
                return (age <= 2 and shed.get("FERTILIZER", 0) > 0
                        and prices.get("FERTILIZER", 100) + lam < 2 * wheat_shadow)
        elif P["m6_fert"] == "sell":
            def fert_rule(crop, t):
                return False
        else:  # "probe" legacy
            fert_rule = None
        tasks += plant_tasks(plants, day, last_day, P, pc.CROPS, fert_rule)

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
        # no placements at hour 23: an animal placed on the day's final turn
        # cannot be fed the same day and starts life with consecutive_unfed 1
        if hour < 23:
            for (x, y), kind in empty_structs:
                if kind == "COOP":
                    animal = "GOOSE"
                else:
                    animal = "COW" if (shed.get("COW", 0) + carried_animals["COW"]) > 0 else "SHEEP"
                if shed.get(animal, 0) + carried_animals[animal] > 0:
                    tasks.append((0, (x, y), ["PLACE", animal], ("P", x, y)))
        if capex_ok and hour < 20:
            # no planting after hour 19: the planting day counts as unwatered,
            # so a plant that cannot be watered tonight weeds out (3 of 9
            # weed-kills in expB-213 v9 were age-0 late plantings)
            plan = []
            plan += ["MELON"] * max(0, P["melon_tiles"] - crop_counts.get("MELON", 0))
            plan += ["WHEAT"] * max(0, want_wheat - crop_counts.get("WHEAT", 0))
            plan += ["CARROT"] * max(0, pc._want_carrots(P, prices, day) - crop_counts.get("CARROT", 0))
            avail_seeds = dict(seeds)
            for crop in plan:
                if avail_seeds.get(crop, 0) <= 0:
                    continue
                if crop == "MELON":
                    p = site.take_melon_site_capped(board)
                elif crop == "WHEAT" and P["m6_site"]:
                    p = site.take_wheat_site()  # M6: inside the animal cluster
                else:
                    # NOT capped: capping crops too just pushed melons and
                    # pastures into the corners instead (measured: escapes
                    # 0->9, weed-kills 10->20 on the same 64 seats)
                    p = site.take_crop_site()
                if p is None:
                    break
                avail_seeds[crop] -= 1
                tasks.append((8, p, ["PLANT", crop], ("PL", p[0], p[1])))

    tasks.sort(key=lambda t: t[0])

    # ---- assignment ----------------------------------------------------------
    actions = [["PASS"] for _ in units]
    claimed = set()
    busy = [False] * len(units)
    shed_pool = dict(shed)  # shared this-turn shed accounting for all passes

    def feed_adjacent(i, pos, critical_only):
        """M4: wheat carrier on/next to an unfed animal feeds it now."""
        if invs[i].get("WHEAT", 0) <= 0:
            return False
        for (x, y), t in animals:
            if t["fed_today"]:
                continue
            if critical_only and t.get("consecutive_unfed", 0) < 1:
                continue
            key = ("F", x, y)
            if key in claimed:
                continue
            d = abs(pos[0] - x) + abs(pos[1] - y)
            if d == 0:
                actions[i] = ["FEED"]
            elif d == 1:
                actions[i] = [pc._step_toward(pos, (x, y), board)]
            else:
                continue
            claimed.add(key)
            return True
        return False

    if not endgame:
        # critical feeds pre-empt even couriers
        for i, pos in enumerate(units):
            if not busy[i] and feed_adjacent(i, pos, critical_only=True):
                busy[i] = True

        # CRITICAL-FEED GUARANTEE: an animal at consecutive_unfed >= 1 escapes
        # TONIGHT if missed. Router discipline (home cluster, locks) does not
        # apply — the closest wheat-supplied unit is assigned directly.
        # (expB-213 v7: the last two escapes were fringe geese whose critical
        # feeds lost to cluster-first picking two days running.)
        # NOTE: draws down shed_pool, which the router ctx shares — separate
        # copies double-booked the same shed wheat (6 PICKUP no-ops, v8).
        for (ax, ay), t in animals:
            if t["fed_today"] or t.get("consecutive_unfed", 0) < 1:
                continue
            key = ("F", ax, ay)
            if key in claimed:
                continue
            best_i, best_d, best_via = None, None, False
            for i, pos in enumerate(units):
                if busy[i]:
                    continue
                if invs[i].get("WHEAT", 0) > 0:
                    d, via = abs(pos[0] - ax) + abs(pos[1] - ay), False
                elif shed_pool.get("WHEAT", 0) > 0:
                    sp = pc._nearest_shed(pos)
                    d = (abs(pos[0] - sp[0]) + abs(pos[1] - sp[1])
                         + abs(sp[0] - ax) + abs(sp[1] - ay) + 1)
                    via = True
                else:
                    continue
                if best_d is None or d < best_d:
                    best_i, best_d, best_via = i, d, via
            if best_i is None:
                continue
            pos = units[best_i]
            if best_via:
                if pos in pc.SHED_TILES:
                    k = min(shed_pool["WHEAT"], BATCH["WHEAT"])
                    shed_pool["WHEAT"] -= k
                    actions[best_i] = ["PICKUP", "WHEAT", int(k)]
                else:
                    actions[best_i] = [pc._step_toward(pos, pc._nearest_shed(pos), board)]
            elif pos == (ax, ay):
                actions[best_i] = ["FEED"]
            else:
                actions[best_i] = [pc._step_toward(pos, (ax, ay), board)]
            busy[best_i] = True
            claimed.add(key)

    if not endgame:
        # (a "liquidity crisis" courier that delivered at carry >= 2 when cash
        # ran low was tried against the one poverty-escape incident and made
        # every variant WORSE: globally fed 0.988 -> 0.960; day>=22 and
        # day>=25 variants traded broad feeding health for nothing. Reverted.)
        for i, pos in enumerate(units):
            if busy[i]:
                continue
            carry = sum(n for it, n in invs[i].items() if it not in ("WHEAT",) and it not in pc.ANIMALS)
            if carry >= COURIER_CARRY:
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
        # quadrants that still have hungry animals: a wheat carrier there does
        # not stop to clear its tile — feeding first (expB-213 trace: on-tile
        # batching kept 9 loaded carriers busy while unfed counts fell by only
        # ~1-2/hour; the last animals in line slipped past midnight)
        unfed_quads = set()
        for (ax, ay), t in animals:
            if not t["fed_today"]:
                unfed_quads.add(("N" if ay < board // 2 else "S")
                                + ("W" if ax < board // 2 else "E"))
        for i, pos in enumerate(units):
            if busy[i]:
                continue
            x, y = pos
            t = tiles[y][x]
            if not isinstance(t, dict):
                continue
            uq = ("N" if y < board // 2 else "S") + ("W" if x < board // 2 else "E")
            feed_first = uq in unfed_quads and invs[i].get("WHEAT", 0) > 0
            act = None
            if "animal" in t:
                cap = 4 if t["animal"] == "GOOSE" else 6
                if not t["fed_today"] and invs[i].get("WHEAT", 0) > 0:
                    act = ["FEED"]
                elif t.get("fertilizer_available"):
                    # collected even in feed-first mode: one action, and a
                    # skipped collection is lost permanently at the refresh
                    act = ["COLLECT_FERTILIZER"]
                elif feed_first:
                    pass  # this carrier's wheat is needed elsewhere first
                elif t.get("yield_units", 0) >= cap - 2:
                    act = ["HARVEST"]
                elif P["care"] and not t["cared_today"] and t["fed_today"]:
                    act = ["CARE"]  # care on an unfed animal is worth zero
            elif t.get("kind") == "PLANT":
                crop = t["crop"]
                cd = pc.CROPS.get(crop)
                age = day - t["planted_day"]
                r_age = cd["first"] if crop == "MELON" else cd["maxday"] if cd else 99
                dying = t["consecutive_unwatered"] >= 1 and not t["watered_today"]
                if cd and not t["watered_today"] and age < cd["maxday"] and (dying or not feed_first):
                    act = ["WATER"]  # keep-alive watering trumps feed-first
                elif cd and age >= r_age and t.get("yield_units", 0) > 0 and (dying or not feed_first):
                    act = ["HARVEST"]
            if act:
                actions[i] = act
                busy[i] = True
                claimed.add((act[0][0] if act[0] != "COLLECT_FERTILIZER" else "CF", x, y))

        # ordinary unfed animals claim idle wheat carriers standing next door
        for i, pos in enumerate(units):
            if not busy[i] and feed_adjacent(i, pos, critical_only=False):
                busy[i] = True

    ctx = {
        "invs": invs,
        "shed": shed_pool,
        "quads": list(me["unlocked_quadrants"]),
        "step_toward": lambda pos, tgt: pc._step_toward(pos, tgt, board),
        "shed_tiles": pc.SHED_TILES,
        "nearest_shed": pc._nearest_shed,
        "requirement": _requirement,
        "batch": BATCH,
        "claimed": claimed,
    }
    router.plan(day, units, busy, actions, tasks, ctx)

    for i, pos in enumerate(units):
        if busy[i]:
            continue
        carry = sum(n for it, n in invs[i].items() if it != "WHEAT" and it not in pc.ANIMALS)
        if carry >= 3 or (carry > 0 and hour >= 21):
            if pos in pc.SHED_TILES:
                actions[i] = ["DROP"]
            else:
                actions[i] = [pc._step_toward(pos, pc._nearest_shed(pos), board)]

    # ---- M5a productive-action ledger ---------------------------------------
    for a in actions:
        if isinstance(a, list) and a and a[0] not in MOVES and a[0] != "PASS":
            st["prod_today"] += 1
    st["last_orders"] = orders

    return {"farmer": actions[0], "hands": actions[1:], "market": orders}


FEEDCHAIN_DEFAULTS = {
    # m6_wheat bundled three separable changes; expB-204 decomposed them
    # (32 pairs each vs the full bundle, seeds 0-31):
    #   sizing 53.1% (wash) -> kept: adapts tiles to the herd at no cost;
    #   siting OFF wins 71.9% -> cluster-sited wheat displaces premium tiles;
    #   band OFF wins 95.3% -> shadow=MR(EGG,2)~$80-100 made "buy wheat at
    #     any price" the policy and torched the ramp's capital. The band
    #     returns when M2 prices the counterfactual use of cash.
    "m6_size": True,   # wheat tiles = ceil(herd / yield-per-tile-day)
    "m6_site": False,  # wheat back on the fringe (expB-204: -22pp inside)
    "m6_band": False,  # legacy wheat_buy_cap until M2 prices capital
    # expB-206 (32 pairs, corrected build): shadow rule 45.3% [33.7-57.4] vs
    # always-sell — the pre-registered <2pp falsification branch fired, the
    # rule is deleted per spec ("always sell fertilizer").
    "m6_fert": "sell",
    "actions_per_animal": 4.0,  # feed + collect + care + amortised harvest
}


def make_feedchain_agent(params=None):
    """Feed-chain arm (M4 + M6 + M5a) on the routed Mixed chassis."""
    P = dict(pc.DEFAULTS)
    P.update(pc.PRESETS["Mixed"])
    P.update(FEEDCHAIN_DEFAULTS)
    if params:
        params = dict(params)
        if "m6_wheat" in params:  # legacy bundle flag
            v = params.pop("m6_wheat")
            params.setdefault("m6_size", v)
            params.setdefault("m6_site", v)
            params.setdefault("m6_band", v)
        P.update(params)

    def state_factory():
        return {"P": P,
                "st": {"sold_today": {}, "day": -1, "planted_melon_days": {},
                       "last_orders": [], "prod_today": 0, "unit_turns_today": 0,
                       "prod_days": []},
                "router": FeedRouter(),
                "model": MarketModel()}

    return make_chassis(_plan, state_factory)
