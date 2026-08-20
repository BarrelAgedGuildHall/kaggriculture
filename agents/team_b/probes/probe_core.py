"""Team B probe agent core (sim-checklist expB-101..112).

A single parameterised scripted agent used for one-variable A/B probes of the
game-analysis claims (workshop/deliverables/2026-08-19-team-b-game-analysis/analysis.md).
NOT a championship agent: it is deliberately simple, safe (global try/except ->
PASS) and structurally sound so that a one-parameter A/B between two instances
measures the parameter and nothing else.

Usage:
    from probe_core import make_agent, PRESETS
    agent = make_agent({"geese_day0": 9, "goose_target": 60, ...})

Named presets live in PRESETS; the `TeamB-Probe-*.py` files beside this module
freeze one preset each (Pit naming convention: TeamB-Probe-<what>).

Engine: kaggle-environments 1.32.7 (see docs/engine/VERSION.md).
"""
from __future__ import annotations

# ---------------------------------------------------------------- constants
CROPS = {
    "WHEAT":      {"seed": 10, "first": 2, "maxday": 4},
    "CARROT":     {"seed": 20, "first": 2, "maxday": 3},
    "MELON":      {"seed": 80, "first": 10, "maxday": 12},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "product": "WOOL"},
}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
LAND_PRICES = [1000, 2000, 4000]
# products the sell-metering policy holds back (linear/sq glut curves + town demand)
METERED = {"MILK": 1, "WOOL": 1, "STRAWBERRY": 1}
# shop -> products (for the exact town drain rate)
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

DEFAULTS = {
    "season_days": 30,        # (episodeSteps // 24); probes for expB-112 set 5
    # ---- opening (day 0)
    "geese_day0": 8,          # geese bought on turn 0
    "carrots_day0": 0,        # carrot tiles wanted in the opening (gradual ramp)
    "cash_reserve": 0,        # never spend below this
    "wheat_buy_open": 6,      # feed wheat bought turn 0
    # ---- growth
    "goose_target": 55,       # stop buying geese here
    "cows": 0,                # pasture cows (bought from day 1, lead time 8d)
    "sheep": 0,               # pasture sheep (lead time 6d)
    "melon_tiles": 0,         # melon tiles to keep planted (10d program)
    "wheat_tiles": 35,        # wheat tiles to keep planted (autarky engine)
    "fertilize_wheat": True,  # spend collected fertilizer on wheat tiles
    "buy_land": True,         # buy the next quadrant when empty tiles run out
    "hands_target": 10,       # hire depth per day
    "care": True,             # CARE fed animals daily
    "wheat_buy_cap": 55,      # stop buying feed above this quote (autarky band)
    "capex_stop_day": 24,     # no new assets from this day
    # ---- selling
    "sell_policy": "onsight", # "onsight" | "meter" (class-aware, expB-106)
    "wheat_reserve": 2.0,     # keep wheat >= reserve * herd
    # ---- endgame
    "smart_endgame": True,    # last day: stop feed/water, harvest+drop+sell, hire deep
}

PRESETS = {
    "GooseRush":    {"geese_day0": 9, "cash_reserve": 0, "wheat_buy_open": 8, "fertilize_wheat": False},
    "GradualRamp":  {"geese_day0": 5, "carrots_day0": 8, "cash_reserve": 500, "fertilize_wheat": False},
    "Autarky6040":  {"goose_target": 55, "wheat_tiles": 35},
    "NaiveEndgame": {"smart_endgame": False},
    "Mixed":        {"cows": 6, "sheep": 5, "melon_tiles": 6},
    "MeterSell":    {"cows": 6, "sheep": 5, "melon_tiles": 6, "sell_policy": "meter"},
}


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(pos, target, board):
    x, y = pos
    tx, ty = target
    if abs(tx - x) >= abs(ty - y) and tx != x:
        return "EAST" if tx > x else "WEST"
    if ty != y:
        return "SOUTH" if ty > y else "NORTH"
    if tx != x:
        return "EAST" if tx > x else "WEST"
    return "PASS"


def _nearest_shed(pos):
    return min(SHED_TILES, key=lambda t: _dist(pos, t))


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _drain_per_day(town, product):
    """Exact town drain per day at the current shop unlock (1.32.7)."""
    per_day = 1  # town centre, all non-fertilizer products
    if product == "FERTILIZER":
        return 0
    for shop in town.get("unlocked_shops", []):
        prods = SHOPS.get(shop, [])
        if product in prods:
            per_day += 12 if len(prods) == 1 else 6
    return per_day


def make_agent(params=None):
    P = dict(DEFAULTS)
    if params:
        P.update(params)
    st = {"sold_today": {}, "day": -1, "planted_melon_days": {}}

    def agent(obs):
        try:
            return _act(obs, P, st)
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent


# ------------------------------------------------------------------ policy
def _act(obs, P, st):
    pid = obs["player"]
    me = obs["farms"][pid]
    priv = obs["private"]
    market = obs["market"]
    town = obs["town"]
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

    # ---- census -----------------------------------------------------------
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
    empties.sort(key=lambda p: _dist(p, _nearest_shed(p)))
    n_animals = len(animals)
    n_geese = sum(1 for _, t in animals if t["animal"] == "GOOSE")
    n_cows = sum(1 for _, t in animals if t["animal"] == "COW")
    n_sheep = sum(1 for _, t in animals if t["animal"] == "SHEEP")
    crop_counts = {}
    for _, t in plants:
        crop_counts[t["crop"]] = crop_counts.get(t["crop"], 0) + 1
    wheat_stock = shed.get("WHEAT", 0) + sum(iv.get("WHEAT", 0) for iv in invs)

    # ---- market orders ----------------------------------------------------
    orders = []

    # SELLs first (a same-turn sale funds a same-turn purchase).
    herd = n_animals + sum(shed.get(a, 0) for a in ANIMALS)
    # sell-side reserve is deliberately ABOVE the buy-side target: we buy up to
    # wheat_reserve*herd and only sell past 2.5*herd, so feed never round-trips.
    reserve_wheat = 0 if endgame else int(2.5 * max(1, herd))
    sellable = {}
    for item, n in shed.items():
        if item in ANIMALS or n <= 0:
            continue
        if item == "WHEAT":
            n = max(0, n - reserve_wheat)
        if n > 0:
            sellable[item] = n
    if P["sell_policy"] == "meter" and not endgame and day < last_day - 1:
        # class-aware: meter linear/sq premium goods at the town drain rate,
        # hold melon for a dump (flat sq near zero), everything else on sight.
        metered_orders = {}
        for item in list(sellable):
            if item in METERED:
                allowed = _drain_per_day(town, item) - st["sold_today"].get(item, 0)
                metered_orders[item] = max(0, min(sellable[item], allowed))
                sellable.pop(item)
            elif item == "MELON" and shed.get("MELON", 0) < 30:
                sellable.pop(item)  # accumulate, dump later
        for item, n in metered_orders.items():
            if n > 0:
                sellable[item] = n
    expected_rev = 0
    for item, n in sorted(sellable.items(), key=lambda kv: -prices.get(kv[0], 0) * kv[1]):
        orders.append(["SELL", item, n])
        st["sold_today"][item] = st["sold_today"].get(item, 0) + n
        expected_rev += prices.get(item, 0) * n

    # SELLs execute before later orders in the same list, so most of the
    # proceeds are spendable this turn (0.8: prices walk down as we sell).
    budget = money - P["cash_reserve"] + int(0.8 * expected_rev)
    capex_ok = day < P["capex_stop_day"] and not endgame
    shed_room = 100 - sum(shed.values())

    def buy_animals():
        nonlocal budget
        if not (capex_ok and shed_room > 2):
            return
        shed_animals = {a: shed.get(a, 0) for a in ANIMALS}
        pending = sum(shed_animals.values())
        # don't buy an animal we cannot house or cannot feed for 2 days
        wheat_q = prices.get("WHEAT", 25) + 3
        if len(empties) + len(empty_structs) <= pending:
            return
        # save for land instead of the next goose when tiles are binding
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
        pend_cap = 12 if day == 0 else 6  # day-0 burst, steady drip after
        for animal, k in want:
            if pending >= pend_cap:
                break  # don't flood the shed / placement pipeline
            cost = ANIMALS[animal]["cost"]
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
        must = wheat_stock < 1.2 * herd  # keep-alive with a buffer
        if short > 0 and (must or wheat_q <= P["wheat_buy_cap"]) and len(orders) < 10:
            k = min(short, max(0, budget) // max(1, wheat_q + 3))
            if k > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", int(k)])
                budget -= k * (wheat_q + 3)

    def hire(count):
        nonlocal budget
        cost, n_prev = 0, me["hires_today"]
        done = 0
        while done < count and len(orders) < 10:
            c = _fib(n_prev + done)
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
            cost = LAND_PRICES[n_extra]
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
            "CARROT": max(0, (P["carrots_day0"] if day <= 3 else 0) - crop_counts.get("CARROT", 0) - seeds.get("CARROT", 0)),
            "MELON": max(0, (P["melon_tiles"] if day <= 18 else 0) - crop_counts.get("MELON", 0) - seeds.get("MELON", 0)),
        }
        for crop, k in need.items():
            k = min(k, 10, max(0, budget) // CROPS[crop]["seed"])
            if k > 0 and len(orders) < 10:
                orders.append(["BUY_SEED", crop, int(k)])
                budget -= k * CROPS[crop]["seed"]

    want_hands = P["hands_target"] + (4 if (endgame and P["smart_endgame"]) else 0)
    hires_needed = max(0, want_hands - me["hires_today"])
    if day == 0 and hour == 0:
        # opening burst: assets first, feed second, hands from the remainder
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
        buy_feed(keepalive_only=True)   # never starve the herd
        if hour <= 1 and hires_needed:
            hire(hires_needed)
        buy_land()
        buy_animals()
        buy_feed()                      # top up to the reserve
        buy_seeds()
    orders = orders[:10]

    # ---- unit tasks --------------------------------------------------------
    # Build prioritised task list; greedy nearest-unit assignment.
    tasks = []  # (prio, pos, action, key)

    if endgame and P["smart_endgame"]:
        for (x, y), t in animals:
            if t.get("yield_units", 0) > 0:
                tasks.append((0, (x, y), ["HARVEST"], ("H", x, y)))
            if t.get("fertilizer_available"):
                tasks.append((1, (x, y), ["COLLECT_FERTILIZER"], ("CF", x, y)))
        for (x, y), t in plants:
            age = day - t["planted_day"]
            if t.get("yield_units", 0) > 0 and age >= CROPS.get(t["crop"], {"first": 2})["first"]:
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
            cd = CROPS.get(crop)
            if cd is None:
                continue
            age = day - t["planted_day"]
            if not t["watered_today"] and age < cd["maxday"]:
                prio = 2.5 if t["consecutive_unwatered"] >= 1 else 5
                tasks.append((prio, (x, y), ["WATER"], ("W", x, y)))
            ready = age >= cd["maxday"] or (day >= last_day - 1 and age >= cd["first"])
            if ready and t.get("yield_units", 0) > 0:
                tasks.append((4, (x, y), ["HARVEST"], ("H", x, y)))
            if (P["fertilize_wheat"] and crop == "WHEAT" and age <= 2
                    and t.get("fertilized_until_day", -1) < day):
                tasks.append((7, (x, y), ["FERTILIZE"], ("FZ", x, y)))
        # planting plan on empty tiles: coops/pastures for shed animals first,
        # then wheat / carrot / melon up to their targets.
        shed_geese = shed.get("GOOSE", 0)
        shed_pasture = shed.get("COW", 0) + shed.get("SHEEP", 0)
        free = list(empties)
        coop_needs = max(0, shed_geese - sum(1 for _, k in empty_structs if k == "COOP"))
        past_needs = max(0, shed_pasture - sum(1 for _, k in empty_structs if k == "PASTURE"))
        for _ in range(min(coop_needs, len(free))):
            x, y = free.pop(0)
            tasks.append((2, (x, y), ["BUILD_COOP"], ("B", x, y)))
        for _ in range(min(past_needs, len(free))):
            x, y = free.pop(0)
            tasks.append((2, (x, y), ["BUILD_PASTURE"], ("B", x, y)))
        for (x, y), kind in empty_structs:
            animal = "GOOSE" if kind == "COOP" else ("COW" if shed.get("COW", 0) > 0 else "SHEEP")
            if shed.get(animal, 0) > 0:
                tasks.append((0, (x, y), ["PLACE", animal], ("P", x, y)))
        if capex_ok:
            plan = []
            plan += ["MELON"] * max(0, P["melon_tiles"] - crop_counts.get("MELON", 0))
            plan += ["WHEAT"] * max(0, min(P["wheat_tiles"], 3 * len(units)) - crop_counts.get("WHEAT", 0))
            plan += ["CARROT"] * max(0, (P["carrots_day0"] if day <= 3 else 0) - crop_counts.get("CARROT", 0))
            avail_seeds = dict(seeds)
            for crop in plan:
                if not free:
                    break
                if avail_seeds.get(crop, 0) <= 0:
                    continue
                avail_seeds[crop] -= 1
                x, y = free.pop(0)
                tasks.append((8, (x, y), ["PLANT", crop], ("PL", x, y)))

    tasks.sort(key=lambda t: t[0])

    # ---- assignment --------------------------------------------------------
    actions = [["PASS"] for _ in units]
    claimed = set()
    shed_wheat_left = shed.get("WHEAT", 0)
    shed_fert_left = shed.get("FERTILIZER", 0)
    shed_animal_left = {a: shed.get(a, 0) for a in ANIMALS}
    busy = [False] * len(units)

    def requirement(act):
        if act[0] == "FEED":
            return "WHEAT"
        if act[0] == "FERTILIZE":
            return "FERTILIZER"
        if act[0] == "PLACE":
            return act[1]
        return None

    # courier rule: a unit carrying several sellable items runs them home NOW —
    # goods in the shed sell this turn; goods in a pocket sell tomorrow.
    if not endgame:
        for i, pos in enumerate(units):
            carry = sum(n for it, n in invs[i].items() if it not in ("WHEAT",) and it not in ANIMALS)
            if carry >= 5:
                if pos in SHED_TILES:
                    actions[i] = ["DROP"]
                else:
                    actions[i] = [_step_toward(pos, _nearest_shed(pos), board)]
                busy[i] = True

    # endgame: units carrying goods head to the shed and DROP (auto-drop never
    # runs on the last day), selling handled by the market orders above.
    if endgame and P["smart_endgame"]:
        for i, pos in enumerate(units):
            carry = sum(invs[i].values())
            if carry > 0 and hour >= 18:
                if pos in SHED_TILES:
                    actions[i] = ["DROP"]
                else:
                    actions[i] = [_step_toward(pos, _nearest_shed(pos), board)]
                busy[i] = True

    # on-tile service: a unit standing on a tile with pending work does it NOW
    # (turns each animal visit into feed->collect->harvest->care over consecutive
    # turns instead of being re-dispatched across the board every turn).
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
                cd = CROPS.get(crop)
                age = day - t["planted_day"]
                if cd and not t["watered_today"] and age < cd["maxday"]:
                    act = ["WATER"]
                elif cd and age >= cd["maxday"] and t.get("yield_units", 0) > 0:
                    act = ["HARVEST"]
            if act:
                actions[i] = act
                busy[i] = True
                claimed.add((act[0][0] if act[0] != "COLLECT_FERTILIZER" else "CF", x, y))

    for prio, tpos, act, key in tasks:
        if key in claimed:
            continue
        req = requirement(act)
        best, best_d, best_mode = None, 10 ** 9, None
        for i, pos in enumerate(units):
            if busy[i]:
                continue
            if req and invs[i].get(req, 0) > 0:
                d = _dist(pos, tpos)
                mode = "direct"
            elif req:
                left = {"WHEAT": shed_wheat_left, "FERTILIZER": shed_fert_left}.get(req, shed_animal_left.get(req, 0))
                if left <= 0:
                    continue
                sp = _nearest_shed(pos)
                d = _dist(pos, sp) + _dist(sp, tpos) + 1
                mode = "via_shed"
            else:
                d = _dist(pos, tpos)
                mode = "direct"
            if d < best_d:
                best, best_d, best_mode = i, d, mode
        if best is None:
            continue
        i = best
        pos = units[i]
        if best_mode == "direct":
            if pos == tpos:
                actions[i] = act
            else:
                actions[i] = [_step_toward(pos, tpos, board)]
        else:
            sp = _nearest_shed(pos)
            if pos in SHED_TILES:
                # batch pickup: grab enough for several tasks of this kind
                if req == "WHEAT":
                    k = min(shed_wheat_left, 8)
                    shed_wheat_left -= k
                elif req == "FERTILIZER":
                    k = min(shed_fert_left, 4)
                    shed_fert_left -= k
                else:
                    k = min(shed_animal_left[req], 3)
                    shed_animal_left[req] -= k
                if k <= 0:
                    continue
                actions[i] = ["PICKUP", req, int(k)]
            else:
                actions[i] = [_step_toward(pos, sp, board)]
        busy[i] = True
        claimed.add(key)

    # idle units: drop full inventories (throughput hygiene) or drift home.
    for i, pos in enumerate(units):
        if busy[i]:
            continue
        carry = sum(n for it, n in invs[i].items() if it != "WHEAT" and it not in ANIMALS)
        if carry >= 3 or (carry > 0 and hour >= 21):
            if pos in SHED_TILES:
                actions[i] = ["DROP"]
            else:
                actions[i] = [_step_toward(pos, _nearest_shed(pos), board)]

    return {"farmer": actions[0], "hands": actions[1:], "market": orders}
