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
  our own emitted non-move actions over a 3-day rolling window; consumed by the
  M6 fertilize rule. (The original note here claimed a 0.0% no-op rate, so that
  emitted ~= executed. expB-301 measured 559 no-op unit-turns per season —
  22.6% of would-be productive actions — so `prod_rate` OVERSTATES executed
  work by about that much and every lambda derived from it is correspondingly
  low. Recorded, not fixed: it is a second variable.)

expB-302 (flag "m6_fert" = "opp"): fertilizer make-or-buy at a measured
  internal transfer price, plus lambda_opp — the opportunity cost of a
  unit-turn from the agent's own closed books, as opposed to M5a's fib WAGE.
  Default off; the arm was VOID (see the experiment log).

expB-303 (flags "strawberry_tiles" / "strawberry_rule"): STRAWBERRY acreage.
  The first `ongoing` crop the agent grows — HARVEST does not clear the tile,
  production fires on a cadence past max_yield_day, yield_units is a CAP so the
  tile must be emptied between events, and two unwatered days weed it out at
  ANY age. `service.plant_tasks` handles that lifecycle; `tests/
  test_team_b_strawberry.py` pins every one of those facts against the engine.

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

try:  # package context (repo tools/tests: agents/team_b on sys.path)
    from . import probe_core as pc  # vendored frozen chassis constants/helpers
    from .chassis import make_chassis
    from .layout import SitePlan
    from .market import MarketModel
    from .pricer import Pricer, PricedRouter
    from .service import FeedRouter, animal_tasks, plant_tasks
except ImportError:  # Kaggle/arena loader: main.py exec'd flat, dir on sys.path
    import probe_core as pc
    from chassis import make_chassis
    from layout import SitePlan
    from market import MarketModel
    from pricer import Pricer, PricedRouter
    from service import FeedRouter, animal_tasks, plant_tasks

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
    priced_dispatch = P["dispatch"] == "priced"

    if st["day"] != day:
        # ---- expB-304a: close the day's lambda telemetry --------------------
        # Both series, averaged over the turns they were actually USED (a
        # sample taken at hour 0 reads lambda_wage ~ 0.1 because the hands have
        # not been re-hired yet). ruling-02 section 2.7 makes this mandatory.
        if st["lam_n"] > 0:
            st["lam_log"].append({
                "day": st["day"],
                "lam_wage": st["lam_sum"] / st["lam_n"],
                "lam_opp": st["lam_opp_sum"] / st["lam_n"],
                "lam_cut": st["lam_cut_sum"] / st["lam_n"],
                "pinned": st["lam_pinned"] >= st["lam_n"],
            })
        st["lam_sum"] = st["lam_opp_sum"] = st["lam_cut_sum"] = 0.0
        st["lam_n"] = st["lam_pinned"] = 0
        # ---- M5a: close the day's productive-action ledger ------------------
        if st["unit_turns_today"] > 0:
            per_unit_day = st["prod_today"] / (st["unit_turns_today"] / 24.0)
            st["prod_days"].append(per_unit_day)
            st["prod_days"] = st["prod_days"][-3:]
            # ---- expB-302: close the day's lambda_opp ledger -----------------
            # contribution = the day's marginal-revenue-priced sales minus the
            # day's purchased inputs (feed + seed; animals and land are capex
            # and belong to the hurdle rate, not to the margin of an action).
            if st["prod_today"] > 0:
                st["contrib_days"].append(
                    (st["gross_today"] - st["cogs_today"]) / st["prod_today"])
                st["contrib_days"] = st["contrib_days"][-3:]
            # expB-304a: season-to-date TOTALS, not a mean of daily ratios --
            # the daily ratio is undefined-ish on a low-volume day and one big
            # feed purchase can make it negative, which is what makes the
            # 3-day estimator swing 0.08 -> 55.9.
            st["contrib_num"] += st["gross_today"] - st["cogs_today"]
            st["contrib_den"] += st["prod_today"]
            st["idle_num"] += st["idle_today"]
            st["idle_den"] += st["unit_turns_today"]
            st["idle_days"].append(st["idle_today"] / st["unit_turns_today"])
            st["idle_days"] = st["idle_days"][-3:]
        st["prod_today"] = 0
        st["unit_turns_today"] = 0
        st["cogs_today"] = 0.0
        st["idle_today"] = 0
        st["day"] = day
        st["sold_today"] = {}
        # ---- M5b ledgers: close yesterday's gross + clamp/hands diagnostics -
        if st["day_open"] >= 0:
            st["gross_days"].append(st["gross_today"])
            st["gross_days"] = st["gross_days"][-2:]
            st["m5b_log"].append({"day": st["day_open"], "want": st.get("want_hands", -1),
                                  "clamp_bound": st["clamp_bound_today"]})
        st["gross_today"] = 0.0
        st["clamp_bound_today"] = False
        st["day_open"] = day

    units = [tuple(me["farmer"])] + [tuple(h) for h in me["hands"]]
    invs = [dict(priv["inventories"][i]) if i < len(priv["inventories"]) else {} for i in range(len(units))]
    carried_animals = {a: sum(iv.get(a, 0) for iv in invs) for a in pc.ANIMALS}
    st["unit_turns_today"] += len(units)

    # ---- census (probe lineage) ---------------------------------------------
    animals, plants, empties, empty_structs, weeds = [], [], [], [], []
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
                elif t.get("kind") == "WEED":
                    # expB-304a / ruling-02 R1: a weeded tile is RECOVERABLE —
                    # `DIG` sets any non-animal tile to None and `PLANT` requires
                    # `tile is None`. The champion never issued DIG and never
                    # even censused these tiles, so they sat fallow for the rest
                    # of the season.
                    weeds.append((x, y))
    site = SitePlan(empties, board)
    n_animals = len(animals)
    n_geese = sum(1 for _, t in animals if t["animal"] == "GOOSE")
    n_cows = sum(1 for _, t in animals if t["animal"] == "COW")
    n_sheep = sum(1 for _, t in animals if t["animal"] == "SHEEP")
    crop_counts = {}
    for _, t in plants:
        crop_counts[t["crop"]] = crop_counts.get(t["crop"], 0) + 1
    # (realised melon yield ledger moved to harvest issuance — see the hook at
    # the end of _plan. Logging at first eligibility here recorded phantom
    # yield: PR#14 audit traced 18/30 melon weed-kills at age >= first, i.e.
    # after their yield had been booked "realised" for tiles that banked $0.)
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

    # ---- expB-302: lambda_opp, the OPPORTUNITY cost of one unit-turn --------
    # `lam` above is a WAGE: what the marginal hand costs. It is the right
    # basis for the hiring decision and the wrong one for an allocation gate,
    # because unit-turns — not dollars — are the binding resource (expB-301:
    # $51.8 of revenue per productive action against a $8.7 wage, so every
    # `> lam` gate in this agent is ~6x too permissive).
    #
    # lambda_opp = contribution margin per productive action of the marginal
    # executed task. Measuring that per task needs the M2 row table (expB-304);
    # until then it is approximated from the agent's own closed books over a
    # 3-day window: the AVERAGE contribution per productive action, scaled by
    # how hard the action budget is actually binding. Idle units mean the
    # marginal task is worth ~nothing, so lambda_opp collapses toward the wage
    # — which is the correct behaviour on days 0-3 and after the last
    # profitable purchase day. Both terms are runtime-measured ratios; no
    # absolute-dollar or absolute-day constant enters (spec section 0).
    if st["contrib_days"] and st["idle_days"]:
        avg_contrib = sum(st["contrib_days"]) / len(st["contrib_days"])
        idle_frac = sum(st["idle_days"]) / len(st["idle_days"])
        lam_opp = max(lam, (1.0 - min(1.0, idle_frac)) * avg_contrib)
    else:
        lam_opp = lam
    st["lambda_opp"] = lam_opp

    # ---- expB-304a: the dispatch cut ----------------------------------------
    # `lam_opp` above is the expB-302 estimator: a 3-day mean of contribution
    # per productive action, floored at the WAGE. Measured on the champion it
    # swings 0.08 -> 55.9 within a season and sits ON the wage floor on 4 of 30
    # days, because one big feed purchase can make a single day's contribution
    # negative. That is tolerable for a once-a-day make-or-buy question and
    # useless as a per-action cut, which is the first thing ruling-02 section
    # 2.7 warned about.
    #
    # Same quantity, measured with less variance and no new economics: the
    # MEDIAN of every completed day's contribution per productive action
    # (parameter-free window: the season so far), scaled by how hard the action
    # budget is actually binding, and floored at ZERO rather than at the wage —
    # a wage floor makes the cut assert a value it has not measured. Zero on
    # day 0 means "dispatch everything", which is correct before any day has
    # closed. The idle term is the stabiliser: cutting work raises idle, which
    # lowers the cut, which re-admits work.
    if priced_dispatch and st["contrib_den"] > 0:
        _c = st["contrib_num"] / st["contrib_den"]
        _i = (st["idle_num"] / st["idle_den"]) if st["idle_den"] > 0 else 0.0
        lam_cut = max(0.0, _c * (1.0 - min(1.0, _i)))
    else:
        lam_cut = 0.0
    st["lambda_cut"] = lam_cut
    st["lam_sum"] += lam
    st["lam_opp_sum"] += lam_opp
    st["lam_cut_sum"] += lam_cut
    st["lam_n"] += 1
    if lam_opp <= lam + 1e-9:
        st["lam_pinned"] += 1

    # ---- expB-302: fertilizer make-or-buy, decided once per turn ------------
    # Engine truth (`_apply_unit_action` WATER + FERTILIZE, 1.32.7): wheat's
    # watering bonus fires at ages 2..4 (window_start = (max_yield_day+1)//2 =
    # 2); yield starts at 1 and each watered day adds 1, or 2 while
    # `fertilized_until_day >= day`. One FERTILIZE covers day, day+1, day+2, so
    # applying it at age 1 or 2 covers the whole bonus window and takes the
    # tile from 4 units to 6 (1+2+2+2 = 7, capped at max_yield 6) — exactly +2
    # units for one action and one fertilizer unit. Age 0 covers only ages 0-2
    # and is worth +1, so it is excluded; the legacy "shadow"/"probe" rules
    # fired at `age <= 2` and bought that half-price application.
    #
    # Make-or-buy: use the unit internally iff the wheat it saves us buying is
    # worth more than selling the unit plus the unit-turn it costs —
    #   2 x marginal_buy_cost(WHEAT, residual shortfall)
    #        >  marginal_revenue(FERTILIZER, 1) + lambda_opp
    # Every term is measured by M1 at live prices; no constant enters.
    # NOT a re-run of expB-206: that rule compared the fertilizer *quote* plus
    # the *wage* against `2 x wheat_shadow`, where wheat_shadow was
    # `marginal_revenue(EGG, 2)` — the egg shadow price since refuted by
    # expB-218 — and it never priced the wheat we would actually have to BUY.
    fert_targets = 0
    fert_internal = False
    if P["m6_fert"] == "opp":
        for _, t in plants:
            if (t["crop"] == "WHEAT" and 1 <= day - t["planted_day"] <= 2
                    and t.get("fertilized_until_day", -1) < day):
                fert_targets += 1
        # Residual shortfall is a HORIZON quantity, not a same-day one: the herd
        # eats 1 wheat/animal/day for every day left, and we grow far less than
        # that (expB-301: 42 units grown against 502 bought), so we are a
        # structural net buyer and the two units this fertilizer creates
        # displace the NEXT two units we would buy — priced at today's quote,
        # not at the far end of a season-sized walk. Gating on today's stock
        # instead silences the rule for 24 of 30 days, because the sell layer
        # already holds a 2.5 x herd wheat reserve (measured on seed 0 before
        # this correction: 14 firings, arm byte-identical to the champion).
        residual = herd * max(0, last_day - day) - wheat_stock
        if fert_targets and residual > 0:
            gain = model.marginal_buy_cost("WHEAT", 2)
            cost = model.marginal_revenue("FERTILIZER", 1) + lam_opp
            fert_internal = gain > cost
            st["fert_calls"] = st.get("fert_calls", 0) + 1
            st["fert_gain"] = gain
            st["fert_cost"] = cost
            if fert_internal:
                st["fert_fires"] = st.get("fert_fires", 0) + 1

    # ---- expB-303: fertilizer onto STRAWBERRY, same transfer-price form -----
    # Engine (`_daily_refresh_plants`): an ongoing crop's production event adds
    # 2 units instead of 1 when the tile was watered that day and fertilized,
    # and one application covers day..day+2 = two events at interval 2. So a
    # fertilizer unit spent on a strawberry eve buys +2 STRAWBERRY units. Use
    # it internally iff those 2 units beat selling the fertilizer plus the
    # unit-turn — the identical make-or-buy test as the wheat branch, priced
    # against a different alternative use. Independent of `m6_fert`, which is
    # the wheat question.
    straw_eves = 0
    straw_fert = False
    _sc = pc.CROPS["STRAWBERRY"]
    _straw_last_eve = _sc["first"] - 1 + _sc["interval"] * (_sc["max_yield"] - 1)
    for _, t in plants:
        if t["crop"] != "STRAWBERRY":
            continue
        a = day - t["planted_day"]
        if (_sc["first"] - 1 <= a <= _straw_last_eve
                and (a - (_sc["first"] - 1)) % _sc["interval"] == 0
                and t.get("fertilized_until_day", -1) < day):
            straw_eves += 1
    if straw_eves:
        straw_fert = (model.marginal_revenue("STRAWBERRY", 2)
                      > model.marginal_revenue("FERTILIZER", 1) + lam_opp)

    # ---- M4 herd capacity clamp ----------------------------------------------
    # An animal-day costs ~4 productive actions (feed + collect + care +
    # amortised harvest); the probe grew the herd toward goose_target 55 while
    # 11 units can service ~35 — the expB-213 audits put the unfed mass in the
    # big-herd late season. Never buy an animal the measured action budget
    # cannot keep alive. (Generalises the spec's P1 placement clamp; M5b will
    # raise capacity by hiring — until then the herd must fit the crew.)
    crew = 1 + max(len(me["hands"]), P["hands_target"])
    crop_load = 1.3 * (crop_counts.get("WHEAT", 0) + crop_counts.get("CARROT", 0)
                       + crop_counts.get("MELON", 0)
                       + crop_counts.get("STRAWBERRY", 0))
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
        if item == "FERTILIZER" and straw_fert and not endgame:
            n = max(0, n - straw_eves)
        if item == "FERTILIZER" and fert_internal and not endgame:
            # expB-302: hold back exactly the units the transfer price says
            # belong on wheat tiles. Without this the sell-on-sight layer
            # empties the shed every turn and the FERTILIZE task can never be
            # supplied — the rule would be measured as a no-op.
            n = max(0, n - fert_targets)
        if n > 0:
            sellable[item] = n
    expected_rev = 0
    for item, n in sorted(sellable.items(), key=lambda kv: -prices.get(kv[0], 0) * kv[1]):
        orders.append(["SELL", item, n])
        st["sold_today"][item] = st["sold_today"].get(item, 0) + n
        expected_rev += prices.get(item, 0) * n
        st["gross_today"] += model.marginal_revenue(item, n)  # M5b income ledger

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
        if want and cap_left <= 0:
            st["clamp_bound_today"] = True  # M5b confound check: clamp binds
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
                st["cogs_today"] += k * wheat_q  # expB-302 lambda_opp ledger

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

    # ---- expB-220: melon acreage from marginal revenue per action ------------
    # Ship the RULE, not the sweep's number (proposal section 5 Option 1 step 2;
    # spec section 0: no absolute-dollar/absolute-day constant gates a decision).
    # Keep planting/replanting melon while the projected marginal revenue per
    # productive action of the NEXT tile-cycle exceeds lambda (the measured
    # $-cost of a productive action, M5a). Everything in the comparison is a
    # runtime measurement: MR walks M1's observed (inv, price) table from live
    # inventory; yield/cycle is the episode's own realised harvest average
    # (structural max_yield prior before the first harvest); actions/cycle is
    # structural (1 PLANT + first_yield_day WATERs + 1 HARVEST).
    if P["melon_rule"] == "mr":
        mel = pc.CROPS["MELON"]
        harvests = st["melon_harvest_units"]
        y_est = (sum(harvests) / len(harvests)) if harvests else float(mel.get("max_yield", 6) - 1)
        y_est = max(1.0, y_est)
        acts_per_cycle = 2.0 + mel["first"]  # PLANT + daily WATER to first yield + HARVEST
        n_cur = crop_counts.get("MELON", 0)

        # Stock-pool race semantics (proposal 1.5): the walk uses OUR
        # increments over the LIVE inventory only. A symmetric-opponent
        # discount was tried first and measured 21.9% vs the fixed-20 arm
        # (expB-220): pricing the opponent's projected supply into the margin
        # makes the rule retreat from a contested stock pool — the
        # ration-not-race error. The live-inventory walk still stops the rule
        # at a genuinely dead pool (MR collapses once the glut is real).
        def mr_tile(k):
            return (model.marginal_revenue("MELON", (k + 1) * y_est)
                    - model.marginal_revenue("MELON", k * y_est))

        want_melon = n_cur
        # grow: value the next tile's cycle on top of the current tiles' cycles
        while want_melon < n_cur + 10:
            if mr_tile(want_melon) / acts_per_cycle <= lam:
                break
            want_melon += 1
        # shrink: stop replanting tiles whose marginal cycle no longer clears lambda
        while want_melon > 0:
            if mr_tile(want_melon - 1) / acts_per_cycle > lam:
                break
            want_melon -= 1
    else:
        want_melon = P["melon_tiles"]
    st["want_melon"] = want_melon

    # ---- expB-303: strawberry acreage ---------------------------------------
    # STRAWBERRY is the only under-supplied high-value FLOW pool on the board
    # (expB-301: Champion[A] realises $231.8/unit on 74 units; we hold zero).
    # "fixed" is the acreage sweep's arm; "mr" ships the rule — keep planting
    # while the projected marginal revenue per productive action of the NEXT
    # tile-cycle clears lambda_opp, exactly the expB-220 melon machinery with
    # the ongoing-crop action count.
    n_straw = crop_counts.get("STRAWBERRY", 0)
    if P["strawberry_rule"] == "mr":
        # actions per tile-cycle, structural: 1 PLANT + one WATER per day it is
        # alive + one HARVEST per production event + one FERTILIZE per two
        # events (each application covers day..day+2 = two events).
        s_acts = (1.0 + (_straw_last_eve + 1) + _sc["max_yield"]
                  + _sc["max_yield"] // _sc["interval"])
        seen = st["straw_harvest_units"]
        y_est = (sum(seen) / len(seen)) if seen else float(_sc["max_yield"] * 2)
        y_est = max(1.0, y_est)

        def s_tile(k):
            return (model.marginal_revenue("STRAWBERRY", (k + 1) * y_est)
                    - model.marginal_revenue("STRAWBERRY", k * y_est))

        want_straw = n_straw
        # Growth bound copied from expB-220's melon rule. Widening it to the
        # structural site budget (`len(empties)`) was tried and REFUTED:
        # 57.8% [49.2-66.0] vs the champion against the capped rule's 67.2%,
        # because the unbounded ask front-loads planting and starves the
        # service queue. Kept at the melon bound.
        while want_straw < n_straw + 10:
            if s_tile(want_straw) / s_acts <= lam_opp:
                break
            want_straw += 1
        while want_straw > 0:
            if s_tile(want_straw - 1) / s_acts > lam_opp:
                break
            want_straw -= 1
    else:
        want_straw = P["strawberry_tiles"]
    # structural replant window: a strawberry planted on day d first yields at
    # d + first_yield_day, so it is plantable while that still lands in season.
    straw_window = day + _sc["first"] <= last_day
    st["want_straw"] = want_straw

    # melon_replant (expB-219): the replant window is structural — a melon
    # planted on day d first yields at d + first_yield_day, so seeds are
    # bought (and tiles planted) while day + first <= last_day, i.e. while a
    # new plant can still be harvested this season (the endgame sweep
    # harvests at age == first on the last day). No absolute-day constant:
    # the bound moves with season_days and the CROPS table.
    melon_window = (day + pc.CROPS["MELON"]["first"] <= last_day) if P["melon_replant"] \
        else (day <= 18)  # legacy incumbent gate

    def buy_seeds():
        nonlocal budget
        if not capex_ok:
            return
        need = {
            "WHEAT": 0 if seeds.get("WHEAT", 0) > 8 else max(0, want_wheat - crop_counts.get("WHEAT", 0) - seeds.get("WHEAT", 0)),
            "CARROT": max(0, pc._want_carrots(P, prices, day) - crop_counts.get("CARROT", 0) - seeds.get("CARROT", 0)),
            "MELON": max(0, (want_melon if melon_window else 0) - crop_counts.get("MELON", 0) - seeds.get("MELON", 0)),
            "STRAWBERRY": max(0, (want_straw if straw_window else 0) - n_straw - seeds.get("STRAWBERRY", 0)),
        }
        for crop, k in need.items():
            k = min(k, 10, max(0, budget) // pc.CROPS[crop]["seed"])
            if k > 0 and len(orders) < 10:
                orders.append(["BUY_SEED", crop, int(k)])
                budget -= k * pc.CROPS[crop]["seed"]
                st["cogs_today"] += k * pc.CROPS[crop]["seed"]  # expB-302 ledger

    # ---- M5b hire-depth controller (expB-221; spec M5 rule, live inputs) ----
    # hands = clamp(min(queue_need, cash_cap), 6, 14): queue_need converts the
    # day's serviceable work into unit-days at the MEASURED productive rate;
    # cash_cap is the deepest fib ladder the income rule affords (beta x
    # trailing-day gross, else gamma x cash) — every input is a runtime
    # measurement or a structural engine constant, no dollar thresholds.
    if P["m5b_hire"]:
        pending_place = sum(shed.get(a, 0) + carried_animals[a] for a in pc.ANIMALS)
        # the day's task queue = standing service (herd + crops) PLUS growth
        # work: builds/placements (~3 actions each: build, place, same-day
        # feed) and the planting backlog (~2 each: plant + same-day water).
        # The first cut counted standing assets only and under-hired the ramp
        # to the clamp floor (expB-221 screen 3.1%: 6-7 hands on days 0-8).
        plant_backlog = (max(0, want_wheat - crop_counts.get("WHEAT", 0))
                         + max(0, pc._want_carrots(P, prices, day) - crop_counts.get("CARROT", 0))
                         + max(0, want_melon - crop_counts.get("MELON", 0)))
        queue_est = (herd * P["actions_per_animal"]
                     + 1.3 * (crop_counts.get("WHEAT", 0) + crop_counts.get("CARROT", 0)
                              + crop_counts.get("MELON", 0))
                     + 3.0 * pending_place + 2.0 * plant_backlog)
        queue_need = max(0, math.ceil(queue_est / max(1.0, prod_rate)) - 1)
        trailing = st["gross_days"][-1] if st["gross_days"] else st["gross_today"]
        hire_budget = max(P["m5b_beta"] * trailing, P["m5b_gamma"] * money)
        n_aff, spent = 0, 0
        while n_aff < 14 and spent + pc._fib(n_aff) <= hire_budget:
            spent += pc._fib(n_aff)
            n_aff += 1
        want_hands = max(6, min(14, min(queue_need, n_aff)))
        if endgame and P["smart_endgame"]:
            want_hands = min(14, want_hands + 4)
    else:
        want_hands = P["hands_target"] + (4 if (endgame and P["smart_endgame"]) else 0)
    st["want_hands"] = want_hands
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
                st["cogs_today"] += k * wq  # expB-302 lambda_opp ledger
        hire(min(4, hires_needed))
        buy_seeds()
    else:
        buy_feed(keepalive_only=True)
        # M5b hires incrementally through the day as the queue grows (hands
        # vanish nightly; a mid-day hire still buys the remaining hours at the
        # same fib cost, and cash_cap/queue_need already price the depth).
        # The legacy arm keeps the hour<=1 gate.
        if hires_needed and (hour <= 1 or (P["m5b_hire"] and hour < 20)):
            hire(hires_needed)
        buy_land()
        buy_animals()
        buy_feed()
        buy_seeds()
    orders = orders[:10]

    # ---- unit tasks: M4 service scheduler ------------------------------------
    tasks = []
    # expB-304a: the dispatch pricer. `urgent` holds the keys whose value is
    # irrecoverable if the task is skipped today; it replaces the `prio <= 2.5`
    # band test and is the only thing exempt from the lambda_opp cut.
    priced = priced_dispatch
    urgent = set()
    pricer = Pricer(model, pc.CROPS, day, hour, last_day, lam_cut, P["care"]) \
        if priced else None

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
        if priced:
            tasks += pricer.animal_tasks(animals, wheat_stock, urgent)
        else:
            tasks += animal_tasks(animals, wheat_stock, P, hour, day >= last_day - 1)

        def _straw_fert(t):
            return 3.4 if (straw_fert and shed.get("FERTILIZER", 0) > 0) else False

        if P["m6_fert"] == "opp":  # expB-302 transfer price, decided above
            def fert_rule(crop, t):
                if crop == "STRAWBERRY":
                    return _straw_fert(t)
                if not (fert_internal and crop == "WHEAT"
                        and 1 <= day - t["planted_day"] <= 2
                        and t.get("fertilized_until_day", -1) < day
                        and shed.get("FERTILIZER", 0) > 0):
                    return False
                # Dispatch priority from engine truth, not from tuning. A
                # FERTILIZE inside the bonus window turns each of the tile's
                # three bonus waters from +1 into +2 yield, so it is worth
                # exactly 2x the bonus WATER it sits next to in the queue —
                # and `plant_tasks` already bands bonus watering at 3.5. 3.4
                # places it immediately above the task it dominates 2:1 and
                # strictly below every keep-alive band (0.25-1.5) and melon
                # watering (2.4). Left at the legacy band 7 the rule is
                # unreachable: measured on seed 0, 48 firings produced 0
                # dispatched FERTILIZE actions and a byte-identical episode.
                return 3.4
        elif P["m6_fert"] == "shadow":
            def fert_rule(crop, t):
                if crop == "STRAWBERRY":
                    return _straw_fert(t)
                if crop != "WHEAT":
                    return False
                age = day - t["planted_day"]
                return (age <= 2 and shed.get("FERTILIZER", 0) > 0
                        and prices.get("FERTILIZER", 100) + lam < 2 * wheat_shadow)
        elif P["m6_fert"] == "sell":
            def fert_rule(crop, t):
                # "sell" is the WHEAT ruling (expB-206); strawberry is priced
                # on its own alternative use and is unaffected by it.
                return _straw_fert(t) if crop == "STRAWBERRY" else False
        else:  # "probe" legacy
            fert_rule = None
        if priced:
            # the priced FERTILIZE row carries its own value, so the rule only
            # has to answer "is this tile eligible" — a float return is ignored
            _pf = fert_rule
            tasks += pricer.plant_tasks(
                plants, urgent,
                (lambda crop, t: bool(_pf(crop, t))) if _pf is not None else None)
        else:
            tasks += plant_tasks(plants, day, last_day, P, pc.CROPS, fert_rule)

        avail_geese = shed.get("GOOSE", 0) + carried_animals["GOOSE"]
        avail_pasture = (shed.get("COW", 0) + carried_animals["COW"]
                         + shed.get("SHEEP", 0) + carried_animals["SHEEP"])
        coop_needs = max(0, avail_geese - sum(1 for _, k in empty_structs if k == "COOP"))
        past_needs = max(0, avail_pasture - sum(1 for _, k in empty_structs if k == "PASTURE"))
        def _build_prio(animal):
            # the structure is worth what the animal waiting for it will still
            # produce, less the PLACE action it commits (the animal's purchase
            # price is sunk and is not charged to the dispatch decision)
            if not priced:
                return 2
            return -max(0.0, pricer.animal_stream(animal, day) - lam_cut)

        build_value = 0.0   # value a DIG unlocks by freeing a site for a build
        sited = 0
        for _ in range(coop_needs):
            p = site.take_animal_site()
            if p is None:
                break
            sited += 1
            tasks.append((_build_prio("GOOSE"), p, ["BUILD_COOP"], ("B", p[0], p[1])))
        _pasture_animal = "COW" if (shed.get("COW", 0) + carried_animals["COW"]) > 0 else "SHEEP"
        for _ in range(past_needs):
            p = site.take_animal_site()
            if p is None:
                break
            sited += 1
            tasks.append((_build_prio(_pasture_animal), p, ["BUILD_PASTURE"],
                          ("B", p[0], p[1])))
        if priced and sited < coop_needs + past_needs:
            # animals in hand with nowhere to go: a dug tile becomes a site
            _a = "GOOSE" if sited < coop_needs else _pasture_animal
            build_value = max(0.0, pricer.animal_stream(_a, day) - 2 * lam_cut)
        # no placements at hour 23: an animal placed on the day's final turn
        # cannot be fed the same day and starts life with consecutive_unfed 1
        if hour < 23:
            for (x, y), kind in empty_structs:
                if kind == "COOP":
                    animal = "GOOSE"
                else:
                    animal = "COW" if (shed.get("COW", 0) + carried_animals["COW"]) > 0 else "SHEEP"
                if shed.get(animal, 0) + carried_animals[animal] > 0:
                    prio = -pricer.animal_stream(animal, day) if priced else 0
                    tasks.append((prio, (x, y), ["PLACE", animal], ("P", x, y)))
        plan = []
        if capex_ok and hour < 20:
            # no planting after hour 19: the planting day counts as unwatered,
            # so a plant that cannot be watered tonight weeds out (3 of 9
            # weed-kills in expB-213 v9 were age-0 late plantings)
            melon_plantable = melon_window or not P["melon_replant"]
            plan += ["MELON"] * (max(0, want_melon - crop_counts.get("MELON", 0))
                                 if melon_plantable else 0)
            plan += ["STRAWBERRY"] * (max(0, want_straw - n_straw) if straw_window else 0)
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
                prio = -pricer.fresh_plant_value(crop) if priced else 8
                tasks.append((prio, p, ["PLANT", crop], ("PL", p[0], p[1])))

        # ---- expB-304a / ruling-02 R1: DIG the weeds back into service -------
        # A weeded tile costs one action to recover and the champion never spent
        # it, so 89 tile-days a season sat dead. DIG is priced by what the tile
        # unlocks — the best plant we still want, or a site for an animal in
        # hand — so it competes with everything else instead of needing a band.
        if priced and P["dig"] and weeds:
            plant_value = 0.0
            if capex_ok:
                for crop in ({c for c in plan} if plan else ()):
                    plant_value = max(plant_value, pricer.fresh_plant_value(crop))
            tasks += pricer.dig_tasks(weeds, plant_value, build_value)

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
            # Ordered candidates for the tile under this unit's feet, in the
            # elif-chain order this loop has always used. `dedupe_ontile` then
            # takes the first one NOBODY has claimed yet instead of blindly
            # taking the head: two units standing on the same tile (common
            # around the shed and the cluster anchors) were both emitting the
            # same action, and the second one was a no-op. Measured on seed 0:
            # 528 such unit-turns a season — already_fed 94, already_cared 74,
            # already_watered 70, fertilizer already collected 157, nothing left
            # to harvest 133 — i.e. every single no-op the census reports.
            cands = []
            if "animal" in t:
                cap = 4 if t["animal"] == "GOOSE" else 6
                if not t["fed_today"] and invs[i].get("WHEAT", 0) > 0:
                    cands.append((("F", x, y), ["FEED"]))
                if t.get("fertilizer_available"):
                    # collected even in feed-first mode: one action, and a
                    # skipped collection is lost permanently at the refresh
                    cands.append((("CF", x, y), ["COLLECT_FERTILIZER"]))
                if not feed_first:  # this carrier's wheat is needed elsewhere first
                    if t.get("yield_units", 0) >= cap - 2:
                        cands.append((("H", x, y), ["HARVEST"]))
                    if P["care"] and not t["cared_today"] and t["fed_today"]:
                        # care on an unfed animal is worth zero
                        cands.append((("C", x, y), ["CARE"]))
            elif t.get("kind") == "PLANT":
                crop = t["crop"]
                cd = pc.CROPS.get(crop)
                age = day - t["planted_day"]
                r_age = cd["first"] if crop == "MELON" else cd["maxday"] if cd else 99
                dying = t["consecutive_unwatered"] >= 1 and not t["watered_today"]
                # ruling-02 R3: for an `ongoing` crop the service span is the
                # LAST PRODUCTION EVE, not max_yield_day. STRAWBERRY has
                # first == maxday == 10 and produces on ages 9/11/13/15, so
                # `age < cd["maxday"]` stopped this loop watering the tile from
                # age 10 while it still had three events to run — and it kept
                # HARVESTing it instead. Two unwatered days weed the tile out at
                # ANY age: that is the age-12.9 strawberry weed mode (272 of 444
                # kills).
                if cd and cd.get("ongoing") and P["ongoing_water_span"]:
                    _iv = max(1, int(cd.get("interval", 1)))
                    water_span = cd["first"] - 1 + _iv * (int(cd.get("max_yield", 1)) - 1)
                    water_ok = age <= water_span
                else:
                    water_ok = bool(cd) and age < cd["maxday"]
                _ready = bool(cd) and (age >= r_age) and t.get("yield_units", 0) > 0
                # a ready non-ongoing tile is about to be harvested, and HARVEST
                # clears it: watering it in the same turn is an engine-rejected
                # action (see pricer.plant_tasks)
                _spent = (P["dedupe_ontile"] and _ready and not dying
                          and bool(cd) and not cd.get("ongoing")
                          and age >= cd["maxday"])
                if cd and not t["watered_today"] and water_ok and not _spent                         and (dying or not feed_first):
                    cands.append((("W", x, y), ["WATER"]))  # keep-alive trumps feed-first
                if _ready and (dying or not feed_first):
                    cands.append((("H", x, y), ["HARVEST"]))
            act = key = None
            if P["dedupe_ontile"]:
                for k, a in cands:
                    if k not in claimed:
                        key, act = k, a
                        break
            elif cands:
                key, act = cands[0]
            if act:
                actions[i] = act
                busy[i] = True
                claimed.add(key)

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
        # expB-304a: the keys whose value is irrecoverable if skipped today, and
        # the cut below which a unit-turn is worth more idle than spent.
        "urgent": urgent,
        "lam": (lam_cut if P["lam_cut"] else 0.0) if priced else 0.0,
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
        if not (isinstance(a, list) and a):
            continue
        if a[0] == "PASS":
            st["idle_today"] += 1  # expB-302: binding-ness of the action budget
        elif a[0] not in MOVES:
            st["prod_today"] += 1
    st["last_orders"] = orders

    # ---- realised melon yield ledger (expB-220, corrected expB-222) ---------
    # Record a melon's yield_units at the turn a HARVEST order is actually
    # issued for its tile (HARVEST acts on the issuing unit's own tile, so
    # every issuance — on-tile loop, router task, endgame — lands here), NOT
    # at first eligibility: the PR#14 audit showed 60% of melon weed-kills
    # happen after age >= first (phantom "realised" revenue for tiles that
    # bank $0) and survivors keep accruing yield until maxday (undercount).
    # At issuance the tile is alive and ready, so yield_units IS what banks.
    for i, a in enumerate(actions):
        if isinstance(a, list) and a and a[0] == "HARVEST" and i < len(units):
            ux, uy = units[i]
            t = tiles[uy][ux]
            if isinstance(t, dict) and t.get("kind") == "PLANT" \
                    and t.get("crop") == "MELON":
                key = (ux, uy, t["planted_day"])
                if key not in st["melon_seen"]:
                    st["melon_seen"].add(key)
                    st["melon_harvest_units"].append(t.get("yield_units", 0))
            # STRAWBERRY is ongoing: one tile is harvested max_yield times, so
            # the ledger accumulates PER TILE and the running total is the
            # conservative estimate of that tile-cycle's yield (it can only
            # grow). Recorded at harvest issuance, per the expB-222 ruling.
            elif isinstance(t, dict) and t.get("kind") == "PLANT" \
                    and t.get("crop") == "STRAWBERRY":
                key = (ux, uy, t["planted_day"])
                st["straw_tiles"][key] = (st["straw_tiles"].get(key, 0)
                                          + t.get("yield_units", 0))
                st["straw_harvest_units"] = list(st["straw_tiles"].values())

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
    # expB-302: "opp" = internal transfer price for fertilizer —
    # 2 x marginal_buy_cost(WHEAT, shortfall) > marginal_revenue(FERT, 1) +
    # lambda_opp. Default off = shipped 725fcf7 behaviour (always sell).
    "m6_fert": "sell",
    "actions_per_animal": 4.0,  # feed + collect + care + amortised harvest
    # expB-219 (stock-pool saturation): structural melon replant window
    # (plant while day + first_yield <= last_day) instead of the legacy
    # day<=18 seed gate; also blocks dead plantings (a melon planted after
    # the window can never yield). Default off = shipped e8a76b6 behaviour.
    "melon_replant": False,
    # expB-220: "mr" sizes melon acreage by marginal revenue per productive
    # action vs lambda (no tile-count constant); "fixed" uses melon_tiles.
    "melon_rule": "fixed",
    # expB-221 (M5b): hire depth from the income/queue rule instead of the
    # fixed hands_target. beta/gamma per spec M5 (expB-209 fits them).
    # expB-303: STRAWBERRY acreage. "fixed" = the sweep's arm (0 = shipped
    # 725fcf7 behaviour, the crop is never planted); "mr" sizes it by marginal
    # revenue per productive action against lambda_opp, no tile-count constant.
    "strawberry_tiles": 0,
    "strawberry_rule": "fixed",
    "m5b_hire": False,
    "m5b_beta": 0.10,
    "m5b_gamma": 0.03,
    # ---- expB-304a: the dispatch pricer (ruling-02 sections 2.3-2.5) ---------
    # "bands"  = the hand-set 0.25..8 priority constants (shipped h9d854a3d8c);
    # "priced" = every candidate action carries its marginal contribution from
    #            M1 live prices and the queue executes in descending
    #            contribution per unit-turn down to the lambda_opp cut.
    "dispatch": "bands",
    # DIG a weeded tile back into service (R1). Rides inside the pricer: a
    # weeded tile is a candidate action worth what it unlocks, which is exactly
    # the object the pricer prices, and standalone it is unpowered ($1.0k
    # against a $7.1k paired SD).
    "dig": False,
    # Claim-leak fix: the on-tile loop never checked `claimed`, so two units on
    # one tile both emitted the same action. Every no-op the census reports.
    "dedupe_ontile": False,
    # R3: service an `ongoing` crop to its LAST PRODUCTION EVE, not to
    # max_yield_day (the age-12.9 strawberry weed mode).
    "ongoing_water_span": False,
    # The lambda_opp cut itself. True = refuse a task whose contribution per
    # unit-turn (travel included) is below the season-to-date contribution per
    # productive action; False = order by price but refuse only NEGATIVE work.
    # Decomposition arm: the cut is the half of the pricer that can destroy
    # value if the estimator is biased, the ordering is the half that cannot.
    "lam_cut": True,
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
                       "prod_days": [], "melon_seen": set(), "melon_harvest_units": [],
                       "gross_today": 0.0, "gross_days": [], "day_open": -1,
                       "clamp_bound_today": False, "m5b_log": [],
                       # expB-302 lambda_opp ledgers
                       "cogs_today": 0.0, "contrib_days": [], "idle_today": 0,
                       "idle_days": [],
                       # expB-303 strawberry ledgers
                       "straw_tiles": {}, "straw_harvest_units": [],
                       # expB-304a: full-season lambda windows + telemetry
                       "contrib_num": 0.0, "contrib_den": 0.0,
                       "idle_num": 0.0, "idle_den": 0.0,
                       "lam_sum": 0.0, "lam_opp_sum": 0.0, "lam_cut_sum": 0.0,
                       "lam_n": 0, "lam_pinned": 0,
                       # expB-304a: both lambda series, one row per day
                       # (ruling-02 section 2.7 makes this mandatory: a
                       # lambda_opp pinned at the lambda_wage floor means the
                       # pricer is silently a wage gate and the arm is VOID)
                       "lam_log": []},
                "router": (PricedRouter() if P["dispatch"] == "priced"
                           else FeedRouter()),
                "model": MarketModel()}

    return make_chassis(_plan, state_factory)
