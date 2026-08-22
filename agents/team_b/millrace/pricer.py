"""Millrace M2b — the dispatch pricer (expB-304a).

Replaces the hand-set dispatch bands (`service.py`: FEED-critical 0.25, PLACE 0,
WATER-critical 0.5, harvest-or-lose 0.6, FEED 1, BUILD 2, COLLECT_FERTILIZER
2.5/3, HARVEST 4, WATER-bonus 5, CARE 6, FERTILIZE 7, PLANT 8) with a **price**:
every candidate action carries the marginal dollar contribution of completing
it, derived from M1's live observed price table and the engine's own production
rules, and the queue executes in descending contribution per unit-turn down to
the lambda_opp cut.

Why (expB-301, ruling-02 sections 2.2-2.3): the bands are 12 absolute constants
that encode an ordering nobody has re-derived since expB-211, and they produce
three measured pathologies —
  * 477-559 unit-turns a season that change nothing (a claim-leak, fixed by
    `dedupe_ontile`, not by pricing);
  * PLANT parked at band 8 behind ~1,900 committed actions, so the arm asks for
    16 strawberry tiles and stands 3.3;
  * GOOSE service at $4.3 of contribution per action against a marginal task
    worth $35-45 — 22% of the action budget spent below the cut.
A price fixes all three by construction and deletes the constants.

What is priced, and off which engine fact (1.32.7, read in
`_apply_unit_action` / `_daily_refresh_animals` / `_daily_refresh_plants`):

  FEED, critical      the animal escapes tonight at `consecutive_unfed >= 2`.
                      Irrecoverable: no purchase buys the animal-days back.
                      Priced at the animal's whole remaining stream, and flagged
                      URGENT so it is never subject to the cut. This is the one
                      lexicographic safety layer in the module (spec section 6.1's
                      "irrecoverable vs delayed" line, ruling-02 section 1.2).
  FEED, routine       base production and `fertilizer_available` are set
                      UNCONDITIONALLY on a scheduled day; `fed_today` buys only
                      the CARE bonus. So a routine feed is worth
                      (CARE bonus it unlocks) - (wheat it burns) - (the CARE
                      action that must follow). For a goose that is negative at
                      any plausible egg price, which is exactly the legal
                      feed-every-other-day keep-alive (`consecutive_unfed` never
                      reaches 2) that ruling-02 section 3.2 prices at $8-11k.
  CARE                +1 unit of the animal's product on its next production day,
                      worth 0 if the tile is already at `max_held` or no
                      production day is left in the season.
  COLLECT_FERTILIZER  1 unit; `fertilizer_available` re-arms nightly and does not
                      accumulate, so the loss from skipping is exactly one unit.
  HARVEST (animal)    the standing units at M1's marginal sale price, plus the
                      overflow tonight's production would lose against `max_held`.
  WATER               survival value when `consecutive_unwatered >= 1` (the tile
                      weeds out tonight) plus the yield the water itself adds:
                      +1 (+2 fertilized) inside the non-ongoing bonus window
                      [(max_yield_day+1)//2 .. max_yield_day]; for an ongoing crop
                      watering adds NO base yield (the engine grants the base unit
                      regardless) and is worth the fertilizer bonus on a
                      production eve.
  HARVEST (plant)     standing units, plus - for an ongoing crop on a production
                      eve - the units tonight's event would throw away because
                      `yield_units` is a CAP, not a season total.
  FERTILIZE           the extra units the 3-day application actually buys, less
                      the fertilizer unit's own sale value.
  PLANT               the tile-cycle's units less seed and the actions it commits.
  DIG                 (new, ruling-02 R1) a weeded tile is recoverable: DIG sets
                      it to None and PLANT requires `tile is None`. Worth the
                      plant that will occupy it, less the PLANT action.
  BUILD / PLACE       the animal's remaining production stream (the animal is
                      already bought: its cost is sunk and is NOT charged here).

Everything above is a runtime measurement (M1 marginal revenue / marginal buy
cost off the observed table) or a structural engine constant. No absolute dollar
or absolute day threshold enters — spec section 0.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

try:  # package context (repo tools/tests)
    from .layout import dist, quadrant_of
    from .service import FeedRouter
except ImportError:  # Kaggle/arena loader: exec'd flat, dir on sys.path
    from layout import dist, quadrant_of
    from service import FeedRouter

# Structural engine tables, straight off kaggriculture 1.32.7 ANIMALS/CROPS.
# Same class of constant as `probe_core.CROPS`: the observation does not expose
# the engine's own tables, so a production model has to carry them.
ANIMAL_ENGINE = {
    "GOOSE": {"first": 4, "interval": 1, "max_held": 4},
    "COW":   {"first": 8, "interval": 2, "max_held": 6},
    "SHEEP": {"first": 6, "interval": 3, "max_held": 6},
}
CROP_MAX_YIELD = {"WHEAT": 6, "CARROT": 4, "MELON": 6, "STRAWBERRY": 4, "TOMATO": 4}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


def _cadence_count(first_ok_day, interval, lo, hi):
    """How many days d in [lo, hi] satisfy d >= first_ok_day and
    (d - first_ok_day) % interval == 0."""
    if interval <= 0:
        interval = 1
    lo = max(lo, first_ok_day)
    if hi < lo:
        return 0
    r = (lo - first_ok_day) % interval
    start = lo + ((interval - r) % interval)
    if start > hi:
        return 0
    return (hi - start) // interval + 1


class Pricer:
    """One instance per turn. Caches M1 walks (the same (product, n) pair is
    asked for by dozens of tasks a turn and each walk is O(n) bisects)."""

    def __init__(self, model, crops, day, hour, last_day, lam, care_on):
        self.model = model
        self.crops = crops
        self.day = day
        self.hour = hour
        self.last_day = last_day
        self.lam = lam
        self.care_on = care_on
        self._mr = {}
        self._buy = {}

    # ---- M1 primitives -------------------------------------------------------
    def mr(self, product, n):
        """Proceeds of selling `n` more units now, off the observed table."""
        n = int(n)
        if n <= 0:
            return 0.0
        key = (product, n)
        v = self._mr.get(key)
        if v is None:
            v = self.model.marginal_revenue(product, n)
            self._mr[key] = v
        return v

    def dmr(self, product, base, extra):
        """Value of `extra` units on top of `base` already being sold."""
        if extra <= 0:
            return 0.0
        return self.mr(product, base + extra) - self.mr(product, base)

    def buy(self, product, n=1):
        key = (product, int(n))
        v = self._buy.get(key)
        if v is None:
            v = self.model.marginal_buy_cost(product, n)
            self._buy[key] = v
        return v

    # ---- forward streams -----------------------------------------------------
    def animal_events_left(self, animal, placed_day):
        """Production events still harvestable this season. The event resolves
        at the END of day d (`_daily_refresh_animals` computes with next_day),
        so it can only be banked if d + 1 <= last_day."""
        ae = ANIMAL_ENGINE.get(animal)
        if ae is None:
            return 0
        # k = (d + 1) - placed - first >= 0 and k % interval == 0
        first_ok = placed_day + ae["first"] - 1
        return _cadence_count(first_ok, ae["interval"], self.day, self.last_day - 1)

    def animal_stream(self, animal, placed_day):
        """Gross value of an animal's remaining base production. Feed and
        service are decided action by action (and each of those decisions is
        priced on its own), so they are not netted off here — this is what the
        asset is worth if we keep it, which is what an escape destroys."""
        n = self.animal_events_left(animal, placed_day)
        if n <= 0:
            return 0.0
        return self.mr(ANIMAL_PRODUCT[animal], n)

    def plant_forward(self, tile):
        """What the standing tile is still worth: units it will yield at M1
        prices, less the unit-turns it will cost to get them."""
        crop = tile["crop"]
        cd = self.crops.get(crop)
        if cd is None:
            return 0.0
        age = self.day - tile["planted_day"]
        y = int(tile.get("yield_units", 0) or 0)
        cap = CROP_MAX_YIELD.get(crop, 6)
        if cd.get("ongoing"):
            interval = max(1, int(cd.get("interval", 1)))
            eves = int(cd.get("max_yield", 1))
            last_eve_day = tile["planted_day"] + cd["first"] - 1 + interval * (eves - 1)
            first_ok = tile["planted_day"] + cd["first"] - 1
            n = _cadence_count(first_ok, interval, self.day, min(last_eve_day, self.last_day - 1))
            units = y + n
            # one WATER per day the tile must stay alive + one HARVEST per event
            service = max(0, min(last_eve_day, self.last_day) - self.day)
            actions = service + n
        else:
            ws = (cd["maxday"] + 1) // 2
            hi = min(cd["maxday"], self.last_day - tile["planted_day"])
            nb = _cadence_count(max(age, ws), 1, max(age, ws), hi)
            units = min(cap, y + nb)
            actions = nb + 1
        return max(0.0, self.mr(crop, units) - actions * self.lam)

    def fresh_plant_value(self, crop):
        """Contribution of planting `crop` on an empty tile today, net of seed
        and of every unit-turn the cycle commits. Negative means do not plant."""
        cd = self.crops.get(crop)
        if cd is None:
            return 0.0
        seed = cd.get("seed", 0)
        planted = self.day
        cap = CROP_MAX_YIELD.get(crop, 6)
        if cd.get("ongoing"):
            interval = max(1, int(cd.get("interval", 1)))
            eves = int(cd.get("max_yield", 1))
            last_eve_day = planted + cd["first"] - 1 + interval * (eves - 1)
            first_ok = planted + cd["first"] - 1
            n = _cadence_count(first_ok, interval, planted, min(last_eve_day, self.last_day - 1))
            units = n
            service = max(0, min(last_eve_day, self.last_day) - planted)
            actions = 1 + service + n
        else:
            ws = (cd["maxday"] + 1) // 2
            hi = min(cd["maxday"], self.last_day - planted)
            nb = _cadence_count(ws, 1, ws, hi)
            units = min(cap, 1 + nb)
            if planted + cd["first"] > self.last_day:
                units = 0
            actions = 1 + nb + 1
        return self.mr(crop, units) - seed - actions * self.lam

    # ---- task pricing --------------------------------------------------------
    def animal_tasks(self, animals, wheat_stock, urgent):
        """Priced rows for animal tiles. Returns [(-value, pos, act, key)] and
        fills `urgent` with the keys that are irrecoverable if skipped."""
        rows = []
        n_critical = sum(1 for _, t in animals
                         if not t["fed_today"] and t.get("consecutive_unfed", 0) >= 1)
        routine_budget = wheat_stock - n_critical
        for (x, y), t in animals:
            animal = t["animal"]
            ae = ANIMAL_ENGINE.get(animal, {"interval": 1, "max_held": 6})
            product = ANIMAL_PRODUCT.get(animal, "EGG")
            held = int(t.get("yield_units", 0) or 0)
            room = max(0, ae["max_held"] - held)
            events = self.animal_events_left(animal, t.get("placed_day", self.day))
            if not t["fed_today"] and wheat_stock > 0:
                if t.get("consecutive_unfed", 0) >= 1:
                    key = ("F", x, y)
                    urgent.add(key)
                    rows.append((-self.animal_stream(animal, t.get("placed_day", self.day)),
                                 (x, y), ["FEED"], key))
                elif routine_budget > 0:
                    routine_budget -= 1
                    # The only thing a routine feed buys is the CARE bonus.
                    bonus = self.mr(product, 1) if (self.care_on and room > 0
                                                    and events > 0) else 0.0
                    v = bonus - self.buy("WHEAT", 1) - self.lam
                    rows.append((-v, (x, y), ["FEED"], ("F", x, y)))
            if t.get("fertilizer_available"):
                rows.append((-self.mr("FERTILIZER", 1), (x, y),
                             ["COLLECT_FERTILIZER"], ("CF", x, y)))
            if self.care_on and not t["cared_today"] and t["fed_today"]:
                v = self.mr(product, 1) if (room > 0 and events > 0) else 0.0
                rows.append((-v, (x, y), ["CARE"], ("C", x, y)))
            if held > 0:
                # tonight's event throws away whatever will not fit under max_held
                tonight = 1 if self.animal_events_left(
                    animal, t.get("placed_day", self.day)) > 0 else 0
                overflow = max(0, held + tonight - ae["max_held"])
                v = self.mr(product, held) + self.dmr(product, held, overflow)
                rows.append((-v, (x, y), ["HARVEST"], ("H", x, y)))
        return rows

    def plant_tasks(self, plants, urgent, fertilize_fn=None):
        """Priced rows for plant tiles."""
        rows = []
        day, last_day = self.day, self.last_day
        for (x, y), t in plants:
            crop = t["crop"]
            cd = self.crops.get(crop)
            if cd is None:
                continue
            age = day - t["planted_day"]
            held = int(t.get("yield_units", 0) or 0)
            cap = CROP_MAX_YIELD.get(crop, 6)
            ongoing = bool(cd.get("ongoing"))
            interval = max(1, int(cd.get("interval", 1)))
            # service span. For an ongoing crop that is the LAST PRODUCTION EVE
            # (ruling-02 R3). For a non-ongoing crop the engine's watering bonus
            # window is [(max_yield_day+1)//2 .. max_yield_day] INCLUSIVE, so
            # `maxday - 1` (inherited from service.plant_tasks) drops the last
            # and most valuable watered day of every wheat/carrot/melon tile.
            last_eve = (cd["first"] - 1 + interval * (int(cd.get("max_yield", 1)) - 1)
                        if ongoing else cd["maxday"])
            prod_eve = (ongoing and cd["first"] - 1 <= age <= last_eve
                        and (age - (cd["first"] - 1)) % interval == 0)
            fertilized = t.get("fertilized_until_day", -1) >= day
            lapsed = t["consecutive_unwatered"] >= 1 and not t["watered_today"]

            ready_age0 = cd["first"] if (crop == "MELON" or ongoing) else cd["maxday"]
            ready0 = age >= ready_age0 or (day >= last_day - 1 and age >= cd["first"])
            # A non-ongoing tile that is ready with yield on it is about to be
            # HARVESTed, and HARVEST CLEARS the tile — so a WATER dispatched to
            # the same tile in the same turn is thrown away. (Measured: 53
            # engine-rejected WATERs over 8 seat-seasons once the pricer started
            # unstarving both queues.) The keep-alive case still stands: if the
            # tile dies tonight the yield goes with it.
            # NB `ready0` is not the test: MELON is harvestable from age 10 but
            # keeps ADDING yield to age 12, so the tile is only "spent" once it
            # is past max_yield_day and a water can no longer add anything.
            spent = (not ongoing) and age >= cd["maxday"] and held > 0 and not lapsed
            if not t["watered_today"] and age <= last_eve and not spent:
                v = 0.0
                if lapsed:
                    # the tile weeds out tonight; DIG makes the TILE recoverable
                    # but the standing crop and its sunk tile-days are not
                    v += self.plant_forward(t)
                    urgent.add(("W", x, y))
                if ongoing:
                    # base production is granted watered or not; watering only
                    # unlocks the fertilizer bonus on a production eve
                    if prod_eve and fertilized:
                        v += self.dmr(crop, held, 1)
                else:
                    ws = (cd["maxday"] + 1) // 2
                    if ws <= age <= cd["maxday"] and held < cap:
                        b = 2 if fertilized else 1
                        v += self.dmr(crop, held, min(b, cap - held))
                rows.append((-v, (x, y), ["WATER"], ("W", x, y)))

            if ready0 and held > 0:
                v = self.mr(crop, held)
                if lapsed:
                    urgent.add(("H", x, y))
                elif prod_eve:
                    # `yield_units` is a CAP: whatever tonight's event cannot fit
                    # is thrown away unless the tile is emptied first
                    tonight = 2 if fertilized else 1
                    v += self.dmr(crop, held, max(0, held + tonight - cap))
                rows.append((-v, (x, y), ["HARVEST"], ("H", x, y)))

            if fertilize_fn is not None and t.get("fertilized_until_day", -1) < day:
                extra = self.fertilizer_units(t, crop, cd, age, held, cap, ongoing,
                                              interval, last_eve)
                if extra > 0 and fertilize_fn(crop, t):
                    v = self.dmr(crop, held, extra) - self.mr("FERTILIZER", 1)
                    rows.append((-v, (x, y), ["FERTILIZE"], ("FZ", x, y)))
        return rows

    def fertilizer_units(self, t, crop, cd, age, held, cap, ongoing, interval, last_eve):
        """Extra units one FERTILIZE actually buys. It covers day..day+2."""
        if ongoing:
            # +1 per production eve inside the 3-day cover, and only if the tile
            # is watered that day (the caller waters production eves)
            n = 0
            for k in range(3):
                a = age + k
                if cd["first"] - 1 <= a <= last_eve and (a - (cd["first"] - 1)) % interval == 0:
                    n += 1
            return max(0, min(n, cap - held))
        ws = (cd["maxday"] + 1) // 2
        n = sum(1 for k in range(3) if ws <= age + k <= cd["maxday"])
        return max(0, min(n, cap - held))

    def dig_tasks(self, weeds, plant_value, build_value):
        """DIG a weeded tile back to None (ruling-02 R1). `plant_value` is the
        best fresh-plant contribution available today, `build_value` the value
        of a structure a pending animal is waiting for; both already net of the
        follow-up action, so DIG carries what the tile unlocks."""
        v = max(plant_value, build_value) - self.lam
        if v <= 0:
            return []
        return [(-v, (x, y), ["DIG"], ("D", x, y)) for (x, y) in weeds]


class PricedRouter(FeedRouter):
    """FeedRouter with the band comparison replaced by contribution per
    unit-turn. Everything else — home-cluster binding, target locks, batched
    pickups, requirement-aware picking, the keep-alive borrowing rule — is
    unchanged, so expB-304a moves exactly one variable.

    `ctx["urgent"]` is the set of task keys whose value is irrecoverable if the
    task is skipped today (an animal that escapes tonight, a tile that weeds out
    tonight with a crop on it). Those replace the `prio <= URGENT_PRIO` band
    test for pre-emption / cross-cluster borrowing, and they alone bypass the
    lambda_opp cut.
    """

    def _is_urgent(self, prio, key, ctx):
        return key in ctx.get("urgent", ())

    def _pick_for(self, i, pos, open_tasks, home, ctx, urgent_only=False):
        invs = ctx["invs"]
        shed = ctx["shed"]
        requirement = ctx["requirement"]
        nearest_shed = ctx["nearest_shed"]
        urgent = ctx.get("urgent", ())
        lam = ctx.get("lam", 0.0)
        best, best_score, best_urgent = None, None, False
        home_exec = False
        home_urgent_exec = False
        for t in open_tasks:
            prio, tpos, act, key = t
            req = requirement(act)
            carried = bool(req) and invs[i].get(req, 0) > 0
            if req and not carried and shed.get(req, 0) <= 0:
                continue  # cannot supply this task at all
            if not req or carried:
                d = dist(pos, tpos)
            else:
                sp = nearest_shed(pos)
                d = dist(pos, sp) + dist(sp, tpos) + 1
            is_urgent = key in urgent
            rate = (-prio) / (1.0 + d)
            if not is_urgent and rate < lam:
                continue  # below the marginal task: the unit-turn is worth more idle
            in_home = home is not None and quadrant_of(tpos, self.board) == home
            if in_home:
                home_exec = True
                if is_urgent:
                    home_urgent_exec = True
            if urgent_only and not is_urgent:
                continue
            score = (0 if in_home else 1, -rate)
            if best_score is None or score < best_score:
                best, best_score, best_urgent = t, score, is_urgent
        if best is None:
            return None
        if best_score[0] == 1:  # best task is outside the home cluster
            if best_urgent:
                if home_urgent_exec:
                    return None  # serve the home cluster's own urgent first
            elif home_exec:
                return None  # never leave home for routine work while home has any
        return best
