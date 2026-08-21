"""Millrace M4 — service scheduler: keep-alive absolutes, cadence, priorities.

Engine facts this module encodes (verified against kaggle-environments 1.32.7
`_daily_refresh_animals` / `_daily_refresh_plants` / `_apply_unit_action`):
  - an animal escapes at the refresh when `consecutive_unfed` reaches 2, so an
    animal already at 1 MUST be fed today — that outranks everything;
  - a plant weeds out at `consecutive_unwatered` >= 2 and the PLANTING DAY
    counts as unwatered (init 1), so a new plant must be watered the same day;
    MATURE plants die of thirst too — a ready-to-harvest tile with a lapsed
    watering is harvested at keep-alive priority (harvest-or-lose beats
    spending a second action on water);
  - `fertilizer_available` re-arms every night and does not accumulate: a
    skipped collection is lost permanently -> collect daily, escalating late
    in the day;
  - production fires on interval days regardless of feeding; the CARE bonus is
    consumed only on a fed production day (cows +2 per 2-day cycle, sheep +3
    per 3-day cycle, goose +1 per day) -> cows/sheep are fed+cared daily;
  - `max_held` caps accumulation (goose 4, cow/sheep 6): harvest at cap-2.

expB-211 routed the build order here: 62.5% of unfed animal-days had a
wheat-carrying unit on/adjacent to the tile doing something else — a priority
and pre-emption bug, not a supply problem. The fixes:
  1. keep-alive FEED priority 0.25 (above PLACE) when consecutive_unfed >= 1;
  2. the FeedRouter lets a unit leave its home cluster for an URGENT task when
     its own cluster has no open urgent work (starving animals beat cluster
     discipline);
  3. the arm runs an adjacent-feed pass (feedchain.py): a wheat carrier next
     to an unfed animal steps onto it instead of walking past.

Priorities (lower = first): FEED-critical 0.25, PLACE 0 (unchanged, engine
lead times), WATER-critical 0.75, FEED 1, HARVEST-or-lose 1.5, BUILD 2,
COLLECT_FERTILIZER 3 (2.5 from hour 16), HARVEST 4, WATER-bonus 5, CARE 6,
FERTILIZE 7, PLANT 8.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

try:  # package context (repo tools/tests)
    from .layout import dist, quadrant_of
    from .routing import Router, URGENT_PRIO
except ImportError:  # Kaggle/arena loader: exec'd flat, dir on sys.path
    from layout import dist, quadrant_of
    from routing import Router, URGENT_PRIO

URGENT_FEED = 0.25
URGENT_WATER = 0.5     # dies tonight; below FEED-critical, above routine FEED
HARVEST_OR_LOSE = 0.6  # mature tile with lapsed watering weeds out tonight


def animal_tasks(animals, wheat_stock, P, hour, last_day_soon):
    """Task rows for animal tiles. `animals` = [((x, y), tile_dict)].

    Feed rationing under scarcity: critical animals (escape tonight) always
    eat first; non-criticals only get FEED tasks while wheat remains after
    the criticals are covered. Under sustained shortage this degrades into
    the spec's maintenance mode (feed every other day, alternating cohorts,
    demand halved) instead of random starvation with escapes — the 6-escape
    day-28 poverty incident in the expB-213 gate run fed non-criticals while
    criticals starved."""
    tasks = []
    n_critical = sum(1 for _, t in animals
                     if not t["fed_today"] and t.get("consecutive_unfed", 0) >= 1)
    routine_budget = wheat_stock - n_critical
    for (x, y), t in animals:
        if not t["fed_today"] and wheat_stock > 0:
            critical = t.get("consecutive_unfed", 0) >= 1
            if critical:
                tasks.append((URGENT_FEED, (x, y), ["FEED"], ("F", x, y)))
            elif routine_budget > 0:
                routine_budget -= 1
                tasks.append((1, (x, y), ["FEED"], ("F", x, y)))
        if t.get("fertilizer_available"):
            prio = 2.5 if hour >= 16 else 3
            tasks.append((prio, (x, y), ["COLLECT_FERTILIZER"], ("CF", x, y)))
        # CARE only after feeding: the refresh consumes the bonus only on a fed
        # day, and `cared_today` on an unfed animal is worth exactly zero —
        # expB-213 trace showed units caring for starving animals.
        if P["care"] and not t["cared_today"] and t["fed_today"]:
            tasks.append((6, (x, y), ["CARE"], ("C", x, y)))
        cap = 4 if t["animal"] == "GOOSE" else 6
        if t.get("yield_units", 0) >= cap - 2 or (t.get("yield_units", 0) > 0 and last_day_soon):
            tasks.append((4, (x, y), ["HARVEST"], ("H", x, y)))
    return tasks


def plant_tasks(plants, day, last_day, P, crops, fertilize_fn=None):
    """Task rows for plant tiles. `crops` = the structural CROPS table.
    `fertilize_fn(crop, tile)` -> bool decides FERTILIZE (M6 shadow rule);
    None falls back to P['fertilize_wheat'] on young wheat (probe behaviour)."""
    tasks = []
    for (x, y), t in plants:
        crop = t["crop"]
        cd = crops.get(crop)
        if cd is None:
            continue
        age = day - t["planted_day"]
        lapsed = t["consecutive_unwatered"] >= 1 and not t["watered_today"]
        if not t["watered_today"] and age < cd["maxday"]:
            # melon routine watering sits inside the urgent band: the highest
            # capital tile on the board, 10-12 daily waters, and a single lapse
            # puts it one miss from death (M4 gate run: corner melons died at
            # age 8-11 with 3-5 units of yield on board)
            if crop == "MELON":
                routine = 2.4  # highest-capital tile, 10-12 daily waters
            elif not cd.get("ongoing", False) and (cd["maxday"] + 1) // 2 <= age:
                routine = 3.5  # bonus window: each water is +1 yield
            else:
                routine = 5
            prio = URGENT_WATER if t["consecutive_unwatered"] >= 1 else routine
            tasks.append((prio, (x, y), ["WATER"], ("W", x, y)))
        ready_age = cd["first"] if crop == "MELON" else cd["maxday"]
        ready = age >= ready_age or (day >= last_day - 1 and age >= cd["first"])
        if ready and t.get("yield_units", 0) > 0:
            # a mature unwatered tile weeds out TONIGHT: harvest-or-lose
            prio = HARVEST_OR_LOSE if lapsed else 4
            tasks.append((prio, (x, y), ["HARVEST"], ("H", x, y)))
        elif age >= cd["maxday"] and lapsed and not t["watered_today"]:
            # mature but yieldless (already stripped): keep alive only if the
            # tile still matters — it does not; let it weed out and re-plant.
            pass
        do_fert = (fertilize_fn(crop, t) if fertilize_fn is not None
                   else (P["fertilize_wheat"] and crop == "WHEAT" and age <= 2))
        if do_fert and t.get("fertilized_until_day", -1) < day:
            # `fertilize_fn` may return a float to set the task's own priority
            # (expB-302); a plain True keeps the legacy band-7 placement.
            prio = do_fert if (isinstance(do_fert, float) or isinstance(do_fert, int)
                               and not isinstance(do_fert, bool)) else 7
            tasks.append((prio, (x, y), ["FERTILIZE"], ("FZ", x, y)))
    return tasks


class FeedRouter(Router):
    """Router with requirement-aware picking and keep-alive borrowing (M4).

    Two starvation modes of the base router, both measured in expB-213:
      1. `_pick` was requirement-blind: a wheatless unit would claim a FEED
         task, `_execute` would give up (shed dry — harvested wheat lives in
         harvesters' POCKETS by design), and the claim churned one turn at a
         time while actual wheat carriers were locked on lower-value work.
         Picking now skips tasks the unit cannot supply (not carried AND not
         in the shed) and prices a via-shed trip into the distance, which
         makes carriers win FEED tasks naturally.
      2. A unit could idle over a home-cluster task it could never execute
         while urgent work starved elsewhere. Home-first now counts only
         EXECUTABLE home tasks, and urgent tasks (keep-alive band) may be
         borrowed cross-cluster when the home cluster has no open urgent work.

    `plan` is a copy of Router.plan with the pick call swapped — the base
    stays frozen because the expB-201 champion arm binds to it."""

    def plan(self, day, units, busy, actions, tasks, ctx):
        quads = ctx["quads"]
        if day != self.day:
            self.day = day
            self.locks = {}
            self._rebind(units, tasks, quads)
        for i in range(len(units)):
            if i not in self.home:
                self._bind_new(i, units[i], tasks, quads)
        for i in list(self.locks):
            if i >= len(units):
                del self.locks[i]

        by_key = {t[3]: t for t in tasks}
        claimed = ctx.get("claimed", set())

        for i in list(self.locks):
            lk = self.locks[i]
            if busy[i] or lk["key"] not in by_key or lk["key"] in claimed:
                del self.locks[i]
                continue
            claimed.add(lk["key"])

        open_tasks = [t for t in tasks if t[3] not in claimed]
        for i, pos in enumerate(units):
            if busy[i]:
                continue
            lock = self.locks.get(i)
            if lock is not None and lock["prio"] > URGENT_PRIO:
                urgent = self._pick_for(i, pos, open_tasks, self.home.get(i), ctx, urgent_only=True)
                if urgent is not None:
                    claimed.discard(lock["key"])
                    open_tasks.append(by_key[lock["key"]])
                    lock = None
            if lock is None:
                choice = self._pick_for(i, pos, open_tasks, self.home.get(i), ctx)
                if choice is None:
                    continue
                prio, tpos, act, key = choice
                lock = {"key": key, "pos": tpos, "prio": prio, "act": act,
                        "req": ctx["requirement"](act)}
                claimed.add(key)
                open_tasks = [t for t in open_tasks if t[3] != key]
            self.locks[i] = lock
            actions[i] = self._execute(i, pos, lock, ctx)
            busy[i] = True
            if actions[i] is None:
                actions[i] = ["PASS"]
                claimed.discard(lock["key"])
                del self.locks[i]
                busy[i] = False
        return claimed

    def _pick_for(self, i, pos, open_tasks, home, ctx, urgent_only=False):
        invs = ctx["invs"]
        shed = ctx["shed"]
        requirement = ctx["requirement"]
        nearest_shed = ctx["nearest_shed"]
        best, best_score = None, None
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
            in_home = home is not None and quadrant_of(tpos, self.board) == home
            if in_home:
                home_exec = True
                if prio <= URGENT_PRIO:
                    home_urgent_exec = True
            if urgent_only and prio > URGENT_PRIO:
                continue
            score = (0 if in_home else 1, prio, d)
            if best_score is None or score < best_score:
                best, best_score = t, score
        if best is None:
            return None
        if best_score[0] == 1:  # best task is outside the home cluster
            if best[0] <= URGENT_PRIO:
                if home_urgent_exec:
                    return None  # serve the home cluster's own urgent first
            elif home_exec:
                return None  # never leave home for routine work while home has any
        return best

    def _execute(self, i, pos, lock, ctx):
        # A placer standing at the shed with its animal in hand also grabs
        # wheat for the placement-day feed (placed-today animals were 24% of
        # unfed animal-days in expB-213 — nobody near a fresh PLACE carries
        # wheat, so the same-day FEED never lands).
        req = lock["req"]
        invs = ctx["invs"]
        shed = ctx["shed"]
        if (lock["act"][0] == "PLACE" and req and invs[i].get(req, 0) > 0
                and invs[i].get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0
                and pos in ctx["shed_tiles"]):
            k = min(shed["WHEAT"], ctx["batch"].get("WHEAT", 8))
            shed["WHEAT"] -= k
            return ["PICKUP", "WHEAT", int(k)]
        return super()._execute(i, pos, lock, ctx)
