"""Shared observation -> feature encoding for the Team A behaviour-cloning policy.

This module is imported by BOTH the training-row extractor and the packaged `main.py`.
That is deliberate: the Pokemon-challenge team's most expensive packaging bug was a wrapper
that bypassed the trained policy's feature path and then agreed with it on only 62.9% of
decisions ("a correct model behind an incorrect interface is a different policy"). There is
exactly one feature path here, and `tests/test_bc_parity.py` asserts it.

Design notes
------------
* A Kaggriculture unit op is *positional*: WATER means "water the tile I stand on", so more
  than half of a strong agent's ops are movement. A per-observation classifier can only
  learn movement if it is told, for each direction, how far away the nearest tile needing
  each kind of work is. That is what the "task field" block does, and it is the single most
  important part of the encoding.
* Task-relative features also collapse strategically equivalent choices: three identical
  dry plants produce one feature, so the model is not asked to reproduce which physical
  tile the teacher happened to pick. This is our stand-in for the Pokemon "semantic action
  set" loss.
* Units are encoded autoregressively: unit k sees the tiles that units 0..k-1 already
  claimed this turn, so two farm hands do not chase the same job.

Everything here is stdlib + numpy so the same code runs inside the Kaggle sandbox.
"""
from __future__ import annotations

import math

import numpy as np

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
# market price/inventory keys (9 tradeable products)
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
# every key that can appear in the shed / a unit inventory
ITEMS = PRODUCTS + ANIMALS                      # 12
QUADRANTS = ["NW", "NE", "SW", "SE"]

BOARD = 10
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
TURNS_PER_DAY = 24
DAYS = 30
LAST_ACTING_STEP = 718                          # DONE fires at step 718; day 29 has no end-of-day

# ---------------------------------------------------------------------------
# action vocabulary (factored into three heads)
# ---------------------------------------------------------------------------
OPS = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST",
    "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE", "COLLECT_FERTILIZER",
    "DROP", "PLANT", "PICKUP", "PLACE",
]
OP_IX = {o: i for i, o in enumerate(OPS)}
N_OPS = len(OPS)                                # 18
CROP_IX = {c: i for i, c in enumerate(CROPS)}
N_CROPS = len(CROPS)                            # 5   (argument head for PLANT)
ITEM_IX = {c: i for i, c in enumerate(ITEMS)}
N_ITEMS = len(ITEMS)                            # 12  (argument head for PICKUP / PLACE)

MOVE_OPS = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

# market heads
MKT_TYPES = ["END", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"]
MKT_IX = {t: i for i, t in enumerate(MKT_TYPES)}
N_MKT = len(MKT_TYPES)                          # 7
QTY_BUCKETS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128]
N_QTY = len(QTY_BUCKETS)                        # 18

# --- concrete market orders, as a per-turn MULTI-LABEL problem ---------------
# The autoregressive slot formulation (predict a type per slot, terminated by an END token)
# was tried first and collapses: END is 56% of slot rows and BUY_ANIMAL is 0.7%, so an
# argmax head never once fires the order that starts the farm. Measured: the resulting
# clone banked 2,181 because it hired hands and then never bought a seed or an animal.
# One sigmoid per concrete order type gives every rare-but-decisive order its own decision
# boundary, and HIRE (emitted several times in one turn) gets a count head instead.
BUYABLE_PRODUCTS = ["WHEAT", "FERTILIZER"]
ORDER_TYPES = (
    [("BUY_SEED", c) for c in CROPS]
    + [("BUY_PRODUCT", c) for c in BUYABLE_PRODUCTS]
    + [("BUY_ANIMAL", a) for a in ANIMALS]
    + [("SELL", p) for p in PRODUCTS]
    + [("BUY_LAND", None)]
)
ORDER_IX = {t: i for i, t in enumerate(ORDER_TYPES)}
N_ORDERS = len(ORDER_TYPES)                     # 20
MAX_HIRES = 13                                  # count head: 0..12 hires in one turn
MAX_MARKET_ORDERS = 10

# ---------------------------------------------------------------------------
# task field
# ---------------------------------------------------------------------------
TASKS = [
    "WATER_NEEDED",      # plant not watered today
    "HARVEST_PLANT",     # plant with harvestable units
    "FERTILIZE_TARGET",  # plant not currently fertilized
    "EMPTY_PLANTABLE",   # unlocked empty ground
    "WEED",
    "FEED_NEEDED",       # animal not fed today
    "CARE_NEEDED",       # animal fed but not cared
    "FERT_AVAIL",        # animal with fertilizer waiting
    "HARVEST_ANIMAL",    # animal with product waiting
    "EMPTY_STRUCTURE",   # coop/pasture with no animal
    "SHED",              # the four centre access tiles
]
N_TASKS = len(TASKS)
TASK_IX = {t: i for i, t in enumerate(TASKS)}

_NEIGH = ((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0))  # stay, N, S, E, W  (dx, dy)


def _q_of(x: int, y: int) -> str:
    return ("NW" if x < 5 else "NE") if y < 5 else ("SW" if x < 5 else "SE")


def qty_bucket(n: int) -> int:
    """Index of the closest quantity bucket at or below n (never below bucket 0)."""
    best = 0
    for i, b in enumerate(QTY_BUCKETS):
        if b <= n:
            best = i
        else:
            break
    return best


class TurnContext:
    """Everything derived once per observation and shared by all of that turn's decisions."""

    __slots__ = ("obs", "me", "opp", "player", "day", "hour", "step", "tiles", "money",
                 "opp_money", "shed", "seeds", "inventories", "prices", "inv", "hands",
                 "farmer", "unlocked", "hires_today", "n_units", "task_xy", "_global",
                 "n_shops", "unit_pos")

    def __init__(self, obs):
        self.obs = obs
        self.player = int(obs["player"])
        self.day = int(obs["day"])
        self.hour = int(obs["hour"])
        self.step = int(obs.get("step", self.day * TURNS_PER_DAY + self.hour))
        farms = obs["farms"]
        self.me = farms[self.player]
        self.opp = farms[1 - self.player]
        self.tiles = self.me["tiles"]
        self.money = float(self.me["money"])
        self.opp_money = float(self.opp["money"])
        priv = obs.get("private") or {}
        self.shed = priv.get("shed") or {}
        self.seeds = priv.get("seeds") or {}
        self.inventories = priv.get("inventories") or [{}]
        mk = obs.get("market") or {}
        self.prices = mk.get("prices") or {}
        self.inv = mk.get("inventory") or {}
        self.farmer = list(self.me["farmer"])
        self.hands = [list(h) for h in (self.me.get("hands") or [])]
        self.unit_pos = [self.farmer] + self.hands
        self.n_units = len(self.unit_pos)
        self.unlocked = set(self.me.get("unlocked_quadrants") or ["NW"])
        self.hires_today = int(self.me.get("hires_today") or 0)
        self.n_shops = len((obs.get("town") or {}).get("unlocked_shops") or [])
        self.task_xy = self._task_field()
        self._global = self._global_features()

    # -- task field ---------------------------------------------------------
    def _task_field(self):
        buckets = [[] for _ in range(N_TASKS)]
        day = self.day
        for y in range(BOARD):
            row = self.tiles[y]
            for x in range(BOARD):
                t = row[x]
                if t == "LOCKED":
                    continue
                if t is None:
                    if _q_of(x, y) in self.unlocked:
                        buckets[TASK_IX["EMPTY_PLANTABLE"]].append((x, y))
                    continue
                kind = t.get("kind")
                if kind == "PLANT":
                    if not t.get("watered_today"):
                        buckets[TASK_IX["WATER_NEEDED"]].append((x, y))
                    if (t.get("yield_units") or 0) > 0:
                        buckets[TASK_IX["HARVEST_PLANT"]].append((x, y))
                    if int(t.get("fertilized_until_day", -1)) < day:
                        buckets[TASK_IX["FERTILIZE_TARGET"]].append((x, y))
                elif kind == "WEED":
                    buckets[TASK_IX["WEED"]].append((x, y))
                elif kind in ("COOP", "PASTURE"):
                    if t.get("animal"):
                        if not t.get("fed_today"):
                            buckets[TASK_IX["FEED_NEEDED"]].append((x, y))
                        elif not t.get("cared_today"):
                            buckets[TASK_IX["CARE_NEEDED"]].append((x, y))
                        if t.get("fertilizer_available"):
                            buckets[TASK_IX["FERT_AVAIL"]].append((x, y))
                        if (t.get("yield_units") or 0) > 0:
                            buckets[TASK_IX["HARVEST_ANIMAL"]].append((x, y))
                    else:
                        buckets[TASK_IX["EMPTY_STRUCTURE"]].append((x, y))
        buckets[TASK_IX["SHED"]] = list(SHED_TILES)
        return [np.asarray(b, dtype=np.int16).reshape(-1, 2) for b in buckets]

    # -- global block -------------------------------------------------------
    def _global_features(self):
        f = []
        f.append(self.day / (DAYS - 1))
        f.append(self.hour / (TURNS_PER_DAY - 1))
        f.append(math.sin(2 * math.pi * self.hour / TURNS_PER_DAY))
        f.append(math.cos(2 * math.pi * self.hour / TURNS_PER_DAY))
        left = max(0, LAST_ACTING_STEP - self.step)
        f.append(left / LAST_ACTING_STEP)
        f.append(1.0 if self.day >= DAYS - 1 else 0.0)              # last day: liquidation
        f.append(1.0 if left <= TURNS_PER_DAY else 0.0)
        f.append(1.0 if self.hour >= TURNS_PER_DAY - 2 else 0.0)    # end-of-day drop window
        f.append(math.log1p(max(0.0, self.money)) / 13.0)
        f.append(math.log1p(max(0.0, self.opp_money)) / 13.0)
        f.append(self.money / (self.money + self.opp_money + 1.0))
        for p in PRODUCTS:
            f.append(math.log1p(float(self.prices.get(p, 0))) / 7.0)
            f.append(math.log1p(float(self.inv.get(p, 0))) / 10.0)
        for it in ITEMS:
            f.append(math.log1p(float(self.shed.get(it, 0))) / 5.0)
        for c in CROPS:
            f.append(math.log1p(float(self.seeds.get(c, 0))) / 4.0)
        f.append(sum(float(v) for v in self.shed.values()) / 100.0)  # shed cap pressure
        f.append(self.n_units / 13.0)
        f.append(self.hires_today / 13.0)
        for q in QUADRANTS:
            f.append(1.0 if q in self.unlocked else 0.0)
        f.append(self.n_shops / 8.0)
        # cheap opponent model: what the other farm is made of
        op = self.opp["tiles"]
        n_plant = n_anim = n_weed = 0
        for y in range(BOARD):
            for x in range(BOARD):
                t = op[y][x]
                if isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        n_plant += 1
                    elif k == "WEED":
                        n_weed += 1
                    elif t.get("animal"):
                        n_anim += 1
        f.append(n_plant / 100.0)
        f.append(n_anim / 100.0)
        f.append(n_weed / 100.0)
        f.append(len(self.opp.get("hands") or []) / 13.0)
        return np.asarray(f, dtype=np.float32)

    # -- per-unit block -----------------------------------------------------
    def tile_at(self, x, y):
        return self.tiles[y][x]

    def unit_features(self, u: int, claimed=()):
        x, y = self.unit_pos[u]
        t = self.tiles[y][x]
        f = [1.0 if u == 0 else 0.0, u / 13.0, x / 9.0, y / 9.0]
        f.append(1.0 if _q_of(x, y) in self.unlocked else 0.0)
        f.append(1.0 if (x, y) in SHED_TILES else 0.0)

        kinds = [0.0] * 8       # empty, locked, weed, plant, coop-, pasture-, coop+, pasture+
        if t is None:
            kinds[0] = 1.0
        elif t == "LOCKED":
            kinds[1] = 1.0
        else:
            k = t.get("kind")
            if k == "WEED":
                kinds[2] = 1.0
            elif k == "PLANT":
                kinds[3] = 1.0
            elif k == "COOP":
                kinds[6 if t.get("animal") else 4] = 1.0
            elif k == "PASTURE":
                kinds[7 if t.get("animal") else 5] = 1.0
        f += kinds

        plant = [0.0] * (N_CROPS + 6)
        if isinstance(t, dict) and t.get("kind") == "PLANT":
            ci = CROP_IX.get(t.get("crop"))
            if ci is not None:
                plant[ci] = 1.0
            plant[N_CROPS + 0] = min(1.0, (self.day - int(t.get("planted_day", self.day))) / 12.0)
            plant[N_CROPS + 1] = 1.0 if t.get("watered_today") else 0.0
            plant[N_CROPS + 2] = min(1.0, float(t.get("consecutive_unwatered", 0)) / 2.0)
            plant[N_CROPS + 3] = min(1.0, float(t.get("yield_units", 0)) / 6.0)
            plant[N_CROPS + 4] = 1.0 if int(t.get("fertilized_until_day", -1)) >= self.day else 0.0
            plant[N_CROPS + 5] = 1.0 if self.step >= int(t.get("max_lifespan_step", 10 ** 9)) else 0.0
        f += plant

        anim = [0.0] * (len(ANIMALS) + 6)
        if isinstance(t, dict) and t.get("animal"):
            ai = ANIMALS.index(t["animal"]) if t["animal"] in ANIMALS else None
            if ai is not None:
                anim[ai] = 1.0
            n = len(ANIMALS)
            anim[n + 0] = 1.0 if t.get("fed_today") else 0.0
            anim[n + 1] = 1.0 if t.get("cared_today") else 0.0
            anim[n + 2] = min(1.0, float(t.get("yield_units", 0)) / 6.0)
            anim[n + 3] = 1.0 if t.get("fertilizer_available") else 0.0
            anim[n + 4] = min(1.0, float(t.get("consecutive_unfed", 0)) / 2.0)
            anim[n + 5] = min(1.0, float(t.get("pending_care_bonus", 0)) / 6.0)
        f += anim

        carried = self.inventories[u] if u < len(self.inventories) else {}
        for it in ITEMS:
            f.append(math.log1p(float(carried.get(it, 0))) / 3.0)
        f.append(min(1.0, sum(float(v) for v in carried.values()) / 20.0))

        near1 = near2 = 0
        for v, (ox, oy) in enumerate(self.unit_pos):
            if v == u:
                continue
            d = abs(ox - x) + abs(oy - y)
            if d <= 1:
                near1 += 1
            if d <= 2:
                near2 += 1
        f.append(min(1.0, near1 / 3.0))
        f.append(min(1.0, near2 / 5.0))

        f += self._task_block(x, y, claimed)
        return np.concatenate([self._global, np.asarray(f, dtype=np.float32)])

    def _task_block(self, x, y, claimed):
        """For every task type: 1/(1+distance) from here and from each of the four steps,
        plus an exists bit. This is what makes movement learnable."""
        out = []
        for ti in range(N_TASKS):
            xy = self.task_xy[ti]
            if claimed and len(xy):
                keep = [i for i in range(len(xy)) if (int(xy[i, 0]), int(xy[i, 1])) not in claimed]
                xy = xy[keep] if len(keep) != len(xy) else xy
            if not len(xy):
                out += [0.0] * 5 + [0.0]
                continue
            tx = xy[:, 0].astype(np.int32)
            ty = xy[:, 1].astype(np.int32)
            for dx, dy in _NEIGH:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < BOARD and 0 <= ny < BOARD):
                    out.append(0.0)
                    continue
                d = int(np.min(np.abs(tx - nx) + np.abs(ty - ny)))
                out.append(1.0 / (1.0 + d))
            out.append(1.0)
        return out

    # -- legality -----------------------------------------------------------
    def unit_mask(self, u: int):
        """Sound-but-not-tight legality mask over the op head. Never masks out something a
        strong teacher actually did (asserted during extraction)."""
        x, y = self.unit_pos[u]
        t = self.tiles[y][x]
        on_playable = (t != "LOCKED") and (_q_of(x, y) in self.unlocked)
        carried = self.inventories[u] if u < len(self.inventories) else {}
        m = np.zeros(N_OPS, dtype=np.float32)
        m[OP_IX["PASS"]] = 1.0
        if y > 0:
            m[OP_IX["NORTH"]] = 1.0
        if y < BOARD - 1:
            m[OP_IX["SOUTH"]] = 1.0
        if x < BOARD - 1:
            m[OP_IX["EAST"]] = 1.0
        if x > 0:
            m[OP_IX["WEST"]] = 1.0
        is_plant = isinstance(t, dict) and t.get("kind") == "PLANT"
        is_weed = isinstance(t, dict) and t.get("kind") == "WEED"
        is_struct = isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE")
        has_animal = is_struct and bool(t.get("animal"))
        if on_playable:
            if is_plant:
                m[OP_IX["WATER"]] = 1.0
                if (t.get("yield_units") or 0) > 0:
                    m[OP_IX["HARVEST"]] = 1.0
                if carried.get("FERTILIZER", 0) > 0:
                    m[OP_IX["FERTILIZE"]] = 1.0
            if is_plant or is_weed or (is_struct and not has_animal):
                m[OP_IX["DIG"]] = 1.0
            if t is None:
                m[OP_IX["BUILD_COOP"]] = 1.0
                m[OP_IX["BUILD_PASTURE"]] = 1.0
                if any(self.seeds.get(c, 0) > 0 for c in CROPS):
                    m[OP_IX["PLANT"]] = 1.0
            if has_animal:
                m[OP_IX["CARE"]] = 1.0
                if carried.get("WHEAT", 0) > 0:
                    m[OP_IX["FEED"]] = 1.0
                if t.get("fertilizer_available"):
                    m[OP_IX["COLLECT_FERTILIZER"]] = 1.0
                if (t.get("yield_units") or 0) > 0:
                    m[OP_IX["HARVEST"]] = 1.0
            if is_struct and not has_animal and any(carried.get(a, 0) > 0 for a in ANIMALS):
                m[OP_IX["PLACE"]] = 1.0
        if (x, y) in SHED_TILES:
            if any(v > 0 for v in carried.values()):
                m[OP_IX["DROP"]] = 1.0
                m[OP_IX["PLACE"]] = 1.0
            if any(v > 0 for v in self.shed.values()):
                m[OP_IX["PICKUP"]] = 1.0
        return m

    def crop_mask(self):
        m = np.zeros(N_CROPS, dtype=np.float32)
        for i, c in enumerate(CROPS):
            if self.seeds.get(c, 0) > 0:
                m[i] = 1.0
        return m if m.any() else np.ones(N_CROPS, dtype=np.float32)

    def item_mask_pickup(self):
        m = np.zeros(N_ITEMS, dtype=np.float32)
        for i, it in enumerate(ITEMS):
            if self.shed.get(it, 0) > 0:
                m[i] = 1.0
        return m if m.any() else np.ones(N_ITEMS, dtype=np.float32)

    def item_mask_place(self, u):
        carried = self.inventories[u] if u < len(self.inventories) else {}
        m = np.zeros(N_ITEMS, dtype=np.float32)
        for i, it in enumerate(ITEMS):
            if carried.get(it, 0) > 0:
                m[i] = 1.0
        return m if m.any() else np.ones(N_ITEMS, dtype=np.float32)

    # -- market -------------------------------------------------------------
    def market_features(self, slot: int, emitted):
        """Autoregressive market-slot features: global state, which slot we are filling, and
        what has already been queued this turn (type counts + items already sold/bought).
        Exact remaining bank is not derivable from the observation -- seed/animal/land prices
        are not in `obs` -- so the model gets the order history instead of a spend figure."""
        f = [slot / 10.0, 1.0 if slot == 0 else 0.0, len(emitted) / 10.0]
        counts = [0.0] * N_MKT
        sold = [0.0] * N_ITEMS
        for tname, item, n in emitted:
            counts[MKT_IX[tname]] += 1.0
            if item in ITEM_IX:
                sold[ITEM_IX[item]] += 1.0
        f += [c / 10.0 for c in counts]
        f += sold
        f.append(min(1.0, self.hires_today / 13.0))
        return np.concatenate([self._global, np.asarray(f, dtype=np.float32)])

    def market_mask(self):
        left = self.money
        m = np.ones(N_MKT, dtype=np.float32)
        if left < 1:
            for t in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND"):
                m[MKT_IX[t]] = 0.0
        if not any(v > 0 for v in self.shed.values()):
            m[MKT_IX["SELL"]] = 0.0
        if len(self.unlocked) >= 4:
            m[MKT_IX["BUY_LAND"]] = 0.0
        return m

    # -- market, per-turn multi-label form ----------------------------------
    def market_turn_features(self):
        f = [math.log1p(max(0.0, self.money)) / 13.0,
             self.hires_today / 13.0,
             self.n_units / 13.0,
             min(1.0, sum(float(v) for v in self.shed.values()) / 100.0),
             1.0 if self.money >= 1000 else 0.0,
             1.0 if self.money >= 5000 else 0.0]
        for it in ITEMS:
            f.append(1.0 if self.shed.get(it, 0) else 0.0)
        return np.concatenate([self._global, np.asarray(f, dtype=np.float32)])

    def order_mask(self):
        """Never emit an order that cannot possibly do anything."""
        m = np.ones(N_ORDERS, dtype=np.float32)
        for i, (t, item) in enumerate(ORDER_TYPES):
            if t == "SELL" and not self.shed.get(item, 0):
                m[i] = 0.0
            elif t == "BUY_LAND" and len(self.unlocked) >= 4:
                m[i] = 0.0
            elif t.startswith("BUY") and self.money < 1:
                m[i] = 0.0
        return m


def unit_dim() -> int:
    """Length of a unit feature vector (computed once from a synthetic minimal obs)."""
    return _PROBE["unit"]


def market_dim() -> int:
    return _PROBE["market"]


def _probe():
    tiles = [[None] * BOARD for _ in range(BOARD)]
    obs = {
        "player": 0, "day": 0, "hour": 0, "step": 0,
        "farms": [
            {"money": 3000, "tiles": tiles, "farmer": [0, 0], "hands": [],
             "unlocked_quadrants": ["NW"], "hires_today": 0},
            {"money": 3000, "tiles": [[None] * BOARD for _ in range(BOARD)], "farmer": [0, 0],
             "hands": [], "unlocked_quadrants": ["NW"], "hires_today": 0},
        ],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }
    ctx = TurnContext(obs)
    return {"unit": int(ctx.unit_features(0).shape[0]),
            "market": int(ctx.market_features(0, []).shape[0]),
            "market_turn": int(ctx.market_turn_features().shape[0])}


_PROBE = _probe()
UNIT_DIM = _PROBE["unit"]
MARKET_DIM = _PROBE["market"]
MARKET_TURN_DIM = _PROBE["market_turn"]
