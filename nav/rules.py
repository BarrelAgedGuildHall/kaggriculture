"""The engine's RULEBOOK, inlined -- and nothing else.

`nav` is vendored into a Kaggle submission, where `kaggle_environments` does not exist, so
the tables the marks need (which crop yields when, which animal eats what, which shop drains
which product) have to live here. That is the *only* reason anything is hard-coded in this
package.

The line this file draws, and why it is the whole doctrine of S-1:

* **Rulebook structure is inlined.** "A tomato first yields on day 8, every day, four times"
  is a rule of the game. It cannot be inferred from one observation and it is not a price.
* **Prices are never inlined.** No base, no amplitude, no curve shape, no per-product
  constant tuned to make a number come out right. The price curve is fitted at runtime from
  observed `(inventory, price)` pairs (`nav.curves`), which is what lets the sheet survive an
  engine rebalance. `MARKET_I0_FALLBACK` below is a fallback for a trace that does not start
  at step 0; in a real episode `base` and `I0` are *observed exactly* on turn 0.

`tests/test_nav.py::test_rules_match_the_engine` compares every table here against the
installed engine and fails loudly when a release changes one. That is deliberate: an engine
bump must break a test, not silently re-price the whole balance sheet.

Source: kaggle-environments 1.32.7, `kaggle_environments/envs/kaggriculture/kaggriculture.py`
(sha256 bc8a5487..., see docs/engine/VERSION.md).
"""
from __future__ import annotations

# --- the engine version these tables were transcribed from --------------------------------
ENGINE_VERSION = "1.32.7"

# --- goods ---------------------------------------------------------------------------------
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")

CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

# --- the town -------------------------------------------------------------------------------
SHOPS = {
    "BAKERY":         ("EGG", "WHEAT"),
    "PIZZA_SHOP":     ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT":    ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE":     ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE":       ("CARROT",),
    "SMOOTHIE_SHOP":  ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
TOWN_CENTER_PRODUCTS = tuple(p for p in PRODUCTS if p != "FERTILIZER")
MAX_SHOP_INSTANCES = 8          # shops are drawn WITH replacement: caps count, not variety
SHOP_SELL_INTERVAL = 4          # steps; configuration.townShopSellInterval
CENTER_SELL_INTERVAL = 24       # steps; configuration.townCenterSellInterval

# --- the clock ------------------------------------------------------------------------------
TURNS_PER_DAY = 24
DAYS = 30
LAST_DAY = DAYS - 1             # day 29: its end-of-day tick never runs
EPISODE_STEPS = TURNS_PER_DAY * DAYS        # 720 snapshots, indices 0..719
TERMINAL_INDEX = EPISODE_STEPS - 1          # 719: the first index at which no action remains
# exp-N1 §1(a): a SELL still executes on step 718, so 718 is NOT terminal. H(i) = 719 - i.

# --- the board ------------------------------------------------------------------------------
BOARD_SIZE = 10
SHED_CAPACITY = 100             # configuration.shedCapacity; the covenant of plan §1 row 2
LAND_PRICES = (1000, 2000, 4000)
MAX_MARKET_ORDERS_PER_TURN = 10

# --- market mechanics that are structure, not price -----------------------------------------
PRICE_FLOOR = 1                 # `max(PRICE_FLOOR, int(round(price)))`; prices are integers
MARKET_I0_FALLBACK = 10000      # only used when step 0 was never observed (offline splices)
SHAPES = ("linear", "sq", "sqrt", "log", "hinge")
HINGE_GAIN = 8.0                # `u + 8*max(0, u-1)**2`, u = x/T


def shed_access_tiles(board_size: int = BOARD_SIZE) -> tuple[tuple[int, int], ...]:
    """The four inner-corner tiles around the shed, in NWSE order."""
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def hours_left(step: int) -> int:
    """Turns in which an action can still be taken, from the snapshot at `step`."""
    return max(0, TERMINAL_INDEX - step)


def ticks_left(day: int) -> int:
    """Remaining end-of-day ticks. Day 29's never runs, so it is 0 from day 29's morning."""
    return max(0, LAST_DAY - day)
