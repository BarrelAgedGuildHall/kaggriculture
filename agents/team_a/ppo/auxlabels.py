"""Rollout-computed auxiliary labels for `exp-014`'s factored critic.

Named `auxlabels` and not `aux`: `AUX` is a reserved DOS device name, and on Windows git
refuses to open `aux.py` at all ("No such file or directory" on a file that plainly exists).
The trainer runs on Linux but the dev clone is Windows, so the file could be written, imported
and tested and still never be committed.

Ratified by the 2026-08-22 halt-ruling addendum-01 A-4.1: **the training labels are computed on
our own rollouts, and Ops' S-2 shard is the out-of-distribution calibration set.** The reason is
not preference. `warehouse/datasets/s2-labels/` carries labels and context but **no feature
vectors**, so training a head on it would require re-featurising 24 ladder replays we would then
still be fitting off-distribution; the same labels computed in-process are *exact* (both seats'
commits are observed) and *distribution-matched* (the aux head is fitted on the very states the
critic sees, which is the only version that shapes the trunk the critic uses).

**The definitions here are Ops', not new ones.** Every column below is the live-episode twin of a
column in `scripts/extract_s2_labels.py` (branch `feature/ops-s2-labels`) with the same name and
the same semantics, which is precisely what makes the S-2 shard a valid external calibration set
rather than a differently-defined table that happens to have nine columns:

    opp_net[t]   == `p{1-seat}_future_net_true[t]`   (9)  units the RIVAL will still push into
                                                          each pool between t and the horizon
    headroom[t]  == `headroom_actual[t]`             (9)  units the pool can still absorb above
                                                          the $1 floor before the horizon
    nav[t]                                           (4)  the exp-N1 NAV mark block -- NOT an S-2
                                                          column; see `set_nav_fn` below

Two engine facts the labels rest on, both verified in S-2's own validation (0 residual cells over
155,304):

1. Only `_commit_unit` and `_town_consume` write `market["inventory"]`, so
   `inv[t+1] - inv[t] == sum_players (sold_above_floor - bought) - drain(t)` is exact. We get the
   first term from `lines.Accountant` (which already patches `_commit_unit`, so this module adds
   **no second monkeypatch** -- it rides the accountant's `on_step` hook) and the second from a
   pure function of the unlocked-shop set and the step counter.
2. A SELL at the $1 floor pays the seller but adds **no** market supply. `net_true` counts it,
   `net_visible` does not. `net_true` is the supervised target: the critic is discarded at
   deployment, so it is allowed to be told things no runtime estimator could infer (A-4.2).

**The privileged-information boundary (A-4.2), which this module sits directly on top of.**
Privileged *labels* are admissible in the critic and its auxiliary heads without limit. Privileged
*inputs to the actor* are prohibited outright. Nothing in this file is ever read on the actor's
forward path -- the labels are produced after the fact from engine state and consumed only by
`critic.FactoredCritic`, whose aux heads are excluded from `value()` by construction. The red test
for the boundary is `tests/test_ppo_aux.py::test_aux_heads_are_off_the_deployed_path`.
"""
from __future__ import annotations

import numpy as np

from . import lines as L

# name -> width. `train.AUX_DIMS` defers to this so there is one declaration.
AUX_DIMS = {"opp_net": L.N_PRODUCTS, "headroom": L.N_PRODUCTS, "nav": 4}

NO_FLOOR = -1                 # S-2's sentinel: this pool's $1 floor is unreachable
# Below this target variance an R^2 is undefined, not bad. In label units (whole units traded).
R2_MIN_VAR = 1e-6
FLOOR_SEARCH_CAP = 100_000    # S-2's reachability cap, verbatim

# exp-N1's NAV mark block is S-1's deliverable (`feature/s1-nav-package`), not Team A's, and it is
# not on `main` yet. Rather than fork a second NAV implementation -- which would guarantee that
# the critic's "nav" head and the balance-sheet program's NAV drift apart -- the head is wired to a
# hook. `--critic-aux nav` fails loudly until S-1 lands and `set_nav_fn` is called; `opp_net` and
# `headroom` are fully live today and are the two heads exp-014 was specced on.
_NAV_FN = None


def set_nav_fn(fn):
    """Install `fn(state, seat) -> (4,) float` from S-1's `nav.py` once it merges."""
    global _NAV_FN
    _NAV_FN = fn


def available():
    """Aux head names that can actually be produced in this process."""
    return tuple(k for k in AUX_DIMS if k != "nav" or _NAV_FN is not None)


# --------------------------------------------------------------------------- town drain


def town_drain(K, cfg, shops, step, n_prod, p_index):
    """Units of each product the town removes in the transition out of `step`.

    Mirrors `_town_consume` exactly (shop interval 4, town-centre interval 24, single-product
    shops consume at 2x). Shops are drawn in `_end_of_day`, which runs AFTER `_town_consume`, so
    the PRE-step shop list is the one that fires -- which is why the snapshot is taken at the top
    of the interpreter call and not at the bottom.
    """
    out = np.zeros(n_prod, dtype=np.int64)
    shop_interval = max(1, int(cfg.get("townShopSellInterval", 4)))
    center_interval = max(1, int(cfg.get("townCenterSellInterval", 24)))
    if step % shop_interval == 0:
        for shop_name in shops:
            items = K.SHOPS.get(shop_name, ())
            mult = 2 if len(items) == 1 else 1
            for item in items:
                j = p_index.get(item)
                if j is not None:
                    out[j] += mult
    if step % center_interval == 0:
        for item in K.TOWN_CENTER_PRODUCTS:
            j = p_index.get(item)
            if j is not None:
                out[j] += 1
    return out


def floor_inventory(K, item, params, cap=FLOOR_SEARCH_CAP):
    """Smallest inventory at which `item` quotes the $1 floor, or `NO_FLOOR`.

    `market_price` is non-increasing in inventory, so the bisection is exact. Two of the nine
    pools (WHEAT, EGG) have no reachable floor under 1.32.7 -- WHEAT still quotes $15 and EGG $30
    with 100,000 units on the shelf -- and emitting a fictional floor for them would put a made-up
    number in a training label. They get the sentinel and the substitution in `_headroom`.
    """
    if K.market_price(item, cap, params) > K.PRICE_FLOOR:
        return NO_FLOOR
    lo, hi = 0, cap
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if K.market_price(item, mid, params) > K.PRICE_FLOOR:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------- recorder


class Recorder:
    """Per-step engine snapshots, taken through `lines.Accountant.on_step`.

    Cost is one small array copy per step (9 ints + a 2x3x9 tally); over a 720-turn episode that
    is under 100 kB and no measurable wall time, which is why it is always on when any aux head is
    requested rather than being sampled.
    """

    def __init__(self, K, cfg=None, want_nav=False):
        self.K = K
        self.cfg = dict(cfg or {})     # filled from `env.configuration` on the first hook call
        self.want_nav = bool(want_nav)
        self.p_index = {p: i for i, p in enumerate(L.PRODUCTS)}
        self.n_prod = L.N_PRODUCTS
        self.params = None
        self.floors = None
        self.steps = []            # step index, ascending, deduped
        self.inv = []              # (9,) market inventory BEFORE the transition
        self.cum = []              # (2, 3, 9) cumulative flow BEFORE the transition
        self.drain = []            # (9,) town drain OUT OF this step
        self.nav = []              # (2, 4) NAV marks, or empty when no nav fn
        self._seen = set()

    # -- the hook -----------------------------------------------------------
    def __call__(self, state, env, acc):
        obs = state[0]["observation"]
        market = obs.get("market") or {}
        if not market or len(obs.get("farms") or ()) != 2:
            return          # the `env.reset` call: no farms, no market, not a transition
        if self.params is None:
            self.params = market.get("params")
            # The drain intervals live in the env configuration, not the observation, and the env
            # only exists once `play` has made it -- so the config is taken here rather than at
            # construction. `dict(...)` because `env.configuration` is a Struct.
            for k, v in dict(getattr(env, "configuration", None) or {}).items():
                self.cfg.setdefault(k, v)
            self.floors = np.array([floor_inventory(self.K, p, self.params)
                                    for p in L.PRODUCTS], dtype=np.int64)
        turns_per_day = max(1, int(self.cfg.get("turnsPerDay", 24)))
        step = int(obs.get("step", int(obs["day"]) * turns_per_day + int(obs["hour"])))
        if step in self._seen:
            # `env.reset` runs the interpreter once to build the initial state, so step 0 can be
            # visited twice with identical content. Keep the first; a duplicate must never shift
            # the suffix sums by a whole transition.
            return
        self._seen.add(step)
        inventory = market.get("inventory") or {}
        shops = (obs.get("town") or {}).get("unlocked_shops") or []
        self.steps.append(step)
        self.inv.append(np.array([int(inventory.get(p, 0)) for p in L.PRODUCTS], dtype=np.int64))
        self.cum.append(acc.prod.copy())
        self.drain.append(town_drain(self.K, self.cfg, shops, step, self.n_prod, self.p_index))
        if self.want_nav and _NAV_FN is not None:
            self.nav.append(np.stack([np.asarray(_NAV_FN(state, s), dtype=np.float64)
                                      for s in (0, 1)]))

    # -- labels -------------------------------------------------------------
    def _future_net_true(self, acc):
        """(n_steps, 2, 9): units each seat will still SELL minus BUY from this step on."""
        cum = np.stack(self.cum)                       # (n, 2, 3, 9) cumulative BEFORE step
        net_cum = cum[:, :, L.FLOWS.index("sold"), :] - cum[:, :, L.FLOWS.index("bought"), :]
        total = (acc.prod[:, L.FLOWS.index("sold"), :]
                 - acc.prod[:, L.FLOWS.index("bought"), :])
        return total[None, :, :] - net_cum

    def _headroom(self):
        """(n_steps, 9): S-2's `headroom_actual`, with the README's sentinel substitution."""
        inv = np.stack(self.inv)                                   # (n, 9)
        drain = np.stack(self.drain)                               # (n, 9)
        remaining = np.cumsum(drain[::-1], axis=0)[::-1]           # suffix sum, inclusive
        static = self.floors[None, :] - inv
        out = (static + remaining).astype(np.float64)
        # WHEAT and EGG are bottomless: `headroom_actual` is undefined and S-2 emits -1. The
        # README names `drain_remaining_actual` as the usable substitute, so the column carries a
        # real quantity for all nine products instead of a masked constant the head must ignore.
        bottomless = self.floors == NO_FLOOR
        out[:, bottomless] = remaining[:, bottomless]
        return out

    def labels(self, acc, seat, names, steps_wanted):
        """{name: (len(steps_wanted), dim)} float32, one row per recorded market turn."""
        if not names:
            return {}
        order = np.argsort(np.asarray(self.steps))
        step_row = {int(np.asarray(self.steps)[i]): k for k, i in enumerate(order)}
        # A recorded turn whose step never reached the interpreter cannot be labelled; clamping
        # to the last snapshot would silently give it a different episode's suffix sum.
        rows = np.array([step_row.get(int(s), -1) for s in steps_wanted], dtype=np.int64)
        if (rows < 0).any():
            missing = int((rows < 0).sum())
            raise RuntimeError(f"{missing} recorded turns have no engine snapshot -- "
                               "the accountant was attached after the episode started")
        out = {}
        if "opp_net" in names:
            fut = self._future_net_true(acc)[order]                 # (n, 2, 9)
            out["opp_net"] = fut[rows, 1 - int(seat), :].astype(np.float32)
        if "headroom" in names:
            out["headroom"] = self._headroom()[order][rows].astype(np.float32)
        if "nav" in names:
            if not self.nav:
                raise RuntimeError("aux head 'nav' requested but no NAV implementation is "
                                   "installed -- call aux.set_nav_fn() once S-1's nav.py lands "
                                   "(branch feature/s1-nav-package)")
            out["nav"] = np.stack(self.nav)[order][rows, int(seat), :].astype(np.float32)
        return out


# --------------------------------------------------------------------------- S-2 calibration


def s2_calibration(critic, x, s2_npz, seat_col="p1", names=("opp_net", "headroom")):
    """A-4.4: report the aux heads' fit on Ops' out-of-distribution S-2 shard. Gates nothing.

    `x` must be market-turn features for the S-2 rows in the shard's own row order -- which the
    shard does NOT carry (it has labels and context, no feature vectors; that is exactly why
    training uses rollout labels). Producing `x` means re-featurising the 24 ladder replays, an
    Ops/Engineer step. This function is the consumer side of that hook: given the features, it
    scores the trained heads against the external labels and returns per-head R^2.

    Pre-registered interpretation (A-4.4): large disagreement between this and the rollout
    held-out fit means the head is fitting self-play artefacts and its critic contribution is
    in-family only. That is **a caveat on the row, not a blocker** -- exp-014's variable is the
    critic, and an aux head's external validity is not on its critical path.
    """
    z = np.load(str(s2_npz), allow_pickle=True) if not hasattr(s2_npz, "files") else s2_npz
    products = [str(p) for p in z["products"]]
    if products != list(L.PRODUCTS):
        raise ValueError(f"S-2 product order {products} != {list(L.PRODUCTS)}")
    truth = {}
    if "opp_net" in names:
        truth["opp_net"] = z[f"{seat_col}_future_net_true"].astype(np.float64)
    if "headroom" in names:
        h = z["headroom_actual"].astype(np.float64)
        rem = z["drain_remaining_actual"].astype(np.float64)
        truth["headroom"] = np.where(z["inv_floor"] == NO_FLOOR, rem, h)
    xn = (np.asarray(x, dtype=np.float32) - critic.mu) / critic.sd
    out = critic.net.forward(xn)
    report = {}
    for k, t in truth.items():
        if k not in critic.aux_dims:
            continue
        pred = out[k] * critic.aux_sd[k] + critic.aux_mu[k]
        resid = np.var(t - pred, axis=0)
        var = np.var(t, axis=0)
        # Same degeneracy as `critic.explained_var`, and it bites hardest exactly here: over the
        # 24 ladder episodes the field sold 24 units of TOMATO and ZERO of EGG, so those columns
        # have no variance to explain and 1 - Var(resid)/~0 reads -37 and -40. Reporting a number
        # there would put "the head is catastrophically wrong about EGG" on exp-014's row when
        # what happened is that nobody traded EGG. `None` and a named list instead.
        per = [None if v < R2_MIN_VAR else round(float(1.0 - r / v), 4)
               for r, v in zip(resid, var)]
        vt = float(np.var(t.sum(1)))
        report[k] = {"r2_per_product": per,
                     "degenerate_products": [products[j] for j, v in enumerate(per) if v is None],
                     # `None` only catches variance that is exactly zero; a product with tiny but
                     # nonzero variance still yields a wild R^2, and no threshold can be chosen
                     # here without inventing a knob. Reporting the target SD beside the R^2 makes
                     # such a cell readable instead of alarming -- STRAWBERRY at SD 200 and TOMATO
                     # at SD 0.4 are not the same claim even at the same R^2.
                     "target_sd_per_product": [round(float(s), 3) for s in np.sqrt(var)],
                     "r2_pooled": (None if vt < R2_MIN_VAR
                                   else round(float(1.0 - np.var(t.sum(1) - pred.sum(1)) / vt), 4)),
                     "n_rows": int(len(t))}
    report["source"] = str(getattr(s2_npz, "fid", s2_npz))
    return report
