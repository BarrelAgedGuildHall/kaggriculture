"""Inference-time behaviour-cloned policy (numpy only).

Same feature path as training (`features.py`) and the same forward pass (`net.py`), so the
packaged agent and the evaluated policy are literally the same code.

Ensembling follows the Pokemon-challenge contract: each member normalises over the SAME
legal set, the probability vectors are averaged (not logits, not weights), and the argmax is
taken. It is deterministic and seat-agnostic. Argument heads and the autoregressive market
sequence are delegated to the primary member (index 0) so its ordering state stays coherent.

A sanitiser always runs last. It is not cosmetic: the engine drops EVERY `PLANT` op for a
crop whose requests in one turn exceed the seeds held, so an unsanitised policy can silently
lose its whole planting turn.
"""
from __future__ import annotations

import json
import os

import numpy as np

try:
    import features as F
    from net import MLP, masked_softmax, sigmoid
except ImportError:  # imported as a package
    from . import features as F
    from .net import MLP, masked_softmax, sigmoid

UNIT_HEADS = {"op": F.N_OPS, "crop": F.N_CROPS, "item": F.N_ITEMS, "qty": F.N_UQTY}
MKT_HEADS = {"orders": F.N_ORDERS, "qty": F.N_ORDERS * F.N_QTY, "hire": F.MAX_HIRES}


class Member:
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.path = path
        self.card = json.loads(str(d["card"][0]))
        # Thresholds calibrated on the ENSEMBLE's averaged probabilities, if they have been
        # computed (calibrate_ensemble.py). Per-member thresholds are tuned against a single
        # member's sharper outputs and mis-fire on the mixture.
        self.ens_thresholds = d["m_ens_thresholds"] if "m_ens_thresholds" in d.files else None
        self.unit = MLP.from_state({k[2:]: d[k] for k in d.files if k.startswith("u_")
                                    and k not in ("u_mu", "u_sd")}, UNIT_HEADS)
        self.mkt = MLP.from_state({k[2:]: d[k] for k in d.files if k.startswith("m_")
                                   and k not in ("m_mu", "m_sd")}, MKT_HEADS)
        self.umu, self.usd = d["u_mu"], d["u_sd"]
        self.mmu, self.msd = d["m_mu"], d["m_sd"]
        self.thresholds = d["m_thresholds"]
        self.slot_order = d["slot_order"]        # mean emission slot per order type (+HIRE)

    def unit_probs(self, x, mask):
        out = self.unit.forward(((x - self.umu) / self.usd)[None, :])
        return (masked_softmax(out["op"], mask[None, :])[0],
                masked_softmax(out["crop"])[0], masked_softmax(out["item"])[0],
                masked_softmax(out["qty"])[0])

    def mkt_probs(self, x):
        out = self.mkt.forward(((x - self.mmu) / self.msd)[None, :])
        return (sigmoid(out["orders"])[0],
                masked_softmax(out["qty"].reshape(F.N_ORDERS, F.N_QTY)),
                masked_softmax(out["hire"])[0])


class BCPolicy:
    """`members` = list of .npz paths. The first is the primary member."""

    LIQUIDATE_ORDERS = ("buys_first", "sells_first")

    def __init__(self, members, safety=3, liquidate_order="buys_first"):
        self.members = [Member(p) for p in members]
        self.safety = int(safety)
        # exp-022a. `buys_first` is the shipped behaviour and stays the default until the
        # exp-013b/exp-014 pair closes (integration-ruling I-3, the wrapper pin): every PPO
        # rollout must keep running on the environment exp-013b trained in.
        self.liquidate_order = str(liquidate_order)
        if self.liquidate_order not in self.LIQUIDATE_ORDERS:
            raise ValueError(f"liquidate_order must be one of {self.LIQUIDATE_ORDERS}, "
                             f"got {liquidate_order!r}")
        # The Pokemon ensemble is only well-defined over members that speak the same
        # distribution: jun177's LogProbabilityEnsemble refuses any member whose config or
        # tensor schema differs from member 0. Mixing shapes here would not raise -- it would
        # quietly average two different policies' probabilities -- so check it explicitly.
        reference = self.members[0]
        ref_schema = {k: v.shape for k, v in reference.unit.state().items()}
        ref_schema.update({"m_" + k: v.shape for k, v in reference.mkt.state().items()})
        for index, member in enumerate(self.members[1:], 1):
            schema = {k: v.shape for k, v in member.unit.state().items()}
            schema.update({"m_" + k: v.shape for k, v in member.mkt.state().items()})
            if schema != ref_schema:
                raise ValueError(f"ensemble member {index} tensor schema differs from member 0")

    # -- unit ops -----------------------------------------------------------
    def _unit_op(self, ctx, u, claimed):
        mask = ctx.unit_mask(u)
        x = ctx.unit_features(u, claimed)
        pops = pcrop = pitem = pqty = None
        for m in self.members:
            a, b, c, q = m.unit_probs(x, mask)
            pops = a if pops is None else pops + a
            pcrop = b if pcrop is None else pcrop + b
            pitem = c if pitem is None else pitem + c
            pqty = q if pqty is None else pqty + q
        oi = int(np.argmax(pops))
        name = F.OPS[oi]
        if name == "PLANT":
            cm = ctx.crop_mask()
            return ["PLANT", F.CROPS[int(np.argmax(pcrop * cm))]], True
        if name == "PICKUP":
            im = ctx.item_mask_pickup()
            item = F.ITEMS[int(np.argmax(pitem * im))]
            # The COUNT is cloned. Hardcoding "take the whole shed" (the first version of
            # this line) matched the teacher on 4.2% of PICKUPs and let unit 0 empty the
            # shed, which silently invalidated every later unit's plan in the same turn.
            n = F.QTY_BUCKETS[int(np.argmax(pqty))]
            return ["PICKUP", item, int(max(1, min(n, ctx.shed.get(item, 0))))], True
        if name == "PLACE":
            im = ctx.item_mask_place(u)
            return ["PLACE", F.ITEMS[int(np.argmax(pitem * im))]], True
        return [name], name not in ("PASS", "NORTH", "SOUTH", "EAST", "WEST")

    # -- market -------------------------------------------------------------
    def _market(self, ctx):
        """Per-turn multi-label emission. Each concrete order fires on its own tuned
        threshold; quantity comes from that order's own softmax; HIRE has a count head.
        Orders are emitted in the mean slot order the teacher used, because each order
        spends whatever the earlier ones left."""
        x = ctx.market_turn_features()
        mask = ctx.order_mask()
        pord = pqty = phire = None
        for m in self.members:
            a, b, c = m.mkt_probs(x)
            pord = a if pord is None else pord + a
            pqty = b if pqty is None else pqty + b
            phire = c if phire is None else phire + c
        k = len(self.members)
        pord, pqty, phire = pord / k, pqty / k, phire / k
        th = self.members[0].ens_thresholds
        if th is None:
            th = self.members[0].thresholds
        slots = self.members[0].slot_order

        fired = []
        for oi in range(F.N_ORDERS):
            if mask[oi] <= 0 or pord[oi] < th[oi]:
                continue
            tname, item = F.ORDER_TYPES[oi]
            n = F.QTY_BUCKETS[int(np.argmax(pqty[oi]))]
            if tname == "SELL":
                n = min(n, int(ctx.shed.get(item, 0)))
                if n <= 0:
                    continue
            order = ["BUY_LAND"] if tname == "BUY_LAND" else [tname, item, int(n)]
            fired.append((float(slots[oi]), order))

        hires = int(np.argmax(phire))
        hire_slot = float(slots[F.N_ORDERS]) if len(slots) > F.N_ORDERS else 99.0
        for i in range(hires):
            fired.append((hire_slot + i * 1e-3, ["HIRE"]))

        fired.sort(key=lambda t: t[0])
        return [o for _, o in fired[:F.MAX_MARKET_ORDERS]]

    @staticmethod
    def _item_mask(ctx, tname):
        m = np.zeros(F.N_ITEMS, dtype=np.float32)
        if tname == "BUY_SEED":
            for c in F.CROPS:
                m[F.ITEM_IX[c]] = 1.0
        elif tname == "BUY_PRODUCT":
            for c in ("WHEAT", "FERTILIZER"):
                m[F.ITEM_IX[c]] = 1.0
        elif tname == "BUY_ANIMAL":
            for c in F.ANIMALS:
                m[F.ITEM_IX[c]] = 1.0
        elif tname == "SELL":
            for c in F.ITEMS:
                if ctx.shed.get(c, 0) > 0:
                    m[F.ITEM_IX[c]] = 1.0
        return m

    # -- safety layer -------------------------------------------------------
    # level 0  network only (pure clone)
    # level 1  endgame only: drop carried goods and liquidate before the bell
    # level 2  + keep-alive invariants (feed / water / wheat supply)
    #
    # Why this exists: behaviour cloning has no recovery behaviour for states the teacher
    # never visited, and the usual remedy (DAgger -- roll out, ask the teacher what it
    # would have done) is unavailable because the teacher is another team's binary on the
    # Kaggle ladder. Measured in exp-003: one slightly-off opening let the animals reach
    # two unfed days, they escaped, production stopped, and the clone banked 478 against
    # v0's 41,048. The rules below own only the invariants where a single miss is
    # unrecoverable and the correct action is certain.

    def _apply_safety(self, ctx, ops, orders):
        if self.safety >= 1:
            ops, orders = self._endgame(ctx, ops, orders)
        if self.safety >= 2:
            ops, orders = self._keep_alive(ctx, ops, orders)
        if self.safety >= 3:
            orders = self._liquidate(ctx, orders)
        return ops, orders

    def _liquidate(self, ctx, orders):
        """Replace the learned SELL head with a rule.

        The learned head reaches F1 0.54-0.76 on the teacher's own states but fires far too
        rarely on the clone's, so the shed fills, the bank never grows, and the clone can
        neither hire, buy land nor restock wheat -- exp-004 ended 30 days with three
        quadrants still LOCKED. Selling is the money loop; a head that is merely
        *mostly* right about when to sell breaks it. WHEAT is held back to feed the
        animals; everything else goes as it arrives.

        `liquidate_order` (exp-022a) flips only the SEQUENCE of the returned list, never its
        contents: the slot budget is still `room = MAX - len(kept)`, so exactly the same
        orders are emitted either way. The engine commits market orders in list order
        (`_process_market`), so `sells_first` lets today's sale revenue fund today's
        BUY_LAND / BUY_ANIMAL instead of the buy failing `no_cash` first (exp-021 measured
        42.9% of BUY_LAND and 44.0% of COW orders failing entirely, all `no_cash`)."""
        n_animals = sum(1 for row in ctx.tiles for t in row
                        if isinstance(t, dict) and t.get("animal"))
        reserve = {"WHEAT": 3 * n_animals}
        kept = [o for o in orders if o[0] != "SELL"]
        sells = []
        for item in F.PRODUCTS:
            have = int(ctx.shed.get(item, 0)) - reserve.get(item, 0)
            if have > 0:
                sells.append((int(ctx.prices.get(item, 1)) * have, [item, have]))
        sells.sort(key=lambda t: -t[0])
        room = F.MAX_MARKET_ORDERS - len(kept)
        chosen = [["SELL", i, n] for _, (i, n) in sells[:max(0, room)]]
        if self.liquidate_order == "sells_first":
            return chosen + kept
        return kept + chosen

    def _endgame(self, ctx, ops, orders):
        """Day 29 has no end-of-day, so the free auto-drop of unit inventories into the
        shed never happens: anything still carried at the bell is worth zero. Unsold shed
        stock is worth zero too."""
        if ctx.day < F.DAYS - 1:
            return ops, orders
        for u in range(len(ops)):
            carried = ctx.inventories[u] if u < len(ctx.inventories) else {}
            if not any(v > 0 for v in carried.values()):
                continue
            x, y = ctx.unit_pos[u]
            if (x, y) in F.SHED_TILES:
                ops[u] = ["DROP"]
            elif ctx.hour >= F.TURNS_PER_DAY - 10:
                ops[u] = [_step_towards(x, y, 4 if x <= 4 else 5, 4 if y <= 4 else 5)]
        if ctx.hour >= F.TURNS_PER_DAY - 4:
            orders = [["SELL", k, int(v)] for k, v in ctx.shed.items()
                      if v and k in F.PRODUCTS][:F.MAX_MARKET_ORDERS]
        return ops, orders

    def _keep_alive(self, ctx, ops, orders):
        """Two consecutive missed refreshes kill a plant (weeds) or lose an animal
        (escapes, unrecoverably). `consecutive_unwatered`/`consecutive_unfed` >= 1 means one
        night has already been missed, so this is the last day to act. Left alone until
        midday so the cloned policy owns the ordinary schedule."""
        if ctx.hour < 11 or ctx.day >= F.DAYS - 1:
            return ops, orders

        animals, plants, n_animals = [], [], 0
        for y in range(F.BOARD):
            for x in range(F.BOARD):
                t = ctx.tiles[y][x]
                if not isinstance(t, dict):
                    continue
                if t.get("animal"):
                    n_animals += 1
                    if not t.get("fed_today") and int(t.get("consecutive_unfed", 0)) >= 1:
                        animals.append((x, y))
                elif t.get("kind") == "PLANT":
                    if not t.get("watered_today") and int(t.get("consecutive_unwatered", 0)) >= 1:
                        plants.append((x, y))

        # keep the wheat supply alive -- an animal cannot be fed out of an empty shed
        if n_animals:
            have = int(ctx.shed.get("WHEAT", 0)) + sum(
                int(inv.get("WHEAT", 0)) for inv in ctx.inventories)
            need = 2 * n_animals
            if have < need and ctx.money > 300 and len(orders) < F.MAX_MARKET_ORDERS:
                if not any(o[0] == "BUY_PRODUCT" and o[1] == "WHEAT" for o in orders):
                    orders = list(orders) + [["BUY_PRODUCT", "WHEAT", int(need - have)]]

        if not animals and not plants:
            return ops, orders

        free = [u for u in range(len(ops))
                if ops[u] and ops[u][0] in ("PASS", "NORTH", "SOUTH", "EAST", "WEST")]
        used = set()
        for tx, ty in animals + plants:
            is_animal = (tx, ty) in animals
            best, bestd = None, 10 ** 9
            for u in free:
                if u in used:
                    continue
                ux, uy = ctx.unit_pos[u]
                d = abs(ux - tx) + abs(uy - ty)
                carried = ctx.inventories[u] if u < len(ctx.inventories) else {}
                if is_animal and not carried.get("WHEAT", 0):
                    d += 12          # must detour to the shed first
                if d < bestd:
                    best, bestd = u, d
            if best is None:
                break
            used.add(best)
            ux, uy = ctx.unit_pos[best]
            carried = ctx.inventories[best] if best < len(ctx.inventories) else {}
            if is_animal and not carried.get("WHEAT", 0):
                if (ux, uy) in F.SHED_TILES:
                    ops[best] = (["PICKUP", "WHEAT", min(6, int(ctx.shed.get("WHEAT", 0)))]
                                 if ctx.shed.get("WHEAT", 0) else ["PASS"])
                else:
                    ops[best] = [_step_towards(ux, uy, 4 if ux <= 4 else 5,
                                               4 if uy <= 4 else 5)]
                continue
            if (ux, uy) == (tx, ty):
                ops[best] = ["FEED"] if is_animal else ["WATER"]
            else:
                ops[best] = [_step_towards(ux, uy, tx, ty)]
        return ops, orders

    # -- sanitiser ----------------------------------------------------------
    @staticmethod
    def _sanitise(ctx, ops, orders):
        # The engine drops EVERY PLANT op for a crop whose requests exceed the seeds held.
        budget = {c: int(ctx.seeds.get(c, 0)) for c in F.CROPS}
        for i, op in enumerate(ops):
            if op and op[0] == "PLANT":
                c = op[1]
                if budget.get(c, 0) <= 0:
                    ops[i] = ["PASS"]
                else:
                    budget[c] -= 1
        return ops, list(orders)[:F.MAX_MARKET_ORDERS]

    # -- entry point --------------------------------------------------------
    def act(self, obs):
        ctx = F.TurnContext(obs)
        ops, claimed = [], set()
        for u in range(ctx.n_units):
            op, claims = self._unit_op(ctx, u, claimed)
            ops.append(op)
            if claims:
                claimed.add(tuple(ctx.unit_pos[u]))
        orders = self._market(ctx)
        if self.safety:
            ops, orders = self._apply_safety(ctx, ops, orders)
        ops, orders = self._sanitise(ctx, ops, orders)
        return {"farmer": ops[0] if ops else ["PASS"],
                "hands": ops[1:ctx.n_units],
                "market": orders}


def _step_towards(x, y, tx, ty):
    """One 4-directional step from (x,y) toward (tx,ty). Every tile is passable, locked
    quadrants included, so a Manhattan step is always a legal move."""
    if x != tx:
        return "EAST" if tx > x else "WEST"
    if y != ty:
        return "SOUTH" if ty > y else "NORTH"
    return "PASS"


def load_default(directory=None, safety=3, liquidate_order="buys_first"):
    """Every .npz in weights/ (sorted) becomes an ensemble member; `BC_MEMBERS` overrides."""
    here = directory or os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    env = os.environ.get("BC_MEMBERS")
    if env:
        paths = [p for p in env.split(os.pathsep) if p]
    else:
        paths = sorted(os.path.join(here, f) for f in os.listdir(here) if f.endswith(".npz"))
    if not paths:
        raise RuntimeError(f"no BC weights found in {here}")
    return BCPolicy(paths, safety=safety, liquidate_order=liquidate_order)
