"""Millrace M3b — routing: persistent cluster assignment + target-locked routes.

The change vs the probe's per-turn greedy nearest-unit dispatch (measured MOVE
38.6-45.8% of unit-turns; v0 52.5-54.8%):

  1. **Persistent assignment.** Each unit is bound to one cluster (= unlocked
     quadrant) for the whole day. Bindings recompute at day boundaries, or when
     a new hand appears mid-day (it joins the neediest cluster). A unit whose
     cluster queue is empty may borrow the nearest task anywhere, but its home
     binding does not move.
  2. **Target locks.** Once a unit picks a task it walks to it until the task
     is done or invalidated, instead of being re-dispatched across the board
     every turn. An URGENT task (keep-alive band, prio <= 2.5) in the unit's
     own cluster may pre-empt a locked non-urgent target.
  3. **Batched pickups.** A unit that needs WHEAT / FERTILIZER / an animal it
     does not carry routes via the shed and PICKUPs enough for several tasks in
     one action (PICKUP <item> n is one action for n items).

On-tile batching (finish all pending work on the current tile before moving)
is handled by the caller before route assignment, exactly as in the probe, so
this module only ever sees units with no work under their feet.

Route order within a cluster is incremental nearest-neighbour under the locks
(v0 simplification of the spec's hour-0 route + 2-opt: the task set mutates
every turn as animals get fed and fertilizer appears, so a frozen tour goes
stale within hours; locks give the same thrash-kill at zero replan cost).
MOVE fraction is the judge — expB-201.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

from .layout import ANCHORS, dist, quadrant_of

URGENT_PRIO = 2.5  # keep-alive band: PLACE 0, FEED 1, BUILD 2, WATER-critical 2.5


class Router:
    def __init__(self, board=10):
        self.board = board
        self.day = -1
        self.home = {}     # unit index -> quadrant name
        self.locks = {}    # unit index -> {"key","pos","prio","act","req"}

    # ---- daily assignment ---------------------------------------------------
    def _workloads(self, tasks, quads):
        w = {q: 0 for q in quads}
        for prio, pos, act, key in tasks:
            q = quadrant_of(pos, self.board)
            if q in w:
                w[q] += 1
        return w

    def _rebind(self, units, tasks, quads):
        """Assign all units to clusters proportionally to task load."""
        self.home = {}
        w = self._workloads(tasks, quads)
        total = sum(w.values()) or 1
        counts = {q: 0 for q in quads}
        order = sorted(range(len(units)),
                       key=lambda i: min(dist(units[i], ANCHORS[q]) for q in quads))
        for i in order:
            # pick the cluster with the largest remaining need; distance breaks ties
            def need(q):
                share = w[q] / total * len(units)
                return share - counts[q]
            best = max(quads, key=lambda q: (need(q), -dist(units[i], ANCHORS[q])))
            self.home[i] = best
            counts[best] += 1

    def _bind_new(self, i, unit_pos, tasks, quads):
        w = self._workloads(tasks, quads)
        assigned = {q: 0 for q in quads}
        for q in self.home.values():
            if q in assigned:
                assigned[q] += 1
        best = max(quads, key=lambda q: (w[q] - 2 * assigned[q],
                                         -dist(unit_pos, ANCHORS[q])))
        self.home[i] = best

    # ---- per-turn planning ---------------------------------------------------
    def plan(self, day, units, busy, actions, tasks, ctx):
        """Fill `actions` for non-busy units. `tasks` = [(prio,pos,act,key)].

        ctx: dict with
          invs:        per-unit inventory dicts
          shed:        shed contents dict (this turn's remaining, mutated here)
          quads:       list of unlocked quadrant names
          step_toward: fn(pos, target) -> move action string
          shed_tiles:  list of shed-access tiles
          nearest_shed:fn(pos) -> shed tile
          requirement: fn(act) -> item name or None
          batch:       dict item -> pickup batch size
        """
        quads = ctx["quads"]
        if day != self.day:
            self.day = day
            self.locks = {}
            self._rebind(units, tasks, quads)
        for i in range(len(units)):
            if i not in self.home:
                self._bind_new(i, units[i], tasks, quads)
        # units that vanished (never within a day, but be safe)
        for i in list(self.locks):
            if i >= len(units):
                del self.locks[i]

        by_key = {t[3]: t for t in tasks}
        claimed = ctx.get("claimed", set())

        # 1. re-validate locks (they hold their claim before free picks)
        for i in list(self.locks):
            lk = self.locks[i]
            if busy[i] or lk["key"] not in by_key or lk["key"] in claimed:
                del self.locks[i]
                continue
            claimed.add(lk["key"])

        # 2. urgent pre-emption + free units pick targets
        open_tasks = [t for t in tasks if t[3] not in claimed]
        for i, pos in enumerate(units):
            if busy[i]:
                continue
            lock = self.locks.get(i)
            if lock is not None and lock["prio"] > URGENT_PRIO:
                urgent = self._pick(pos, open_tasks, self.home.get(i), urgent_only=True)
                if urgent is not None:
                    claimed.discard(lock["key"])
                    open_tasks.append(by_key[lock["key"]])
                    lock = None
            if lock is None:
                choice = self._pick(pos, open_tasks, self.home.get(i))
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
            if actions[i] is None:  # requirement unsatisfiable: release
                actions[i] = ["PASS"]
                claimed.discard(lock["key"])
                del self.locks[i]
                busy[i] = False
        return claimed

    def _pick(self, pos, open_tasks, home, urgent_only=False):
        """Nearest task by (priority, distance); own cluster first."""
        best, best_score = None, None
        for t in open_tasks:
            prio, tpos, act, key = t
            if urgent_only and prio > URGENT_PRIO:
                continue
            in_home = (home is not None and quadrant_of(tpos, self.board) == home)
            score = (0 if in_home else 1, prio, dist(pos, tpos))
            if best_score is None or score < best_score:
                best, best_score = t, score
        # a unit never leaves its cluster while its cluster has work:
        if best is not None and best_score[0] == 1:
            # only borrow if the home cluster is empty at every priority
            if any((home is not None and quadrant_of(t[1], self.board) == home)
                   for t in open_tasks):
                return None
        return best

    def _execute(self, i, pos, lock, ctx):
        """Turn a locked target into this turn's action for unit i."""
        req = lock["req"]
        invs = ctx["invs"]
        shed = ctx["shed"]
        if req and invs[i].get(req, 0) <= 0:
            if shed.get(req, 0) <= 0:
                return None  # nothing carried, nothing in the shed: give up
            if pos in ctx["shed_tiles"]:
                k = min(shed[req], ctx["batch"].get(req, 1))
                shed[req] -= k
                return ["PICKUP", req, int(k)]
            return [ctx["step_toward"](pos, ctx["nearest_shed"](pos))]
        if pos == lock["pos"]:
            act = lock["act"]
            del self.locks[i]  # done (or attempted) this turn: release
            return list(act)
        return [ctx["step_toward"](pos, lock["pos"])]
