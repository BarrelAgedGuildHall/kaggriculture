"""Millrace M3a — layout/siting: contiguous rings from quadrant anchors.

Every new structure or plant is sited to minimise cluster radius, not "first
free tile". One cluster per unlocked quadrant, anchored at that quadrant's
shed-access tile ((4,4) NW, (5,4) NE, (4,5) SW, (5,5) SE on the 10x10 board).
Assets that need daily multi-action service (animals) take the tiles closest
to the anchor; one-time crops (wheat/carrot) sit just outside; melon
(1 action/day, 10-day cycle) takes the far fringe.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

ANCHORS = {"NW": (4, 4), "NE": (5, 4), "SW": (4, 5), "SE": (5, 5)}


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def quadrant_of(pos, board=10):
    half = board // 2
    return ("N" if pos[1] < half else "S") + ("W" if pos[0] < half else "E")


def anchor_of(pos, board=10):
    return ANCHORS[quadrant_of(pos, board)]


class SitePlan:
    """Orders a set of empty tiles into near-anchor (animal) and far-fringe
    (crop/melon) pools, ring by ring around each tile's own quadrant anchor."""

    def __init__(self, empties, board=10):
        ranked = sorted(empties, key=lambda p: (dist(p, anchor_of(p, board)),
                                                dist(p, ANCHORS["NW"])))
        self._near = list(ranked)     # pop from the front: animals
        self._far = list(ranked)      # pop from the back: melon, then crops
        self._taken = set()

    def _pop(self, seq, from_front):
        it = seq if from_front else reversed(seq)
        for p in it:
            if p not in self._taken:
                self._taken.add(p)
                return p
        return None

    def take_animal_site(self):
        return self._pop(self._near, True)

    def take_melon_site(self):
        return self._pop(self._far, False)

    def take_melon_site_capped(self, board=10):
        """Feed-chain arm only (the frozen expB-201 champion keeps the plain
        far-pop above). Fringe but never the far corners: melon needs one
        WATER every day for 10-12 days, and at anchor-distance 8-9 that
        service reliably loses scheduling races (14 of 17 weed-kills in the
        expB-213 M4 gate run were corner melons at age 8-11)."""
        for p in reversed(self._far):
            if p not in self._taken and dist(p, anchor_of(p, board)) <= 6:
                self._taken.add(p)
                return p
        return self._pop(self._far, False)

    def take_crop_site(self):
        return self._pop(self._far, False)

    def take_crop_site_capped(self, board=10):
        """Feed-chain arm only (the frozen expB-201 champion keeps the plain
        far-pop above). Fringe, but corners (anchor-distance 7-8) only as a
        last resort — daily watering at that distance reliably loses
        scheduling races and the plant weeds out (measured in both expB-213
        M4 gate runs)."""
        for p in reversed(self._far):
            if p not in self._taken and dist(p, anchor_of(p, board)) <= 6:
                self._taken.add(p)
                return p
        return self._pop(self._far, False)

    def take_wheat_site(self, board=10):
        """M6 siting: wheat lives INSIDE the animal clusters so feed does not
        have to be couriered from the shed (the load-bearing feed-courier MOVE
        found by expB-201) — but NOT in the innermost ring: the first cut of
        this method popped the near pool front and displaced later pastures to
        the cluster fringe (measured: unfed cows at anchor-distance 4-5,
        expB-213 audit_v1). Animals keep the ring at distance < 2; wheat takes
        the nearest tile at distance >= 2. Only the feed-chain arm calls this;
        the frozen expB-201 champion keeps wheat on the fringe."""
        for p in self._near:
            if p not in self._taken and dist(p, anchor_of(p, board)) >= 2:
                self._taken.add(p)
                return p
        return self._pop(self._near, True)

    def remaining(self):
        return [p for p in self._near if p not in self._taken]
