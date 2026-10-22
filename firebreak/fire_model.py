"""Probabilistic cellular-automaton fire spread over quadtree leaves.

Each tick every burning unit cell tries to ignite its unburned 8-neighbours.
The probability rises with vegetation, uphill slope and tail wind, and falls
with moisture and rain. Water cells never ignite; burned and extinguished
cells are never revisited. Work per tick is O(burning cells).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import config as C
from .quadtree import BURNED, BURNING, EXTINGUISHED, UNBURNED, QuadNode, RegionQuadtree


@dataclass
class Rain:
    """A circular rain cell in grid units that lasts ``ticks_left`` ticks."""

    x: float
    y: float
    radius: float
    ticks_left: int
    total: int

    def covers(self, px: float, py: float) -> bool:
        """True if grid point ``(px, py)`` is under the rain."""
        return (px - self.x) ** 2 + (py - self.y) ** 2 <= self.radius ** 2

    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box ``(x, y, w, h)`` for quadtree range queries."""
        return self.x - self.radius, self.y - self.radius, 2 * self.radius, 2 * self.radius


@dataclass
class FireEvents:
    """State changes produced by one fire tick."""

    ignited: list[QuadNode] = field(default_factory=list)
    burned_out: list[QuadNode] = field(default_factory=list)
    doused: list[QuadNode] = field(default_factory=list)
    frontier: int = 0


class FireModel:
    """Spread rules evaluated against grid-resolution terrain fields."""

    def __init__(self, terrain) -> None:
        self.grid = terrain.grid
        self.elev = terrain.fine_elev
        self.veg = terrain.fine_veg
        self.moist = terrain.fine_moist
        self.water = terrain.fine_water
        # Burning cells with no flammable unburned neighbour; states never revert
        # to UNBURNED, so these can skip neighbour checks until they burn out.
        self._enclosed: set[QuadNode] = set()

    def is_water(self, x: int, y: int) -> bool:
        """True if grid cell ``(x, y)`` contains water."""
        return self.water[y * self.grid + x]

    def burn_ticks(self, x: int, y: int) -> int:
        """How long a cell burns: denser fuel burns longer."""
        return C.BURN_TICKS_MIN + round(C.BURN_TICKS_VEG * self.veg[y * self.grid + x])

    def spread_probability(self, sx: int, sy: int, dx: int, dy: int, dist: float,
                           wind: tuple[float, float], rains: list[Rain]) -> float:
        """Probability that burning cell ``(sx, sy)`` ignites ``(sx+dx, sy+dy)`` this tick."""
        tx, ty = sx + dx, sy + dy
        t = ty * self.grid + tx
        veg = self.veg[t]
        if veg <= C.MIN_FUEL or self.water[t]:
            return 0.0
        p = C.P_BASE * veg * (1.0 - C.MOISTURE_DAMPING * self.moist[t])
        tail_wind = dx * wind[0] + dy * wind[1]
        rise = self.elev[t] - self.elev[sy * self.grid + sx]
        p *= math.exp((C.WIND_GAIN * tail_wind + C.SLOPE_GAIN * rise) / dist)
        if dist > 1.0:
            p *= C.DIAGONAL_FACTOR
        for rain in rains:
            if rain.covers(tx + 0.5, ty + 0.5):
                p *= C.RAIN_SPREAD_FACTOR
                break
        return p if p < C.MAX_SPREAD_P else C.MAX_SPREAD_P

    def ignite(self, tree: RegionQuadtree, gx: int, gy: int, radius: int) -> list[QuadNode]:
        """Ignite every flammable unburned cell within ``radius`` of ``(gx, gy)``."""
        lit = []
        for y in range(gy - radius, gy + radius + 1):
            for x in range(gx - radius, gx + radius + 1):
                if (x - gx) ** 2 + (y - gy) ** 2 > radius * radius + 0.5:
                    continue
                if not tree.in_bounds(x, y) or self.is_water(x, y):
                    continue
                if self.veg[y * self.grid + x] <= C.MIN_FUEL:
                    continue
                if tree.locate(x, y).state == UNBURNED:
                    lit.append(tree.insert(x, y, BURNING, burn_left=self.burn_ticks(x, y)))
        return lit

    def step(self, tree: RegionQuadtree, wind: tuple[float, float], rains: list[Rain],
             rng: random.Random) -> FireEvents:
        """Advance the fire one tick and apply the results to ``tree``."""
        events = FireEvents()
        doused = self._douse(tree, rains, rng)
        targets: dict[int, None] = {}
        evaluated: set[int] = set()
        g = self.grid
        enclosed = self._enclosed
        for leaf in list(tree.burning):
            if leaf in doused:
                enclosed.discard(leaf)
                continue
            if leaf not in enclosed:
                exposed = False
                for dx, dy, dist, _ in tree.unburned_neighbors(leaf):
                    key = (leaf.y + dy) * g + leaf.x + dx
                    if self.water[key] or self.veg[key] <= C.MIN_FUEL:
                        continue
                    exposed = True
                    if key in targets:
                        continue
                    evaluated.add(key)
                    p = self.spread_probability(leaf.x, leaf.y, dx, dy, dist, wind, rains)
                    if rng.random() < p:
                        targets[key] = None
                if not exposed:
                    enclosed.add(leaf)
            leaf.burn_left -= 1
            if leaf.burn_left <= 0:
                events.burned_out.append(leaf)
                enclosed.discard(leaf)
        events.frontier = len(evaluated)
        events.doused = list(doused)

        for leaf in events.doused:
            tree.set_state(leaf, EXTINGUISHED)
        for leaf in events.burned_out:
            tree.set_state(leaf, BURNED)
        for leaf in events.doused + events.burned_out:
            tree.merge_upward(leaf)
        for key in targets:
            x, y = key % g, key // g
            events.ignited.append(tree.insert(x, y, BURNING, burn_left=self.burn_ticks(x, y)))
        return events

    @staticmethod
    def _douse(tree: RegionQuadtree, rains: list[Rain], rng: random.Random) -> dict[QuadNode, None]:
        """Burning cells put out by rain this tick (found via quadtree range queries)."""
        out: dict[QuadNode, None] = {}
        for rain in rains:
            for leaf in tree.range_query(rain.bbox()):
                if (leaf.state == BURNING and leaf not in out
                        and rain.covers(leaf.x + 0.5, leaf.y + 0.5)
                        and rng.random() < C.RAIN_EXTINGUISH_P):
                    out[leaf] = None
        return out
