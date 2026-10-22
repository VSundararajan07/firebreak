"""Procedural terrain: value-noise fields, a river, towns and a road network.

Everything derives from one integer seed, so a world is fully reproducible.
numpy is used only for array arithmetic on the noise fields and the image;
no spatial indexing happens here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import config as C

Point = tuple[float, float]


# --------------------------------------------------------------------- noise
def _value_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Smooth value noise: random lattice values blended with smoothstep."""
    lattice = rng.random((cells + 1, cells + 1))
    t = np.arange(size) * (cells / size)
    i = t.astype(np.int64)
    f = t - i
    f = f * f * (3.0 - 2.0 * f)
    a, b = lattice[i][:, i], lattice[i][:, i + 1]
    c, d = lattice[i + 1][:, i], lattice[i + 1][:, i + 1]
    top = a + (b - a) * f[None, :]
    bottom = c + (d - c) * f[None, :]
    return top + (bottom - top) * f[:, None]


def _normalize(a: np.ndarray) -> np.ndarray:
    """Rescale an array to [0, 1]."""
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-9)


def _fbm(rng: np.random.Generator, size: int, base: int, octaves: int) -> np.ndarray:
    """Fractal Brownian motion: octaves of value noise, normalised to [0, 1]."""
    total, amp, cells = np.zeros((size, size)), 1.0, base
    for _ in range(octaves):
        total += amp * _value_noise(rng, size, cells)
        amp *= 0.5
        cells *= 2
    return _normalize(total)


def _river(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Meandering river crossing the map; returns (distance to centreline, water mask)."""
    ys = np.arange(size, dtype=float)
    phase = rng.uniform(0, 2 * math.pi, 3)
    centre = (rng.uniform(0.35, 0.65) * size
              + rng.uniform(25, 45) * np.sin(2 * math.pi * ys / rng.uniform(260, 420) + phase[0])
              + rng.uniform(8, 16) * np.sin(2 * math.pi * ys / rng.uniform(80, 140) + phase[1]))
    half_width = 4.0 + 1.5 * np.sin(ys / 37.0 + phase[2])
    dist = np.abs(np.arange(size)[None, :] - centre[:, None])
    water = dist <= half_width[:, None]
    if rng.random() < 0.5:
        return dist.T.copy(), water.T.copy()
    return dist, water


# ------------------------------------------------------------------- terrain
@dataclass
class Terrain:
    """Generated world: pixel fields, grid-resolution fuel arrays and road network."""

    seed: int
    size: int
    grid: int
    cell_px: int
    elevation: np.ndarray
    vegetation: np.ndarray
    moisture: np.ndarray
    water: np.ndarray
    river_dist: np.ndarray
    nodes: list[Point]
    edges: list[tuple[int, int, list[Point]]]
    shelters: list[int]
    towns: list[dict]
    town_blocks: list[tuple[int, int, int, int, int]]
    fine_elev: list[float] = field(default_factory=list)
    fine_veg: list[float] = field(default_factory=list)
    fine_moist: list[float] = field(default_factory=list)
    fine_water: list[bool] = field(default_factory=list)
    _sat_veg: list[float] = field(default_factory=list)
    _sat_moist: list[float] = field(default_factory=list)
    _image: bytes | None = None

    def build_grid_fields(self) -> None:
        """Downsample pixel fields to grid cells and build summed-area tables."""
        g, cp = self.grid, self.cell_px

        def block(a: np.ndarray, how: str) -> np.ndarray:
            """Reduce each cell_px x cell_px pixel block by mean or max."""
            r = a.reshape(g, cp, g, cp)
            return r.max(axis=(1, 3)) if how == "max" else r.mean(axis=(1, 3))

        veg, moist = block(self.vegetation, "mean"), block(self.moisture, "mean")
        water = block(self.water.astype(float), "max") > 0
        veg[water] = 0.0
        self.fine_elev = block(self.elevation, "mean").ravel().tolist()
        self.fine_veg = veg.ravel().tolist()
        self.fine_moist = moist.ravel().tolist()
        self.fine_water = water.ravel().tolist()
        self._sat_veg = self._summed_area(veg)
        self._sat_moist = self._summed_area(moist)

    @staticmethod
    def _summed_area(a: np.ndarray) -> list[float]:
        sat = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
        sat[1:, 1:] = a.cumsum(0).cumsum(1)
        return sat.ravel().tolist()

    def cell_payload(self, x: int, y: int, size: int) -> tuple[float, float]:
        """Mean ``(moisture, vegetation)`` of a square of grid cells in O(1)."""
        stride = self.grid + 1
        a, b = y * stride + x, y * stride + x + size
        c, d = (y + size) * stride + x, (y + size) * stride + x + size
        area = float(size * size)
        m, v = self._sat_moist, self._sat_veg
        return (m[d] - m[b] - m[c] + m[a]) / area, (v[d] - v[b] - v[c] + v[a]) / area

    def render_rgba(self) -> bytes:
        """Shaded RGBA image of the landscape (cached)."""
        if self._image is None:
            self._image = _render(self)
        return self._image


# ------------------------------------------------------------- road network
def _nudge_to_land(water: np.ndarray, x: float, y: float) -> Point:
    """Move a point to the nearest dry pixel found on a small outward spiral."""
    size = water.shape[0]
    for r in range(0, 40, 2):
        for k in range(max(1, r * 2)):
            a = 2 * math.pi * k / max(1, r * 2)
            px = min(size - 1, max(0, int(x + r * math.cos(a))))
            py = min(size - 1, max(0, int(y + r * math.sin(a))))
            if not water[py, px]:
                return float(px), float(py)
    return x, y


def _curve(rnd: random.Random, p0: Point, p1: Point, steps: int = 8) -> list[Point]:
    """Gently curved road: quadratic Bezier with a jittered control point."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    off = rnd.uniform(-0.12, 0.12) * length
    cx = (p0[0] + p1[0]) / 2 - dy / length * off
    cy = (p0[1] + p1[1]) / 2 + dx / length * off
    pts = []
    for s in range(steps + 1):
        t = s / steps
        pts.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * cx + t * t * p1[0],
                     (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * cy + t * t * p1[1]))
    return pts


def trace_polyline(pts: list[Point], step: float = 1.0) -> list[Point]:
    """Points sampled every ``step`` pixels along a polyline."""
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        out.extend((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(1, n + 1))
    return out


def _crosses_water(water: np.ndarray, pts: list[Point]) -> bool:
    size = water.shape[0]
    return any(water[min(size - 1, int(y)), min(size - 1, int(x))] for x, y in trace_polyline(pts))


def _find(parent: list[int], i: int) -> int:
    """Union-find root with path halving."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _road_network(rnd: random.Random, water: np.ndarray, size: int):
    """Jittered lattice of intersections joined by a connected set of curved roads.

    Returns ``(nodes, edges, shelters, lattice_index)``. The outer lattice ring
    only hosts shelter nodes, each linked to one interior intersection.
    """
    n = C.ROAD_LATTICE
    step = (size - 1) / (n - 1)
    nodes: list[Point] = []
    index: dict[tuple[int, int], int] = {}
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            x = i * step + rnd.uniform(-0.22, 0.22) * step
            y = j * step + rnd.uniform(-0.22, 0.22) * step
            index[(i, j)] = len(nodes)
            nodes.append(_nudge_to_land(water, x, y))

    pairs: list[tuple[int, int]] = []
    for (i, j), u in index.items():
        for di, dj in ((1, 0), (0, 1)):
            if (i + di, j + dj) in index:
                pairs.append((u, index[(i + di, j + dj)]))
        if (i + 1, j + 1) in index and rnd.random() < 0.22:
            pairs.append((u, index[(i + 1, j + 1)]) if rnd.random() < 0.5
                         else (index[(i + 1, j)], index[(i, j + 1)]))

    sides = {"top": [], "bottom": [], "left": [], "right": []}
    edge_px = (3.0, size - 4.0)
    for k in range(1, n - 1):
        sides["top"].append(((k * step, edge_px[0]), (k, 1)))
        sides["bottom"].append(((k * step, edge_px[1]), (k, n - 2)))
        sides["left"].append(((edge_px[0], k * step), (1, k)))
        sides["right"].append(((edge_px[1], k * step), (n - 2, k)))
    slots = [s for side in sides.values() for s in rnd.sample(side, len(side))]
    def dry_slot(slot) -> bool:
        """True if a shelter slot sits on land."""
        return not water[int(slot[0][1]), int(slot[0][0])]

    chosen = []
    for name in rnd.sample(list(sides), 4):
        slot = next((s for s in sides[name] if dry_slot(s)), None)
        if slot is not None:
            chosen.append(slot)
    for slot in slots:
        if len(chosen) >= C.SHELTER_COUNT:
            break
        if slot not in chosen and dry_slot(slot):
            chosen.append(slot)
    shelters, forced = [], []
    for point, inner in chosen:
        shelters.append(len(nodes))
        forced.append((len(nodes), index[inner]))
        nodes.append(point)

    curves = {pair: _curve(rnd, nodes[pair[0]], nodes[pair[1]]) for pair in forced + pairs}
    dry = [p for p in pairs if not _crosses_water(water, curves[p])]
    dry_set = set(dry)
    wet = [p for p in pairs if p not in dry_set]
    rnd.shuffle(dry)
    rnd.shuffle(wet)
    parent = list(range(len(nodes)))
    kept = []
    for group, extra in ((forced, 1.0), (dry, 0.72), (wet, 0.25)):
        for u, v in group:
            ru, rv = _find(parent, u), _find(parent, v)
            if ru != rv or rnd.random() < extra:
                parent[ru] = rv
                kept.append((u, v, curves[(u, v)]))
    kept.sort(key=lambda e: (e[0], e[1]))
    return nodes, kept, shelters, index


def _place_towns(rnd: random.Random, nodes: list[Point], index: dict) -> list[int]:
    """Pick well-separated interior intersections by farthest-point sampling."""
    n = C.ROAD_LATTICE
    candidates = [u for (i, j), u in sorted(index.items()) if 2 <= i <= n - 3 and 2 <= j <= n - 3]
    towns = [rnd.choice(candidates)]
    while len(towns) < C.TOWN_COUNT:
        towns.append(max(candidates, key=lambda c: min(
            math.dist(nodes[c], nodes[t]) for t in towns)))
    return towns


# ---------------------------------------------------------------- generation
def generate_terrain(seed: int = C.DEFAULT_SEED) -> Terrain:
    """Build a complete, reproducible world for ``seed``."""
    size = C.WORLD_PX
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    river_dist, water = _river(rng, size)
    elevation = _fbm(rng, size, 3, 5)
    elevation = _normalize(elevation * (0.45 + 0.55 * np.clip(river_dist / 90.0, 0, 1)))
    moisture = np.clip(0.55 * _fbm(rng, size, 4, 4) + 0.5 * np.exp(-river_dist / 55.0), 0, 1)
    vegetation = _fbm(rng, size, 5, 4)
    vegetation = np.clip(0.08 + 1.05 * vegetation ** 1.15 * (0.55 + 0.45 * moisture)
                         - 0.5 * np.clip(elevation - 0.78, 0, 1), 0.02, 1.0)

    nodes, edges, shelters, index = _road_network(rnd, water, size)
    town_nodes = _place_towns(rnd, nodes, index)
    towns, blocks = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    for tid, node in enumerate(town_nodes):
        tx, ty = nodes[node]
        towns.append({"id": tid, "name": C.TOWN_NAMES[tid % len(C.TOWN_NAMES)],
                      "node": node, "x": tx, "y": ty})
        vegetation[(xx - tx) ** 2 + (yy - ty) ** 2 < 22 ** 2] *= 0.35
        for _ in range(22):
            bx, by = int(tx + rnd.gauss(0, 10)), int(ty + rnd.gauss(0, 10))
            w, h = rnd.randint(3, 7), rnd.randint(3, 6)
            if 0 <= bx < size - w and 0 <= by < size - h and not water[by:by + h, bx:bx + w].any():
                blocks.append((bx, by, w, h, rnd.randrange(4)))
    for _, _, pts in edges:
        for x, y in trace_polyline(pts):
            px, py = min(size - 1, int(x)), min(size - 1, int(y))
            vegetation[py, px] = min(vegetation[py, px], 0.3)

    vegetation[water] = 0.0
    moisture[water] = 1.0
    terrain = Terrain(seed, size, C.GRID, C.CELL_PX, elevation, vegetation, moisture, water,
                      river_dist, nodes, edges, shelters, towns, blocks)
    terrain.build_grid_fields()
    return terrain


# ----------------------------------------------------------------- rendering
_DRY = np.array([190, 163, 110.0])
_GRASS = np.array([122, 152, 74.0])
_FOREST = np.array([34, 86, 50.0])
_ROCK = np.array([150, 141, 128.0])
_ROOFS = np.array([[168, 84, 70], [120, 118, 126], [176, 150, 112], [96, 108, 122.0]])


def _render(t: Terrain) -> bytes:
    """Colour by vegetation, hill-shade by elevation, then river, contours, towns."""
    v = t.vegetation[..., None]
    low = _DRY + (_GRASS - _DRY) * np.clip(v / 0.5, 0, 1)
    rgb = np.where(v < 0.5, low, _GRASS + (_FOREST - _GRASS) * np.clip((v - 0.5) / 0.5, 0, 1))
    rock = np.clip((t.elevation - 0.8) / 0.2, 0, 1)[..., None] * 0.55
    rgb = rgb * (1 - rock) + _ROCK * rock
    gy, gx = np.gradient(t.elevation)
    shade = np.clip(1.0 + 60.0 * (-gx - gy), 0.62, 1.38)[..., None]
    rgb = rgb * shade
    band = np.floor(t.elevation * 16)
    contour = (band != np.roll(band, 1, 0)) | (band != np.roll(band, 1, 1))
    rgb[contour] *= 0.86
    bank = (~t.water) & (t.river_dist < 7.5)
    rgb[bank] = rgb[bank] * 0.4 + np.array([196, 182, 134.0]) * 0.6
    depth = np.clip(1.0 - t.river_dist / 6.0, 0, 1)[..., None]
    river = np.array([70, 140, 190.0]) * (1 - depth) + np.array([28, 82, 140.0]) * depth
    rgb = np.where(t.water[..., None], river, rgb)
    for bx, by, w, h, roof in t.town_blocks:
        rgb[by - 1:by + h + 1, bx - 1:bx + w + 1] = (52, 48, 46)
        rgb[by:by + h, bx:bx + w] = _ROOFS[roof]
    rgba = np.empty((t.size, t.size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba.tobytes()
