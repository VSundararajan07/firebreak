"""Road graph, congestion/fire-aware edge costs and Dijkstra evacuation routing.

Intersections are nodes, road segments are edges. Each road is split into
fixed-length slots tracked by a Fenwick tree, so "vehicles on this road" is
an O(log n) range sum. Burned roads are closed (removed from the graph);
roads near fire get a cost penalty. Dijkstra uses the hand-written heap.
"""

from __future__ import annotations

import math
from typing import Callable, Hashable, Iterable

from . import config as C
from .fenwick import FenwickTree
from .heap import BinaryMinHeap
from .terrain import Point, trace_polyline

NeighborFn = Callable[[Hashable], Iterable[tuple[Hashable, float, int]]]


def dijkstra(neighbors: NeighborFn, source: Hashable, targets: Iterable | None = None):
    """Single-source shortest paths with a binary min-heap and decrease_key.

    ``neighbors(node)`` yields ``(neighbor, weight, edge_id)``. When ``targets``
    is given the search stops at the first target popped. Returns
    ``(dist, prev, reached)`` with ``prev[node] = (previous node, edge_id)``.
    O((V + E) log V).
    """
    goal = frozenset(targets) if targets is not None else None
    dist = {source: 0.0}
    prev: dict = {source: None}
    done: set = set()
    heap = BinaryMinHeap()
    heap.push(source, 0.0)
    while heap:
        node, d = heap.pop()
        done.add(node)
        if goal is not None and node in goal:
            return dist, prev, node
        for nbr, weight, eid in neighbors(node):
            if nbr in done:
                continue
            nd = d + weight
            if nbr not in dist:
                dist[nbr] = nd
                prev[nbr] = (node, eid)
                heap.push(nbr, nd)
            elif nd < dist[nbr]:
                dist[nbr] = nd
                prev[nbr] = (node, eid)
                heap.decrease_key(nbr, nd)
    return dist, prev, None


def reconstruct_path(prev: dict, target: Hashable) -> tuple[list, list[int]]:
    """Walk ``prev`` back from ``target``; returns ``(nodes, edge_ids)`` from the source."""
    nodes, edges, step = [target], [], prev[target]
    while step is not None:
        node, eid = step
        nodes.append(node)
        edges.append(eid)
        step = prev[node]
    nodes.reverse()
    edges.reverse()
    return nodes, edges


class RoadEdge:
    """A road between two intersections, stored as a polyline."""

    __slots__ = ("id", "u", "v", "points", "cum", "length", "seg_start", "seg_count",
                 "open", "threat", "threat_bucket", "cong_bucket")

    def __init__(self, eid: int, u: int, v: int, points: list[Point], seg_start: int) -> None:
        self.id, self.u, self.v, self.points = eid, u, v, points
        self.cum = [0.0]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.length = self.cum[-1]
        self.seg_start = seg_start
        self.seg_count = max(1, math.ceil(self.length / C.ROAD_SEGMENT_PX))
        self.open = True
        self.threat = 0
        self.threat_bucket = 0
        self.cong_bucket = 0

    def other(self, node: int) -> int:
        """The endpoint opposite ``node``."""
        return self.v if node == self.u else self.u

    def point_at(self, pos: float) -> Point:
        """Coordinates ``pos`` pixels along the road measured from ``u``."""
        pos = min(max(pos, 0.0), self.length)
        for i in range(1, len(self.cum)):
            if pos <= self.cum[i]:
                span = self.cum[i] - self.cum[i - 1] or 1.0
                f = (pos - self.cum[i - 1]) / span
                (x0, y0), (x1, y1) = self.points[i - 1], self.points[i]
                return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        return self.points[-1]

    def segment_at(self, pos: float) -> int:
        """Fenwick index of the slot containing position ``pos`` (from ``u``)."""
        k = int(pos / self.length * self.seg_count) if self.length else 0
        return self.seg_start + min(max(k, 0), self.seg_count - 1)


class Route:
    """An evacuation route from ``start`` to a shelter; shared by many vehicles."""

    __slots__ = ("id", "town", "start", "nodes", "edges", "cost", "tick")

    def __init__(self, rid: int, town: int, nodes: list[int], edges: list[int],
                 cost: float, tick: int) -> None:
        self.id, self.town, self.start = rid, town, nodes[0]
        self.nodes, self.edges, self.cost, self.tick = nodes, edges, cost, tick


class RoadGraph:
    """Undirected road network with live congestion (Fenwick) and fire state."""

    def __init__(self, nodes: list[Point], edges: list[tuple[int, int, list[Point]]],
                 shelters: list[int], grid: int, cell_px: int) -> None:
        self.nodes = list(nodes)
        self.edges: list[RoadEdge] = []
        slot = 1
        for eid, (u, v, pts) in enumerate(edges):
            edge = RoadEdge(eid, u, v, pts, slot)
            slot += edge.seg_count
            self.edges.append(edge)
        self.fenwick = FenwickTree(slot - 1)
        self.adj: list[list[tuple[int, int]]] = [[] for _ in self.nodes]
        for e in self.edges:
            self.adj[e.u].append((e.v, e.id))
            self.adj[e.v].append((e.u, e.id))
        self.shelters = frozenset(shelters)
        self.burn_index = self._cell_index(grid, cell_px, C.ROAD_BURN_RADIUS)
        self.threat_index = self._cell_index(grid, cell_px, C.THREAT_RADIUS)

    @classmethod
    def from_terrain(cls, terrain) -> "RoadGraph":
        """Build the graph from a generated :class:`Terrain`."""
        return cls(terrain.nodes, terrain.edges, terrain.shelters, terrain.grid, terrain.cell_px)

    def _cell_index(self, grid: int, cell_px: int, radius: int) -> dict[int, tuple[int, ...]]:
        """Map grid cell index -> roads passing within ``radius`` cells of it."""
        disk = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
                if dx * dx + dy * dy <= radius * radius]
        index: dict[int, list[int]] = {}
        for e in self.edges:
            cells = {(int(x) // cell_px, int(y) // cell_px) for x, y in trace_polyline(e.points)}
            covered = {(cy + dy) * grid + cx + dx for cx, cy in cells for dx, dy in disk
                       if 0 <= cx + dx < grid and 0 <= cy + dy < grid}
            for idx in covered:
                index.setdefault(idx, []).append(e.id)
        return {k: tuple(v) for k, v in index.items()}

    def congestion(self, eid: int) -> int:
        """Vehicles currently on road ``eid``: one Fenwick range sum."""
        e = self.edges[eid]
        return self.fenwick.range_sum(e.seg_start, e.seg_start + e.seg_count - 1)

    def edge_cost(self, eid: int) -> float:
        """base_cost * (1 + w_c * congestion) * (1 + w_f * min(threat, cap))."""
        e = self.edges[eid]
        return (e.length * (1.0 + C.CONGESTION_WEIGHT * self.congestion(eid))
                * (1.0 + C.FIRE_WEIGHT * min(e.threat, C.THREAT_CAP)))

    def neighbors(self, node: int):
        """Open roads leaving ``node`` as ``(neighbor, cost, edge_id)``."""
        for nbr, eid in self.adj[node]:
            if self.edges[eid].open:
                yield nbr, self.edge_cost(eid), eid

    def close(self, eid: int) -> bool:
        """Remove a burned road from the graph; True if it was open."""
        edge = self.edges[eid]
        was_open, edge.open = edge.open, False
        return was_open

    def route_to_shelter(self, source: int) -> tuple[list[int], list[int], float] | None:
        """Cheapest path from ``source`` to the nearest reachable shelter, or None."""
        dist, prev, reached = dijkstra(self.neighbors, source, self.shelters)
        if reached is None:
            return None
        nodes, edges = reconstruct_path(prev, reached)
        return nodes, edges, dist[reached]
