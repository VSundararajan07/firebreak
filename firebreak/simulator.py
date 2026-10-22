"""Tick orchestration: fire spread, evacuee vehicles, congestion and dirty-edge rerouting.

Per tick:
  1. the fire model advances over the quadtree front;
  2. ignitions close roads they touch and raise the threat on nearby roads;
  3. vehicles move, entering/leaving road slots in the Fenwick tree;
  4. roads whose closure, threat or congestion bucket changed are *dirty*;
     only vehicles whose remaining route uses a dirty road are rerouted,
     with one Dijkstra per (source node, town) per tick.
"""

from __future__ import annotations

import math
import random
import time

from . import config as C
from .fire_model import FireModel, Rain
from .profiler import Profiler
from .quadtree import RegionQuadtree, QuadNode
from .routing import RoadGraph, Route
from .terrain import generate_terrain

WAITING, MOVING, EVACUATED, STRANDED = 0, 1, 2, 3


class Agent:
    """One evacuating vehicle. At a node when ``edge`` is None, else on that edge."""

    __slots__ = ("id", "town", "status", "node", "edge", "from_node", "to_node", "dist",
                 "seg", "route", "route_pos", "spawn_tick", "done_tick")

    def __init__(self, aid: int, town: int, node: int, tick: int) -> None:
        self.id, self.town, self.status = aid, town, WAITING
        self.node: int | None = node
        self.edge: int | None = None
        self.from_node = self.to_node = node
        self.dist = 0.0
        self.seg = 0
        self.route: Route | None = None
        self.route_pos = 0          # index into route.edges of the next road to enter
        self.spawn_tick, self.done_tick = tick, -1


def _threat_bucket(threat: int) -> int:
    """Coarse threat level; a bucket change marks the road dirty."""
    return 0 if threat == 0 else 1 if threat < 4 else 2 if threat < C.THREAT_CAP else 3


class Simulation:
    """The complete, deterministic world state for one seed."""

    def __init__(self, seed: int = C.DEFAULT_SEED) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.terrain = generate_terrain(seed)
        self.tree = RegionQuadtree(C.MAX_DEPTH, C.MIN_DEPTH, self.terrain.cell_payload)
        self.fire = FireModel(self.terrain)
        self.roads = RoadGraph.from_terrain(self.terrain)
        self.profiler = Profiler(C.PROFILE_WINDOW)
        self.tick = 0
        self.wind_deg, self.wind_strength = C.DEFAULT_WIND_DEG, C.DEFAULT_WIND_STRENGTH
        self.rains: list[Rain] = []
        self.agents: list[Agent] = []
        self.dirty: set[int] = set()
        self.reroutes = 0
        self.burned_cells = 0
        self._forced: set[Agent] = set()
        self._route_seq = 0

    # ------------------------------------------------------------ commands
    @property
    def wind_vector(self) -> tuple[float, float]:
        """Wind as (x, y) in grid space, scaled by strength. Bearing 0 = up (north)."""
        rad = math.radians(self.wind_deg)
        return self.wind_strength * math.sin(rad), -self.wind_strength * math.cos(rad)

    def ignite(self, px: float, py: float) -> int:
        """Start a fire at pixel ``(px, py)``; returns cells ignited."""
        lit = self.fire.ignite(self.tree, int(px) // C.CELL_PX, int(py) // C.CELL_PX,
                               C.IGNITE_RADIUS)
        self._on_ignited(lit)
        return len(lit)

    def set_wind(self, direction_deg: float, strength: float) -> None:
        """Change wind and re-aim every burning cell's downwind road threat."""
        self.wind_deg = direction_deg % 360.0
        self.wind_strength = min(max(strength, 0.0), 1.0)
        touched: set[int] = set()
        for node in self.tree.burning:
            self._release_threat(node, touched)
            self._claim_threat(node, touched)
        self._refresh_threat(touched)

    def add_rain(self, px: float, py: float, radius_px: float) -> None:
        """Drop a rain cell centred on pixel ``(px, py)``."""
        r = min(max(radius_px, 4.0), C.MAX_RAIN_RADIUS_PX) / C.CELL_PX
        self.rains.append(Rain(px / C.CELL_PX, py / C.CELL_PX, r, C.RAIN_TICKS, C.RAIN_TICKS))

    def evacuate(self, town_id: int, count: int) -> int:
        """Queue ``count`` vehicles at a town; they depart a few per tick."""
        town = self.terrain.towns[town_id]
        route = self._compute_route(town["node"], town_id)
        for _ in range(count):
            agent = Agent(len(self.agents), town_id, town["node"], self.tick)
            agent.route = route
            if route is None:
                agent.status = STRANDED
            self.agents.append(agent)
        return count

    # ---------------------------------------------------------------- tick
    def step(self) -> None:
        """Advance the whole simulation by one tick."""
        t0 = time.perf_counter()
        events = self.fire.step(self.tree, self.wind_vector, self.rains, self.rng)
        self._on_ignited(events.ignited)
        touched: set[int] = set()
        for node in events.burned_out + events.doused:
            self._release_threat(node, touched)
        self._refresh_threat(touched)
        self.burned_cells += len(events.burned_out)
        for rain in self.rains:
            rain.ticks_left -= 1
        self.rains = [r for r in self.rains if r.ticks_left > 0]
        self._move_agents()
        self._mark_congestion()
        t1 = time.perf_counter()
        self._reroute()
        t2 = time.perf_counter()
        self.tick += 1
        self.profiler.record((t2 - t0) * 1000, events.frontier, (t2 - t1) * 1000,
                             len(self.tree.burning), self.tree.leaf_count())

    # ------------------------------------------------------------- fire → roads
    def _on_ignited(self, nodes: list[QuadNode]) -> None:
        """Close roads under new fire and add threat to roads near it."""
        touched: set[int] = set()
        g = self.terrain.grid
        for node in nodes:
            for eid in self.roads.burn_index.get(node.y * g + node.x, ()):
                if self.roads.close(eid):
                    self.dirty.add(eid)
            self._claim_threat(node, touched)
        self._refresh_threat(touched)

    def _claim_threat(self, node: QuadNode, touched: set[int]) -> None:
        """Threaten roads near ``node`` and along its downwind direction."""
        wx, wy = self.wind_vector
        g, reach = self.terrain.grid, C.WIND_THREAT_REACH
        found: set[int] = set()
        for k in (0.0, 0.5, 1.0):
            x = min(g - 1, max(0, node.x + round(wx * reach * k)))
            y = min(g - 1, max(0, node.y + round(wy * reach * k)))
            found.update(self.roads.threat_index.get(y * g + x, ()))
        node.threats = tuple(sorted(found))
        for eid in node.threats:
            self.roads.edges[eid].threat += 1
        touched.update(found)

    def _release_threat(self, node: QuadNode, touched: set[int]) -> None:
        for eid in node.threats:
            self.roads.edges[eid].threat -= 1
        touched.update(node.threats)
        node.threats = ()

    def _refresh_threat(self, touched: set[int]) -> None:
        for eid in touched:
            edge = self.roads.edges[eid]
            bucket = _threat_bucket(edge.threat)
            if bucket != edge.threat_bucket:
                edge.threat_bucket = bucket
                self.dirty.add(eid)

    def _mark_congestion(self) -> None:
        """Dirty every open road whose vehicle-count bucket changed (Fenwick queries)."""
        for edge in self.roads.edges:
            if edge.open:
                bucket = int(self.roads.congestion(edge.id) // C.CONGESTION_BUCKET)
                if bucket != edge.cong_bucket:
                    edge.cong_bucket = bucket
                    self.dirty.add(edge.id)

    # -------------------------------------------------------------- vehicles
    def _move_agents(self) -> None:
        released: dict[int, int] = {}
        for agent in self.agents:
            if agent.status == WAITING:
                if released.get(agent.town, 0) >= C.DEPART_PER_TICK:
                    continue
                released[agent.town] = released.get(agent.town, 0) + 1
                agent.status = MOVING
            if agent.status == MOVING:
                self._advance(agent)

    def _advance(self, agent: Agent) -> None:
        """Move one vehicle along its route; slower on crowded road slots."""
        roads = self.roads
        budget = C.AGENT_SPEED_PX
        if agent.edge is not None:
            others = roads.fenwick.range_sum(agent.seg, agent.seg) - 1
            budget /= 1.0 + C.SPEED_CONGESTION_K * max(others, 0)
        for _ in range(6):
            if agent.edge is None:
                if agent.node in roads.shelters:
                    agent.status, agent.done_tick = EVACUATED, self.tick
                    return
                route = agent.route
                if route is None or agent.route_pos >= len(route.edges):
                    self._forced.add(agent)
                    return
                eid = route.edges[agent.route_pos]
                if not roads.edges[eid].open:
                    self._forced.add(agent)
                    return
                self._enter_edge(agent, eid)
            edge = roads.edges[agent.edge]
            agent.dist += budget
            if agent.dist < edge.length:
                self._update_slot(agent, edge)
                return
            budget = agent.dist - edge.length
            roads.fenwick.point_add(agent.seg, -1)
            agent.node, agent.edge = agent.to_node, None

    def _enter_edge(self, agent: Agent, eid: int) -> None:
        edge = self.roads.edges[eid]
        agent.edge, agent.from_node = eid, agent.node
        agent.to_node, agent.node = edge.other(agent.node), None
        agent.dist = 0.0
        agent.route_pos += 1
        agent.seg = edge.segment_at(self._pos_from_u(agent, edge))
        self.roads.fenwick.point_add(agent.seg, 1)

    def _update_slot(self, agent: Agent, edge) -> None:
        seg = edge.segment_at(self._pos_from_u(agent, edge))
        if seg != agent.seg:
            self.roads.fenwick.point_add(agent.seg, -1)
            self.roads.fenwick.point_add(seg, 1)
            agent.seg = seg

    @staticmethod
    def _pos_from_u(agent: Agent, edge) -> float:
        return agent.dist if agent.from_node == edge.u else edge.length - agent.dist

    def agent_xy(self, agent: Agent) -> tuple[float, float]:
        """Pixel position of a vehicle."""
        if agent.edge is None:
            return self.roads.nodes[agent.node]
        edge = self.roads.edges[agent.edge]
        return edge.point_at(self._pos_from_u(agent, edge))

    # -------------------------------------------------------------- rerouting
    def _compute_route(self, source: int, town: int) -> Route | None:
        found = self.roads.route_to_shelter(source)
        if found is None:
            return None
        self._route_seq += 1
        return Route(self._route_seq, town, found[0], found[1], found[2], self.tick)

    def _affected(self, dirty: set[int]) -> set[Agent]:
        """Active vehicles whose remaining route (or current road) uses a dirty edge."""
        last_dirty: dict[int, int] = {}
        out: set[Agent] = set()
        for agent in self.agents:
            if agent.status not in (WAITING, MOVING) or agent.route is None:
                continue
            route = agent.route
            if route.id not in last_dirty:
                last_dirty[route.id] = max(
                    (i for i, e in enumerate(route.edges) if e in dirty), default=-1)
            if last_dirty[route.id] >= agent.route_pos or agent.edge in dirty:
                out.add(agent)
        return out

    def _reroute(self) -> None:
        """Batch reroute: one Dijkstra per (source, town) for all affected vehicles."""
        dirty, self.dirty = self.dirty, set()
        need = self._affected(dirty) if dirty else set()
        need |= self._forced
        self._forced = set()
        if dirty or self.tick % C.STRANDED_RETRY_TICKS == 0:
            need.update(a for a in self.agents if a.status == STRANDED)
        cache: dict[tuple[int, int], Route | None] = {}
        for agent in sorted(need, key=lambda a: a.id):
            if agent.edge is not None and not self.roads.edges[agent.edge].open:
                edge = self.roads.edges[agent.edge]
                agent.from_node, agent.to_node = agent.to_node, agent.from_node
                agent.dist = edge.length - agent.dist
            source = agent.to_node if agent.edge is not None else agent.node
            key = (source, agent.town)
            if key not in cache:
                cache[key] = self._compute_route(source, agent.town)
            if not self._keeps_route(agent, cache[key]):
                self._assign(agent, cache[key])

    def _keeps_route(self, agent: Agent, best: Route | None) -> bool:
        """Hysteresis: keep a fully open route whose cost is close to the new best."""
        old, roads = agent.route, self.roads
        if old is None or best is None or agent.status == STRANDED:
            return False
        if agent.edge is not None and not roads.edges[agent.edge].open:
            return False
        rest = old.edges[agent.route_pos:]
        if any(not roads.edges[e].open for e in rest):
            return False
        return sum(roads.edge_cost(e) for e in rest) <= best.cost * C.REROUTE_HYSTERESIS

    def _assign(self, agent: Agent, route: Route | None) -> None:
        old = agent.route
        old_rest = old.edges[agent.route_pos:] if old is not None else None
        agent.route, agent.route_pos = route, 0
        if route is None:
            if agent.edge is None:
                agent.status = STRANDED
            return
        if agent.status == STRANDED:
            agent.status = MOVING
        if old_rest != route.edges:
            self.reroutes += 1

    # -------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """JSON-ready state: packed quadtree leaves, roads, routes, vehicles, stats."""
        roads = self.roads
        counts = [0, 0, 0, 0]
        total_time = 0
        agents, routes = [], {}
        for a in self.agents:
            counts[a.status] += 1
            if a.status == EVACUATED:
                total_time += a.done_tick - a.spawn_tick
                continue
            x, y = self.agent_xy(a)
            rid = a.route.id if a.route is not None else -1
            if a.route is not None:
                routes[rid] = a.route
            agents.append([a.id, round(x, 1), round(y, 1), a.town, a.status, rid])
        cp = C.CELL_PX
        return {
            "tick": self.tick,
            "sim_time_s": self.tick * C.SIM_SECONDS_PER_TICK,
            "wind": {"direction_deg": self.wind_deg, "strength": self.wind_strength},
            "pack_bits": self.tree.max_depth,
            "leaves": self.tree.packed_leaves(),
            "rains": [[round(r.x * cp, 1), round(r.y * cp, 1), round(r.radius * cp, 1),
                       round(r.ticks_left / r.total, 2)] for r in self.rains],
            "roads": [0 if e.open and e.threat == 0 else 1 if e.open else 2 for e in roads.edges],
            "congestion": [roads.congestion(e.id) for e in roads.edges],
            "routes": [[r.id, r.town, r.start, self.tick - r.tick, r.edges]
                       for r in routes.values()],
            "agents": agents,
            "stats": {
                "tick_ms": round(self.profiler.last("tick_times"), 2),
                "reroute_ms": round(self.profiler.last("reroute_times"), 2),
                "frontier": self.profiler.last("frontier_sizes"),
                "burning": len(self.tree.burning),
                "leaves": self.tree.leaf_count(),
                "burned_cells": self.burned_cells,
                "waiting": counts[WAITING], "moving": counts[MOVING],
                "evacuated": counts[EVACUATED], "stranded": counts[STRANDED],
                "total_agents": len(self.agents),
                "avg_evac_s": round(total_time / counts[EVACUATED] * C.SIM_SECONDS_PER_TICK)
                if counts[EVACUATED] else 0,
                "reroutes": self.reroutes,
                "roads_closed": sum(1 for e in roads.edges if not e.open),
            },
        }
