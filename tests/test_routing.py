"""Dijkstra on a hand-checked graph, road-graph costs, and live rerouting."""

import json

import pytest

from firebreak import config as C
from firebreak.routing import RoadGraph, dijkstra, reconstruct_path
from firebreak.simulator import MOVING, STRANDED, WAITING, Simulation

#        4         5         3
#   A ------- B ------- D ------- E
#    \       /         /
#   2 \     / 1       / 8
#      \   /         /
#        C ---------
GRAPH = {
    "A": [("B", 4, 0), ("C", 2, 1)],
    "B": [("A", 4, 0), ("C", 1, 2), ("D", 5, 3)],
    "C": [("A", 2, 1), ("B", 1, 2), ("D", 8, 4)],
    "D": [("B", 5, 3), ("C", 8, 4), ("E", 3, 5)],
    "E": [("D", 3, 5)],
    "F": [],
}


def test_dijkstra_hand_checked_distances():
    dist, prev, reached = dijkstra(lambda n: GRAPH[n], "A")
    assert reached is None
    assert dist == {"A": 0, "B": 3, "C": 2, "D": 8, "E": 11}   # B via C (decrease_key)
    nodes, edges = reconstruct_path(prev, "E")
    assert nodes == ["A", "C", "B", "D", "E"]
    assert edges == [1, 2, 3, 5]
    assert "F" not in dist                                    # unreachable


def test_dijkstra_stops_at_nearest_target():
    dist, prev, reached = dijkstra(lambda n: GRAPH[n], "A", targets={"D", "E"})
    assert reached == "D" and dist["D"] == 8
    assert reconstruct_path(prev, "D")[0] == ["A", "C", "B", "D"]


def _square_graph():
    # 0 -- 1          node 3 is the shelter; two routes 0->1->3 and 0->2->3
    # |    |
    # 2 -- 3
    nodes = [(0, 0), (100, 0), (0, 100), (100, 100)]
    edges = [(0, 1, [(0, 0), (100, 0)]), (1, 3, [(100, 0), (100, 100)]),
             (0, 2, [(0, 0), (0, 100)]), (2, 3, [(0, 100), (100, 100)])]
    return RoadGraph(nodes, edges, [3], grid=128, cell_px=2)


def test_closed_roads_are_removed_and_congestion_raises_cost():
    g = _square_graph()
    base = g.edge_cost(0)
    assert base == pytest.approx(100.0)
    first = g.edges[0]
    g.fenwick.point_add(first.seg_start, 3)
    assert g.congestion(0) == 3
    assert g.edge_cost(0) == pytest.approx(100.0 * (1 + C.CONGESTION_WEIGHT * 3))
    nodes, edges, _ = g.route_to_shelter(0)
    assert edges == [2, 3]                                    # avoids the congested road
    g.close(2)
    assert g.route_to_shelter(0)[1] == [0, 1]
    g.close(0)
    assert g.route_to_shelter(0) is None


def test_burned_road_triggers_reroute_of_affected_vehicles():
    sim = Simulation(seed=3)
    sim.evacuate(0, 30)
    for _ in range(3):
        sim.step()
    agent = next(a for a in sim.agents if a.status == MOVING and a.route.edges[a.route_pos:])
    doomed = agent.route.edges[agent.route_pos]
    sim.roads.close(doomed)
    sim.dirty.add(doomed)
    sim.step()
    for a in sim.agents:
        if a.status in (WAITING, MOVING) and a.route is not None:
            assert doomed not in a.route.edges[a.route_pos:]
    assert sim.reroutes > 0


def test_vehicles_reach_shelters_and_state_fits_budget():
    sim = Simulation(seed=5)
    sim.evacuate(1, 60)
    for x, y in ((120, 120), (400, 380), (250, 90)):
        sim.ignite(x, y)
    for _ in range(140):
        sim.step()
    stats = sim.snapshot()["stats"]
    assert stats["evacuated"] + stats["stranded"] + stats["moving"] + stats["waiting"] == 60
    assert stats["evacuated"] > 0
    assert len(json.dumps(sim.snapshot(), separators=(",", ":"))) <= 200_000
    assert all(a.status != STRANDED or a.route is None for a in sim.agents)
