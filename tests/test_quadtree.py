"""Region quadtree correctness plus fire-model sanity checks driven through it."""

import random

import pytest

from firebreak.fire_model import FireModel, Rain
from firebreak.quadtree import (BURNED, BURNING, EXTINGUISHED, NEIGHBOR_OFFSETS, UNBURNED,
                                RegionQuadtree)
from firebreak.simulator import Simulation


def test_insert_subdivides_only_along_path():
    tree = RegionQuadtree(max_depth=3)
    leaf = tree.insert(5, 2, BURNING, burn_left=4)
    assert (leaf.x, leaf.y, leaf.size, leaf.depth) == (5, 2, 1, 3)
    assert tree.leaf_count() == 1 + 3 * 3          # each split adds three leaves
    assert list(tree.burning) == [leaf]
    assert tree.locate(5, 2) is leaf
    assert tree.locate(0, 7).size == 4               # far corner stays coarse


def test_min_depth_pre_splits_root():
    tree = RegionQuadtree(max_depth=4, min_depth=2)
    assert tree.leaf_count() == 16
    assert {leaf.size for leaf in tree.leaves()} == {4}


def test_merge_children_when_uniform_and_never_while_burning():
    tree = RegionQuadtree(max_depth=2)
    parent = tree.root
    kids = tree.subdivide(parent)
    for kid in kids:
        tree.set_state(kid, BURNING, burn_left=2)
    assert not tree.merge_children(parent)           # burning front keeps resolution
    for kid in kids[:3]:
        tree.set_state(kid, BURNED)
    assert not tree.merge_children(parent)           # mixed states do not merge
    tree.set_state(kids[3], BURNED)
    assert tree.merge_children(parent)
    assert parent.is_leaf and parent.state == BURNED and tree.leaf_count() == 1
    assert not tree.burning


def test_merge_upward_compacts_extinguished_region():
    tree = RegionQuadtree(max_depth=3)
    cells = [tree.insert(x, y, EXTINGUISHED) for y in range(4) for x in range(4)]
    assert tree.leaf_count() == 16 + 3
    merges = sum(tree.merge_upward(leaf) for leaf in cells)
    assert merges == 5                                # four 2x2 blocks, then the quadrant
    assert tree.locate(0, 0).size == 4 and tree.locate(0, 0).state == EXTINGUISHED
    assert tree.leaf_count() == 4


def test_range_query_matches_brute_force():
    rng = random.Random(7)
    tree = RegionQuadtree(max_depth=5)
    for _ in range(60):
        tree.insert(rng.randrange(32), rng.randrange(32), BURNING)
    for _ in range(25):
        x, y, w, h = rng.randrange(32), rng.randrange(32), rng.randint(1, 12), rng.randint(1, 12)
        got = {id(n) for n in tree.range_query((x, y, w, h))}
        expected = {id(n) for n in tree.leaves() if n.intersects(x, y, w, h)}
        assert got == expected


def test_frontier_cells_are_unburned_neighbours_of_burning():
    tree = RegionQuadtree(max_depth=4)
    tree.insert(7, 7, BURNING)
    tree.insert(8, 7, BURNING)
    frontier = tree.frontier_cells()
    expected = {id(tree.locate(b.x + dx, b.y + dy)) for b in tree.burning
                for dx, dy, _ in NEIGHBOR_OFFSETS
                if tree.locate(b.x + dx, b.y + dy).state == UNBURNED}
    assert {id(n) for n in frontier} == expected
    assert all(n.state == UNBURNED for n in frontier)


def test_frontier_ignores_cells_outside_tree():
    tree = RegionQuadtree(max_depth=3)
    tree.insert(0, 0, BURNING)
    assert tree.neighbor(tree.locate(0, 0), -1, 0) is None
    assert len(tree.frontier_cells()) == 3           # E, S, SE neighbours only


def test_insert_collapses_finer_subtree():
    tree = RegionQuadtree(max_depth=3)
    tree.insert(1, 1, BURNING)
    node = tree.insert(0, 0, BURNED, depth=1)
    assert node.size == 4 and node.is_leaf
    assert not tree.burning
    with pytest.raises(ValueError):
        tree.insert(99, 0, BURNING)


# ------------------------------------------------------------ fire sanity
@pytest.fixture(scope="module")
def sim():
    return Simulation(seed=7)


def test_fire_never_enters_water(sim):
    terrain, fire = sim.terrain, sim.fire
    g = terrain.grid
    water = [i for i, w in enumerate(terrain.fine_water) if w]
    assert water, "terrain must contain a river"
    tree = RegionQuadtree(8, 2, terrain.cell_payload)
    rng = random.Random(1)
    for idx in water[:: max(1, len(water) // 12)]:   # ignite along the river banks
        x, y = idx % g, idx // g
        fire.ignite(tree, min(g - 1, x + 4), y, 3)
        fire.ignite(tree, max(0, x - 4), y, 3)
    for _ in range(40):
        fire.step(tree, (0.9, 0.0), [], rng)
    assert all(tree.locate(i % g, i // g).state == UNBURNED for i in water)


def test_fire_spreads_downwind_more_than_upwind(sim):
    fire, g = sim.fire, sim.terrain.grid
    up = down = 0
    for _ in range(40):
        x, y = random.Random(_).randrange(10, g - 10), random.Random(_ + 99).randrange(10, g - 10)
        down += fire.spread_probability(x, y, 1, 0, 1.0, (0.8, 0.0), [])
        up += fire.spread_probability(x, y, -1, 0, 1.0, (0.8, 0.0), [])
    assert down > 3 * up


def test_rain_extinguishes_fire(sim):
    terrain, fire, g = sim.terrain, sim.fire, sim.terrain.grid
    tree = RegionQuadtree(8, 2, terrain.cell_payload)
    cx, cy = next((i % g, i // g) for i in range(g * g)
                  if 40 < i % g < g - 40 and 40 < i // g < g - 40 and terrain.fine_veg[i] > 0.5)
    lit = fire.ignite(tree, cx, cy, 3)
    assert lit
    rain = Rain(cx, cy, 12, 50, 50)
    rng = random.Random(3)
    for _ in range(30):
        fire.step(tree, (0.0, 0.0), [rain], rng)
    assert not tree.burning
    assert any(leaf.state == EXTINGUISHED for leaf in tree.range_query(rain.bbox()))
    burned_area = sum(leaf.size ** 2 for leaf in tree.leaves() if leaf.state != UNBURNED)
    assert burned_area < 3.15 * 12 ** 2                # contained inside the rain cell


def test_simulation_is_deterministic_for_a_seed():
    def run():
        s = Simulation(seed=11)
        s.ignite(200, 260)
        s.evacuate(0, 40)
        for _ in range(25):
            s.step()
        snap = s.snapshot()
        return snap["leaves"], snap["agents"]
    assert run() == run()


def test_fire_model_uses_terrain_payload(sim):
    assert isinstance(sim.fire, FireModel)
    m, v = sim.terrain.cell_payload(0, 0, sim.terrain.grid)
    assert 0.0 <= m <= 1.0 and 0.0 <= v <= 1.0
