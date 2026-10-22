# Firebreak 🔥 — Wildfire Spread & Evacuation Simulator

An interactive wildfire and evacuation simulator for an **Advanced Data Structures** course.
The fire spreads over a **region quadtree** that refines itself at the fire front. Road
congestion is tracked with a **Fenwick tree**. Evacuees are routed with **Dijkstra + a
hand-written binary min-heap**, and they reroute live when roads burn or jam. Every data
structure is written from scratch: no scipy, networkx, shapely or `heapq`.

---

## Run it (≈1 minute)

Requires **Python 3.10+**.

```bash
cd firebreak
pip install -r requirements.txt
python -m app
```

Open **http://localhost:8000**. On Windows, if `python` is not on your PATH, use `py` instead
(`py -m pip install -r requirements.txt`, then `py -m app`).

Options: `python -m app --seed 7 --tps 4 --port 8000`. You can also open
`http://localhost:8000/?seed=7` to generate the world for seed 7.

Run the tests with `pytest` (or `python -m pytest`) from the `firebreak/` folder.

### Controls

| Action | How |
|---|---|
| Ignite a fire | **Left-click** the map |
| Drop rain | **Shift + click** (radius slider in *Tools*) |
| Wind | Drag the **dial** (direction it blows *toward*); **Strength** slider |
| Evacuate | Choose a town and a vehicle count, then **Evacuate** (or **Evacuate all towns**) |
| Speed / pause | **Speed** slider (1–20 ticks/s), **Pause/Resume** |
| New world | Type a **Seed** and press **Reset** |
| Quadtree overlay | *View → Quadtree overlay*. Blue = coarse leaves, amber = depth-8 cells at the front |

**Demo script:** evacuate all towns, then ignite a forested hillside between two towns. Turn
on the quadtree overlay and watch cells refine at the front while burned interiors merge
into big blocks. Swing the wind dial: the fire turns, burned roads (dark red, dashed) drop
out of the graph, and routes flash white as vehicles reroute. Shift-click rain onto the
front: cells turn blue-grey (extinguished) and merge back together in the overlay.

---

## Architecture

```
firebreak/
├── app.py               FastAPI routes, background tick thread, state cache, uvicorn launch
├── firebreak/
│   ├── config.py        every tunable constant (grid, tick rate, fire, roads, routing)
│   ├── terrain.py       value-noise elevation/vegetation/moisture, river, towns, road lattice
│   ├── quadtree.py      RegionQuadtree: insert, subdivide, merge, range_query, frontier_cells
│   ├── fenwick.py       FenwickTree: point_add, prefix_sum, range_sum (+ O(n) build)
│   ├── heap.py          BinaryMinHeap: push, pop, peek, decrease_key (index map)
│   ├── routing.py       RoadGraph, Dijkstra, edge costs (congestion × fire), road–cell index
│   ├── fire_model.py    spread probability (fuel, moisture, slope, wind, rain) over the quadtree
│   ├── simulator.py     tick orchestration, vehicles, threat tracking, dirty-edge rerouting
│   └── profiler.py      rolling tick-time / frontier-size / reroute-time series
├── static/index.html    vanilla JS + Canvas UI (no frameworks, no CDN, works offline)
└── tests/               pytest suites for quadtree + fire, Fenwick, heap, routing
```

**One tick** (default 4 ticks/s, background thread):

1. **Fire** (`fire_model.step`): every burning unit cell is in the quadtree's burning
   registry. For each one, find its 8 neighbours with an O(1)-average quadtree neighbour
   lookup, roll ignition for the unburned ones, and count down its burn timer. Rain cells
   use `range_query` to find the burning cells they can douse. New fires are `insert`ed at
   depth 8, which subdivides coarse leaves along the way. Burned-out and doused cells call
   `merge_upward`, so uniform regions collapse back into coarse leaves.
2. **Roads**: an ignited cell looks up a precomputed *cell → roads* index. Roads under the
   fire are **closed** (removed from the graph). Roads within 4 cells of the fire, or
   downwind of it, gain **threat**, which multiplies their cost.
3. **Vehicles** move along their routes. Each road is split into 12 px slots, and every slot
   is one Fenwick index. Entering or leaving a slot is a `point_add(±1)`. Vehicle speed drops
   with slot occupancy, and road congestion is a `range_sum` over the road's slots.
4. **Rerouting**: a road becomes *dirty* when it closes, or when its threat or congestion
   bucket changes. Only vehicles whose *remaining* route (or current road) contains a dirty
   edge are rerouted. Dijkstra runs at most once per `(source node, town)` per tick, and its
   result is shared by every vehicle starting there. Hysteresis keeps a still-open route
   unless the new one is more than 15 % cheaper, which stops routes from flapping.

**Edge cost:** `length × (1 + 0.08 × range_sum(road slots)) × (1 + 0.5 × min(threat, 12))`.
Closed roads are skipped entirely.

**Spread probability** for a burning cell to ignite neighbour `t` this tick:
`0.5 · veg_t · (1 − 0.75·moist_t) · exp((2·wind·cosθ + 30·Δelev) / dist)`, with ×0.7 for
diagonals and ×0.04 inside rain. It is capped at 0.95. Water never ignites, and burned or
extinguished cells are never revisited.

### API

| Method | Path | Body / query |
|---|---|---|
| GET | `/api/state?since=V` | Full state. Returns `{"unchanged":true}` if the version is still `V` |
| GET | `/api/terrain`, `/api/terrain.bin` | Road geometry, towns and shelters; RGBA terrain image |
| GET | `/api/profile` | `tick_times`, `frontier_sizes`, `reroute_times`, `burning_counts`, `leaf_counts` |
| POST | `/api/ignite` | `{x, y}` (terrain pixels, 0–511) |
| POST | `/api/wind` | `{direction_deg, strength_0_to_1}` |
| POST | `/api/rain` | `{x, y, radius}` |
| POST | `/api/evacuate` | `{town_id, count}` |
| POST | `/api/pause`, `/api/resume`, `/api/speed` | `/api/speed` takes `{tps}` |
| POST | `/api/reset` | `{seed}` |

The state never sends pixels. Each quadtree leaf is packed into one integer
(`x | y<<8 | log2(size)<<16 | state<<20 | heat<<22`), so burned regions arrive as merged
rectangles. The server builds and caches the JSON once per version and gzips it.

---

## Complexity

| Structure / operation | Cost | Naive alternative |
|---|---|---|
| Quadtree `locate` from root | O(depth) = O(log N) | — |
| Neighbour lookup starting at a leaf | O(1) average (climb to the common ancestor) | O(log N) from root |
| Fire tick | **O(B)**, B = burning cells (the front), plus O(k·depth) for k new ignitions | O(N²) full-grid update |
| `frontier_cells()` | O(B), independent of map area | O(N²) scan |
| `range_query(bbox)` | O(log N + k) for k leaves returned | O(N²) |
| `merge_upward` | O(depth) per changed cell | — |
| Fenwick `point_add` / `prefix_sum` / `range_sum` | O(log n), n = road slots | O(1) update but O(n) range scan |
| Fenwick build | O(n) | — |
| Heap `push` / `pop` / `decrease_key` | O(log V) (index map makes `decrease_key` possible) | O(V) linear min |
| Dijkstra | **O((V + E) log V)**, early exit at the first shelter | Repeated BFS / all-pairs |
| Reroute pass | O(R·(V+E) log V), R = distinct (source, town) pairs among affected vehicles | Dijkstra for every vehicle, every tick |

The project's key claim is that **tick cost tracks the fire perimeter, not the map area**.
Two quadtree properties make this hold. Burned interiors merge into a handful of coarse
leaves, and only the burning registry is iterated. Fully enclosed burning cells (all
neighbours alight or non-flammable) are cached and skip their neighbour checks. The
*Profiler* panel plots tick time against front cells live; `/api/profile` exports the raw
series for report graphs.

### Measured performance

Measured on a Windows laptop with Python 3.13 on a 512×512 px terrain (256×256 grid,
quadtree depth 8), 8 fires and 400 evacuees:

| Metric | Target | Measured |
|---|---|---|
| Simulation tick | < 100 ms | avg 35 ms, p95 68 ms, max 84 ms |
| Full reroute of every vehicle, all edges dirty | < 300 ms | max 12 ms |
| `/api/state` payload | ≤ 200 KB | ≤ 68 KB raw, ~6–10 KB gzipped |
| Peak quadtree leaves (vs 65 536 unit cells) | — | 7 360 |

---

## Tests

`pytest` runs 26 tests with well over 15 assertions:

- **quadtree:** insert subdivides only along the path; min-depth pre-split; merge only when
  all four children are uniform and never while burning; `merge_upward` compaction;
  `range_query` vs brute force; `frontier_cells` vs brute force; out-of-bounds neighbours;
  subtree collapse.
- **fire:** never enters water; downwind spread beats upwind; rain extinguishes and
  contains a fire; same seed and same commands give an identical simulation.
- **Fenwick:** 600 random updates checked against brute-force prefix and range sums; O(n)
  build; enter/leave symmetry; bounds checks.
- **heap:** sorted pops, `decrease_key` (targeted and random), tie order, error cases.
- **routing:** Dijkstra on a hand-checked 6-node graph (including a `decrease_key`
  relaxation and an unreachable node); early exit at the nearest target; closed roads
  removed and congestion cost formula; burning a road reroutes every affected vehicle;
  vehicles reach shelters and the state stays under 200 KB.

---

## Determinism

All randomness comes from `random.Random(seed)` and `numpy.random.default_rng(seed)`.
Iteration orders are insertion-ordered dicts, sorted ids or fixed child order, so the same
seed and the same command sequence at the same ticks reproduce the run exactly (this is
tested). Commands from the browser land at wall-clock times, so interactive sessions are
only reproducible if commands are replayed at the same ticks.

## Limitations

- The fire model is a probabilistic cellular automaton, not a physical Rothermel
  rate-of-spread model. There is no spotting (embers jumping gaps) and no fuel moisture
  dynamics over time.
- Burn resolution is 2 px (one grid unit). Spread is at most one cell per tick, so the fire
  front speed is capped by the tick rate.
- Vehicles have no lanes, intersections or capacity limits. Congestion only slows them,
  through slot occupancy.
- The road network is a jittered lattice with curved roads, not real map data. Shelters
  sit on the map edge.
- Stranded vehicles wait for a route to reopen, but closed roads never reopen. The model
  does not include casualties or vehicles catching fire.
- One simulation runs per server. All browser tabs share and control the same world.
- The frontend polls at 20 Hz; WebSockets were out of scope. Vehicle motion is
  interpolated between ticks.
