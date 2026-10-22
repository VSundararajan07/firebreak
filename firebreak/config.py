"""All tunable constants for Firebreak in one place.

Units: *px* are terrain pixels (what the browser draws), *grid units* are the
finest quadtree cells (``CELL_PX`` pixels wide). One tick is one simulation step.
"""

# --- World & quadtree -------------------------------------------------------
MAX_DEPTH = 8                       # quadtree depth; finest leaf = 1 grid unit
MIN_DEPTH = 2                       # leaves never merge above this depth (4x4 base grid)
GRID = 1 << MAX_DEPTH               # 256 grid units per side
CELL_PX = 2                         # pixels per grid unit
WORLD_PX = GRID * CELL_PX           # 512 terrain pixels per side

# --- Timing -----------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_TPS = 4                     # simulation ticks per second
MIN_TPS = 1
MAX_TPS = 20
SIM_SECONDS_PER_TICK = 20           # simulated time represented by one tick

# --- Fire model -------------------------------------------------------------
P_BASE = 0.5                        # spread probability for full fuel, no wind/slope
MOISTURE_DAMPING = 0.75             # fraction of P removed by fully saturated fuel
WIND_GAIN = 2.0                     # exp(WIND_GAIN * strength * cos(angle))
SLOPE_GAIN = 30.0                   # exp(SLOPE_GAIN * elevation rise per grid unit)
DIAGONAL_FACTOR = 0.7               # diagonal neighbours are farther away
MAX_SPREAD_P = 0.95
MIN_FUEL = 0.03                     # vegetation below this never ignites
BURN_TICKS_MIN = 3                  # burn duration = MIN + round(VEG * vegetation)
BURN_TICKS_VEG = 5
IGNITE_RADIUS = 2                   # grid units ignited by one click
DEFAULT_WIND_DEG = 60.0             # compass bearing the wind blows TOWARD (0 = north/up)
DEFAULT_WIND_STRENGTH = 0.55

# --- Rain -------------------------------------------------------------------
RAIN_TICKS = 40                     # lifetime of a rain cell
RAIN_EXTINGUISH_P = 0.35            # per tick, per burning cell under rain
RAIN_SPREAD_FACTOR = 0.04           # spread probability multiplier into rained cells
DEFAULT_RAIN_RADIUS_PX = 40
MAX_RAIN_RADIUS_PX = 150

# --- Terrain & roads --------------------------------------------------------
ROAD_LATTICE = 9                    # lattice points per side (outer ring = shelter slots)
SHELTER_COUNT = 5
TOWN_COUNT = 3
TOWN_NAMES = ("Ashford", "Pinecrest", "Cedar Falls")
ROAD_SEGMENT_PX = 12                # Fenwick granularity: one slot per road segment
ROAD_BURN_RADIUS = 1                # grid units around a road whose ignition closes it
THREAT_RADIUS = 4                   # grid units around a road where fire raises its cost
WIND_THREAT_REACH = 10              # grid units downwind that a burning cell threatens

# --- Routing & evacuees -----------------------------------------------------
CONGESTION_WEIGHT = 0.08            # cost = base * (1 + w * vehicles on road) * fire term
FIRE_WEIGHT = 0.5                   # per unit of threat, capped by THREAT_CAP
THREAT_CAP = 12
CONGESTION_BUCKET = 6               # an edge goes dirty when vehicles // bucket changes
AGENT_SPEED_PX = 7.0                # pixels per tick on an empty road
SPEED_CONGESTION_K = 0.3            # speed /= 1 + k * other vehicles on the same segment
DEPART_PER_TICK = 6                 # vehicles leaving each town per tick
MAX_EVACUATE = 1000                 # largest single evacuation request
STRANDED_RETRY_TICKS = 8            # stranded vehicles retry routing at least this often
REROUTE_HYSTERESIS = 1.15           # keep a still-open route unless new one is >15% cheaper

# --- Profiler ---------------------------------------------------------------
PROFILE_WINDOW = 400                # samples kept per series
