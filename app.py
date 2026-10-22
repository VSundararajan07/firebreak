"""Firebreak entrypoint: FastAPI JSON API + static frontend, launched with uvicorn.

Run ``python -m app`` (optionally ``--seed N --port 8000 --tps 4``) and open
http://localhost:8000. The simulation runs in a background thread; request
handlers are plain ``def`` functions, so FastAPI runs them in its threadpool
and the event loop never blocks on simulation work.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from firebreak import config as C
from firebreak.simulator import Simulation

STATIC_DIR = Path(__file__).resolve().parent / "static"
SETTINGS = {"seed": C.DEFAULT_SEED, "tps": C.DEFAULT_TPS}


class Engine:
    """Owns the simulation, its lock, the tick thread and a per-version state cache."""

    def __init__(self, seed: int, tps: float) -> None:
        self.lock = threading.RLock()
        self.sim = Simulation(seed)
        self.tps = float(tps)
        self.paused = False
        self.version = 0
        self.generation = 0
        self._cache: tuple[int, bytes] = (-1, b"")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="sim-tick", daemon=True)

    def start(self) -> None:
        """Start the background tick thread."""
        self._thread.start()

    def stop(self) -> None:
        """Ask the tick thread to exit and wait briefly for it."""
        self._stop.set()
        self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            began = time.perf_counter()
            with self.lock:
                if not self.paused:
                    self.sim.step()
                    self.version += 1
                interval = 1.0 / self.tps
            self._stop.wait(max(0.002, interval - (time.perf_counter() - began)))

    def command(self, fn):
        """Run ``fn(sim)`` under the lock and publish a new state version."""
        with self.lock:
            result = fn(self.sim)
            self.version += 1
            return result

    def configure(self, **settings) -> None:
        """Atomically change engine settings (``paused``, ``tps``) and publish."""
        with self.lock:
            for name, value in settings.items():
                setattr(self, name, value)
            self.version += 1

    def reset(self, seed: int) -> None:
        """Build a fresh world (outside the lock), then swap it in."""
        sim = Simulation(seed)
        with self.lock:
            self.sim = sim
            self.generation += 1
            self.version += 1

    def state_bytes(self, since: int | None) -> bytes:
        """Serialized state, or a tiny ``unchanged`` marker if the client is current."""
        with self.lock:
            if since == self.version:
                return b'{"unchanged":true,"version":%d}' % self.version
            if self._cache[0] != self.version:
                state = self.sim.snapshot()
                state.update(version=self.version, generation=self.generation,
                             paused=self.paused, tps=self.tps,
                             terrain={"seed": self.sim.seed, "size_px": C.WORLD_PX,
                                      "roads": len(self.sim.roads.edges),
                                      "towns": len(self.sim.terrain.towns)})
                self._cache = (self.version, json.dumps(state, separators=(",", ":")).encode())
            return self._cache[1]

    def terrain_meta(self) -> dict:
        """Static description of the current world (geometry, towns, shelters)."""
        with self.lock:
            t = self.sim.terrain
            return {
                "seed": self.sim.seed, "generation": self.generation,
                "size_px": t.size, "grid": t.grid, "cell_px": t.cell_px,
                "max_depth": C.MAX_DEPTH, "min_depth": C.MIN_DEPTH,
                "nodes": [[round(x, 1), round(y, 1)] for x, y in t.nodes],
                "edges": [[u, v, [round(c, 1) for p in pts for c in p]] for u, v, pts in t.edges],
                "shelters": t.shelters,
                "towns": [{"id": tw["id"], "name": tw["name"], "node": tw["node"],
                           "x": round(tw["x"], 1), "y": round(tw["y"], 1)} for tw in t.towns],
                "default_rain_radius": C.DEFAULT_RAIN_RADIUS_PX,
                "max_rain_radius": C.MAX_RAIN_RADIUS_PX,
                "min_tps": C.MIN_TPS, "max_tps": C.MAX_TPS,
            }


# ------------------------------------------------------------------ request bodies
class IgniteBody(BaseModel):
    """Ignition point in terrain pixels."""

    x: float = Field(ge=0, lt=C.WORLD_PX)
    y: float = Field(ge=0, lt=C.WORLD_PX)


class WindBody(BaseModel):
    """Bearing the wind blows toward (0 = north/up, 90 = east) and its strength."""

    direction_deg: float
    strength_0_to_1: float = Field(ge=0, le=1)


class RainBody(BaseModel):
    """Rain cell centre and radius in terrain pixels."""

    x: float = Field(ge=0, lt=C.WORLD_PX)
    y: float = Field(ge=0, lt=C.WORLD_PX)
    radius: float = Field(default=C.DEFAULT_RAIN_RADIUS_PX, gt=0, le=C.MAX_RAIN_RADIUS_PX)


class EvacuateBody(BaseModel):
    """Town index (see /api/terrain) and number of vehicles to evacuate."""

    town_id: int = Field(ge=0)
    count: int = Field(gt=0, le=C.MAX_EVACUATE)


class ResetBody(BaseModel):
    """New world seed; omitted keeps the current seed."""

    seed: int | None = None


class SpeedBody(BaseModel):
    """Simulation ticks per second."""

    tps: float = Field(ge=C.MIN_TPS, le=C.MAX_TPS)


# ------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = Engine(SETTINGS["seed"], SETTINGS["tps"])
    app.state.engine = engine
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="Firebreak", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2048)


def _engine(request: Request) -> Engine:
    return request.app.state.engine


def _ok(**extra) -> dict:
    return {"ok": True, **extra}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The single-page frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/terrain")
def terrain(request: Request) -> dict:
    """World geometry: road polylines, towns, shelters, sizes."""
    return _engine(request).terrain_meta()


@app.get("/api/terrain.bin")
def terrain_image(request: Request) -> Response:
    """Raw RGBA bytes (size_px * size_px * 4) of the shaded landscape."""
    engine = _engine(request)
    with engine.lock:
        terrain_obj = engine.sim.terrain
    return Response(terrain_obj.render_rgba(), media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def state(request: Request, since: int | None = None) -> Response:
    """Full state JSON (packed quadtree leaves, roads, routes, agents, stats)."""
    return Response(_engine(request).state_bytes(since), media_type="application/json")


@app.get("/api/profile")
def profile(request: Request) -> dict:
    """Rolling tick times, frontier sizes and reroute times."""
    engine = _engine(request)
    with engine.lock:
        return engine.sim.profiler.snapshot()


@app.post("/api/ignite")
def ignite(body: IgniteBody, request: Request) -> dict:
    """Start a fire at (x, y)."""
    return _ok(cells=_engine(request).command(lambda s: s.ignite(body.x, body.y)))


@app.post("/api/wind")
def wind(body: WindBody, request: Request) -> dict:
    """Change wind direction and strength."""
    _engine(request).command(lambda s: s.set_wind(body.direction_deg, body.strength_0_to_1))
    return _ok()


@app.post("/api/rain")
def rain(body: RainBody, request: Request) -> dict:
    """Drop a rain cell that douses burning cells over time."""
    _engine(request).command(lambda s: s.add_rain(body.x, body.y, body.radius))
    return _ok()


@app.post("/api/evacuate")
def evacuate(body: EvacuateBody, request: Request) -> dict:
    """Queue vehicles leaving a town toward the nearest reachable shelter."""
    engine = _engine(request)
    with engine.lock:
        if body.town_id >= len(engine.sim.terrain.towns):
            raise HTTPException(status_code=404, detail="unknown town_id")
        return _ok(queued=engine.command(lambda s: s.evacuate(body.town_id, body.count)))


@app.post("/api/pause")
def pause(request: Request) -> dict:
    """Stop ticking (commands still apply)."""
    _engine(request).configure(paused=True)
    return _ok(paused=True)


@app.post("/api/resume")
def resume(request: Request) -> dict:
    """Resume ticking."""
    _engine(request).configure(paused=False)
    return _ok(paused=False)


@app.post("/api/speed")
def speed(body: SpeedBody, request: Request) -> dict:
    """Set ticks per second."""
    _engine(request).configure(tps=body.tps)
    return _ok(tps=body.tps)


@app.post("/api/reset")
def reset(request: Request, body: ResetBody | None = None) -> dict:
    """Regenerate the world from a seed."""
    engine = _engine(request)
    seed = body.seed if body is not None and body.seed is not None else engine.sim.seed
    engine.reset(seed)
    return _ok(seed=seed)


def main() -> None:
    """Parse CLI flags and serve on http://localhost:<port>."""
    parser = argparse.ArgumentParser(description="Firebreak wildfire & evacuation simulator")
    parser.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    parser.add_argument("--tps", type=float, default=C.DEFAULT_TPS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    SETTINGS.update(seed=args.seed, tps=min(max(args.tps, C.MIN_TPS), C.MAX_TPS))
    print(f"Firebreak running at http://localhost:{args.port}  (seed {args.seed})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
