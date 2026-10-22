"""Rolling counters that relate tick cost to fire-front size (served at /api/profile)."""

from __future__ import annotations

from collections import deque


class Profiler:
    """Fixed-window series of per-tick timings and workload sizes."""

    SERIES = ("tick_times", "frontier_sizes", "reroute_times", "burning_counts", "leaf_counts")

    def __init__(self, window: int) -> None:
        self.window = window
        self._series = {name: deque(maxlen=window) for name in self.SERIES}

    def record(self, tick_ms: float, frontier: int, reroute_ms: float,
               burning: int, leaves: int) -> None:
        """Append one tick's measurements."""
        values = (round(tick_ms, 3), frontier, round(reroute_ms, 3), burning, leaves)
        for name, value in zip(self.SERIES, values):
            self._series[name].append(value)

    def last(self, name: str, default: float = 0) -> float:
        """Most recent value of a series."""
        series = self._series[name]
        return series[-1] if series else default

    def snapshot(self) -> dict:
        """All series as lists plus summary statistics."""
        out: dict = {name: list(values) for name, values in self._series.items()}
        ticks = out["tick_times"]
        reroutes = out["reroute_times"]
        out["summary"] = {
            "samples": len(ticks),
            "tick_ms_avg": round(sum(ticks) / len(ticks), 3) if ticks else 0.0,
            "tick_ms_max": max(ticks, default=0.0),
            "reroute_ms_max": max(reroutes, default=0.0),
        }
        return out
