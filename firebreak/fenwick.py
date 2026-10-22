"""Fenwick (binary indexed) tree over road segments.

Index ``i`` (1-based) is one road segment; its value is the number of vehicles
on it. Point updates and prefix/range sums are all O(log n).
"""

from __future__ import annotations

from typing import Iterable


class FenwickTree:
    """1-indexed binary indexed tree of integer or float counts."""

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        self._n = size
        self._tree = [0] * (size + 1)

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "FenwickTree":
        """Build in O(n) from ``values`` where ``values[0]`` is index 1."""
        vals = list(values)
        ft = cls(len(vals))
        tree = ft._tree
        for i, v in enumerate(vals, start=1):
            tree[i] += v
            parent = i + (i & -i)
            if parent <= ft._n:
                tree[parent] += tree[i]
        return ft

    def __len__(self) -> int:
        return self._n

    def point_add(self, i: int, delta: float) -> None:
        """Add ``delta`` at index ``i`` (vehicle enters: +1, leaves: -1)."""
        self._check(i)
        tree, n = self._tree, self._n
        while i <= n:
            tree[i] += delta
            i += i & -i

    def prefix_sum(self, i: int) -> float:
        """Sum of indices ``1..i``; ``prefix_sum(0) == 0``."""
        if not 0 <= i <= self._n:
            raise IndexError(f"prefix index {i} out of range 0..{self._n}")
        tree, total = self._tree, 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    def range_sum(self, left: int, right: int) -> float:
        """Sum of indices ``left..right`` inclusive."""
        self._check(left)
        self._check(right)
        if left > right:
            raise ValueError("left must be <= right")
        return self.prefix_sum(right) - self.prefix_sum(left - 1)

    def _check(self, i: int) -> None:
        if not 1 <= i <= self._n:
            raise IndexError(f"index {i} out of range 1..{self._n}")
