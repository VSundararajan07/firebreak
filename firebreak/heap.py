"""Binary min-heap with an index map, supporting ``decrease_key`` in O(log n).

Written from scratch (no ``heapq``). Each entry is ``[priority, sequence, key]``;
the insertion sequence breaks ties so ordering is deterministic.
"""

from __future__ import annotations

from typing import Hashable


class BinaryMinHeap:
    """Array-backed binary min-heap keyed by hashable items."""

    def __init__(self) -> None:
        self._items: list[list] = []
        self._index: dict[Hashable, int] = {}
        self._seq = 0

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._index

    def priority(self, key: Hashable) -> float:
        """Return the current priority of ``key``."""
        return self._items[self._index[key]][0]

    def push(self, key: Hashable, priority: float) -> None:
        """Insert ``key`` with ``priority``. O(log n)."""
        if key in self._index:
            raise KeyError(f"{key!r} already in heap; use decrease_key")
        self._items.append([priority, self._seq, key])
        self._seq += 1
        self._index[key] = len(self._items) - 1
        self._sift_up(len(self._items) - 1)

    def peek(self) -> tuple[Hashable, float]:
        """Return ``(key, priority)`` of the minimum without removing it."""
        if not self._items:
            raise IndexError("peek from empty heap")
        priority, _, key = self._items[0]
        return key, priority

    def pop(self) -> tuple[Hashable, float]:
        """Remove and return ``(key, priority)`` of the minimum. O(log n)."""
        if not self._items:
            raise IndexError("pop from empty heap")
        top = self._items[0]
        last = self._items.pop()
        del self._index[top[2]]
        if self._items:
            self._items[0] = last
            self._index[last[2]] = 0
            self._sift_down(0)
        return top[2], top[0]

    def decrease_key(self, key: Hashable, priority: float) -> None:
        """Lower the priority of ``key``. O(log n) thanks to the index map."""
        i = self._index[key]
        if priority > self._items[i][0]:
            raise ValueError("new priority is larger than the current one")
        self._items[i][0] = priority
        self._sift_up(i)

    def _less(self, a: int, b: int) -> bool:
        ia, ib = self._items[a], self._items[b]
        return (ia[0], ia[1]) < (ib[0], ib[1])

    def _swap(self, a: int, b: int) -> None:
        items = self._items
        items[a], items[b] = items[b], items[a]
        self._index[items[a][2]] = a
        self._index[items[b][2]] = b

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = (i - 1) >> 1
            if not self._less(i, parent):
                return
            self._swap(i, parent)
            i = parent

    def _sift_down(self, i: int) -> None:
        n = len(self._items)
        while True:
            left = 2 * i + 1
            smallest = i
            if left < n and self._less(left, smallest):
                smallest = left
            if left + 1 < n and self._less(left + 1, smallest):
                smallest = left + 1
            if smallest == i:
                return
            self._swap(i, smallest)
            i = smallest
