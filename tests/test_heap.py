"""Binary min-heap ordering and decrease_key."""

import random

import pytest

from firebreak.heap import BinaryMinHeap


def test_pops_in_sorted_order():
    rng = random.Random(5)
    heap = BinaryMinHeap()
    values = [rng.uniform(0, 100) for _ in range(300)]
    for i, v in enumerate(values):
        heap.push(i, v)
    assert len(heap) == 300
    popped = [heap.pop()[1] for _ in range(300)]
    assert popped == sorted(values)
    assert not heap


def test_decrease_key_moves_item_to_top():
    heap = BinaryMinHeap()
    for key, p in (("a", 5), ("b", 3), ("c", 8), ("d", 6)):
        heap.push(key, p)
    heap.decrease_key("c", 1)
    assert heap.peek() == ("c", 1)
    assert heap.priority("d") == 6
    assert [heap.pop()[0] for _ in range(4)] == ["c", "b", "a", "d"]


def test_random_decrease_keys_keep_heap_valid():
    rng = random.Random(9)
    heap, ref = BinaryMinHeap(), {}
    for k in range(200):
        ref[k] = rng.uniform(50, 100)
        heap.push(k, ref[k])
    for _ in range(300):
        k = rng.randrange(200)
        ref[k] -= rng.uniform(0, 10)
        heap.decrease_key(k, ref[k])
    out = [heap.pop() for _ in range(200)]
    assert [p for _, p in out] == sorted(ref.values())
    assert all(ref[k] == p for k, p in out)


def test_ties_break_by_insertion_order_and_errors():
    heap = BinaryMinHeap()
    for key in "xyz":
        heap.push(key, 1.0)
    assert [heap.pop()[0] for _ in range(3)] == ["x", "y", "z"]
    with pytest.raises(IndexError):
        heap.pop()
    heap.push("q", 2.0)
    with pytest.raises(ValueError):
        heap.decrease_key("q", 3.0)
    with pytest.raises(KeyError):
        heap.push("q", 1.0)
    assert "q" in heap and "x" not in heap
