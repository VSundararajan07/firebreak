"""Fenwick tree updates and sums against a brute-force list."""

import random

import pytest

from firebreak.fenwick import FenwickTree


def test_random_updates_match_brute_force():
    rng = random.Random(42)
    n = 97
    ft, brute = FenwickTree(n), [0] * (n + 1)
    for _ in range(600):
        i, delta = rng.randint(1, n), rng.randint(-3, 5)
        ft.point_add(i, delta)
        brute[i] += delta
        left = rng.randint(1, n)
        right = rng.randint(left, n)
        assert ft.range_sum(left, right) == sum(brute[left:right + 1])
        k = rng.randint(0, n)
        assert ft.prefix_sum(k) == sum(brute[1:k + 1])


def test_from_values_linear_build():
    values = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    ft = FenwickTree.from_values(values)
    assert len(ft) == 10
    assert [ft.range_sum(i, i) for i in range(1, 11)] == values
    assert ft.prefix_sum(10) == sum(values)
    assert ft.range_sum(3, 6) == 4 + 1 + 5 + 9


def test_vehicle_enter_and_leave_restores_zero():
    ft = FenwickTree(8)
    for slot in (2, 2, 5, 8):
        ft.point_add(slot, 1)
    assert ft.range_sum(1, 8) == 4
    assert ft.range_sum(2, 2) == 2
    for slot in (2, 2, 5, 8):
        ft.point_add(slot, -1)
    assert ft.prefix_sum(8) == 0


def test_bounds_are_checked():
    ft = FenwickTree(4)
    with pytest.raises(IndexError):
        ft.point_add(0, 1)
    with pytest.raises(IndexError):
        ft.range_sum(1, 5)
    with pytest.raises(ValueError):
        ft.range_sum(3, 2)
    assert ft.prefix_sum(0) == 0
