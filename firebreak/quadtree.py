"""Region quadtree holding per-cell fire state at adaptive resolution.

The tree covers a square of ``2**max_depth`` grid units. Leaves carry the
payload (fire state, moisture, vegetation, burn timer). Burning cells always
live at ``max_depth`` (unit cells); uniform quiet regions merge back into
coarse leaves. Burning leaves are also kept in an insertion-ordered registry,
so frontier work is proportional to the fire front, never to the map area.
"""

from __future__ import annotations

from typing import Callable, Iterator

UNBURNED, BURNING, BURNED, EXTINGUISHED = 0, 1, 2, 3
STATE_NAMES = ("UNBURNED", "BURNING", "BURNED", "EXTINGUISHED")

_DIAG = 2 ** 0.5
NEIGHBOR_OFFSETS = (
    (-1, -1, _DIAG), (0, -1, 1.0), (1, -1, _DIAG),
    (-1, 0, 1.0), (1, 0, 1.0),
    (-1, 1, _DIAG), (0, 1, 1.0), (1, 1, _DIAG),
)

PayloadFn = Callable[[int, int, int], "tuple[float, float]"]


def _zero_payload(x: int, y: int, size: int) -> tuple[float, float]:
    """Default payload: no moisture, no vegetation."""
    return 0.0, 0.0


class QuadNode:
    """A quadtree node; a leaf when ``children`` is None.

    Children are ordered NW, NE, SW, SE. ``threats`` is a slot the simulator
    uses to remember which roads a burning cell is endangering.
    """

    __slots__ = ("x", "y", "size", "depth", "parent", "children", "state",
                 "moisture", "vegetation", "burn_left", "threats")

    def __init__(self, x: int, y: int, size: int, depth: int, parent: "QuadNode | None",
                 state: int, moisture: float, vegetation: float) -> None:
        self.x = x
        self.y = y
        self.size = size
        self.depth = depth
        self.parent = parent
        self.children: list[QuadNode] | None = None
        self.state = state
        self.moisture = moisture
        self.vegetation = vegetation
        self.burn_left = 0
        self.threats: tuple[int, ...] = ()

    @property
    def is_leaf(self) -> bool:
        """True when the node has no children."""
        return self.children is None

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` in grid units."""
        return self.x, self.y, self.size, self.size

    def contains(self, px: float, py: float) -> bool:
        """True if point ``(px, py)`` lies inside this node (half-open bounds)."""
        return self.x <= px < self.x + self.size and self.y <= py < self.y + self.size

    def intersects(self, x: float, y: float, w: float, h: float) -> bool:
        """True if this node overlaps the half-open box ``(x, y, w, h)``."""
        return (self.x < x + w and x < self.x + self.size
                and self.y < y + h and y < self.y + self.size)

    def __repr__(self) -> str:
        return (f"QuadNode({self.x},{self.y},size={self.size},depth={self.depth},"
                f"{STATE_NAMES[self.state]})")


class RegionQuadtree:
    """Adaptive region quadtree: insert, subdivide, merge, range and frontier queries."""

    def __init__(self, max_depth: int = 8, min_depth: int = 0,
                 payload: PayloadFn | None = None) -> None:
        if not 0 <= min_depth <= max_depth:
            raise ValueError("need 0 <= min_depth <= max_depth")
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.size = 1 << max_depth
        self._payload = payload or _zero_payload
        moisture, vegetation = self._payload(0, 0, self.size)
        self.root = QuadNode(0, 0, self.size, 0, None, UNBURNED, moisture, vegetation)
        self.burning: dict[QuadNode, None] = {}
        self._leaf_total = 1
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.depth < min_depth:
                stack.extend(self.subdivide(node))

    # ------------------------------------------------------------------ queries
    def in_bounds(self, x: int, y: int) -> bool:
        """True if grid point ``(x, y)`` is inside the tree."""
        return 0 <= x < self.size and 0 <= y < self.size

    def locate(self, x: int, y: int, start: QuadNode | None = None) -> QuadNode | None:
        """Return the leaf containing ``(x, y)``, or None when outside the tree.

        Starting from a nearby leaf (``start``) climbs only to the lowest common
        ancestor before descending, so neighbour lookups are O(1) on average.
        """
        if not (0 <= x < self.size and 0 <= y < self.size):
            return None
        node = start if start is not None else self.root
        while not (node.x <= x < node.x + node.size and node.y <= y < node.y + node.size):
            node = node.parent
        while node.children is not None:
            half = node.size >> 1
            i = 1 if x >= node.x + half else 0
            if y >= node.y + half:
                i += 2
            node = node.children[i]
        return node

    def neighbor(self, leaf: QuadNode, dx: int, dy: int) -> QuadNode | None:
        """Leaf containing the cell offset by ``(dx, dy)`` from ``leaf``'s corner."""
        return self.locate(leaf.x + dx, leaf.y + dy, leaf)

    def unburned_neighbors(self, leaf: QuadNode) -> list[tuple[int, int, float, QuadNode]]:
        """``(dx, dy, distance, leaf)`` for each UNBURNED 8-neighbour of a unit leaf."""
        out = []
        for dx, dy, dist in NEIGHBOR_OFFSETS:
            nb = self.locate(leaf.x + dx, leaf.y + dy, leaf)
            if nb is not None and nb.state == UNBURNED:
                out.append((dx, dy, dist, nb))
        return out

    def frontier_cells(self) -> list[QuadNode]:
        """Unique UNBURNED leaves adjacent to a BURNING leaf.

        Iterates only the burning registry: O(burning cells), independent of area.
        """
        seen: dict[QuadNode, None] = {}
        for leaf in self.burning:
            for _, _, _, nb in self.unburned_neighbors(leaf):
                seen[nb] = None
        return list(seen)

    def range_query(self, bbox: tuple[float, float, float, float]) -> list[QuadNode]:
        """All leaves intersecting the half-open box ``(x, y, w, h)``."""
        x, y, w, h = bbox
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if not node.intersects(x, y, w, h):
                continue
            if node.children is None:
                out.append(node)
            else:
                stack.extend(node.children)
        return out

    def leaves(self) -> Iterator[QuadNode]:
        """Iterate over every leaf (depth-first)."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.children is None:
                yield node
            else:
                stack.extend(node.children)

    def leaf_count(self) -> int:
        """Number of leaves currently in the tree (maintained incrementally, O(1))."""
        return self._leaf_total

    def packed_leaves(self) -> list[int]:
        """Every leaf packed into one int: x | y | level | state | heat.

        With ``b = max_depth`` bits per coordinate the layout is
        ``x`` [0,b), ``y`` [b,2b), ``level=log2(size)`` [2b,2b+4),
        ``state`` [2b+4,2b+6), ``heat=min(burn_left,7)`` [2b+6,2b+9).
        """
        b = self.max_depth
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if node.children is not None:
                stack.extend(node.children)
                continue
            heat = node.burn_left if node.burn_left < 7 else 7
            out.append(node.x | (node.y << b) | ((b - node.depth) << (2 * b))
                       | (node.state << (2 * b + 4)) | (heat << (2 * b + 6)))
        return out

    # ---------------------------------------------------------------- mutation
    def subdivide(self, node: QuadNode) -> list[QuadNode]:
        """Split a leaf into four children that inherit its state."""
        if node.children is not None:
            raise ValueError("node is already subdivided")
        if node.depth >= self.max_depth:
            raise ValueError("node is at max depth")
        half, depth = node.size >> 1, node.depth + 1
        kids = []
        for ox, oy in ((0, 0), (half, 0), (0, half), (half, half)):
            cx, cy = node.x + ox, node.y + oy
            moisture, vegetation = self._payload(cx, cy, half)
            child = QuadNode(cx, cy, half, depth, node, node.state, moisture, vegetation)
            child.burn_left = node.burn_left
            kids.append(child)
        node.children = kids
        self._leaf_total += 3
        if node in self.burning:
            del self.burning[node]
            for child in kids:
                self.burning[child] = None
        return kids

    def merge_children(self, node: QuadNode) -> bool:
        """Collapse four leaf children sharing one non-burning state into ``node``.

        Burning cells never merge so the active front stays at full resolution.
        """
        kids = node.children
        if kids is None or node.depth < self.min_depth:
            return False
        state = kids[0].state
        if state == BURNING:
            return False
        for kid in kids:
            if kid.children is not None or kid.state != state:
                return False
        node.children = None
        self._leaf_total -= 3
        node.state = state
        node.burn_left = 0
        node.moisture = sum(k.moisture for k in kids) / 4.0
        node.vegetation = sum(k.vegetation for k in kids) / 4.0
        return True

    def merge_upward(self, leaf: QuadNode) -> int:
        """Merge ancestors of ``leaf`` bottom-up while possible; return merge count."""
        merges, node = 0, leaf.parent
        while node is not None and self.merge_children(node):
            merges += 1
            node = node.parent
        return merges

    def insert(self, x: int, y: int, state: int, depth: int | None = None,
               burn_left: int = 0) -> QuadNode:
        """Set the cell at ``(x, y)`` and ``depth`` (default max) to ``state``.

        Coarser leaves on the way down are subdivided; a finer subtree at the
        target depth is collapsed first. Returns the affected leaf.
        """
        depth = self.max_depth if depth is None else depth
        if not self.min_depth <= depth <= self.max_depth:
            raise ValueError(f"depth must be in {self.min_depth}..{self.max_depth}")
        node = self.locate(x, y)
        if node is None:
            raise ValueError(f"point ({x}, {y}) outside the tree")
        while node.depth > depth:
            node = node.parent
        if node.children is not None:
            self._collapse(node)
        while node.depth < depth:
            half = node.size >> 1
            i = (1 if x >= node.x + half else 0) + (2 if y >= node.y + half else 0)
            node = self.subdivide(node)[i]
        self.set_state(node, state, burn_left)
        return node

    def set_state(self, leaf: QuadNode, state: int, burn_left: int = 0) -> None:
        """Change a leaf's state, keeping the burning registry consistent."""
        if leaf.children is not None:
            raise ValueError("set_state requires a leaf")
        leaf.state = state
        if state == BURNING:
            leaf.burn_left = max(1, burn_left)
            self.burning[leaf] = None
        else:
            leaf.burn_left = 0
            self.burning.pop(leaf, None)

    def _collapse(self, node: QuadNode) -> None:
        """Drop ``node``'s subtree, unregistering any burning descendants."""
        stack = list(node.children or ())
        while stack:
            child = stack.pop()
            if child.children is None:
                self.burning.pop(child, None)
                self._leaf_total -= 1
            else:
                stack.extend(child.children)
        node.children = None
        self._leaf_total += 1
