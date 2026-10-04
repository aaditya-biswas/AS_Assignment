"""Grid world model: map generation, tasks, dynamic blockages, BFS distances.

Coordinates are ``(row, col)``.  Cell values: ``0`` free, ``1`` shelf.
See ``SPEC.md`` section 3.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

Cell = Tuple[int, int]

FREE = 0
SHELF = 1

# wait, down, up, right, left  (row, col deltas)
MV = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


class TaskStatus(Enum):
    PENDING = "PENDING"
    PICKED = "PICKED"
    DONE = "DONE"


@dataclass
class Task:
    id: int
    pickup: Cell
    delivery: Cell
    status: TaskStatus = TaskStatus.PENDING

    @property
    def done(self) -> bool:
        return self.status is TaskStatus.DONE

    @property
    def pending(self) -> bool:
        return self.status is TaskStatus.PENDING

    @property
    def picked(self) -> bool:
        return self.status is TaskStatus.PICKED


# ----------------------------------------------------------------------
# map generation
# ----------------------------------------------------------------------
def generate_grid(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Aisle-and-shelf layout: 1x4 / 1x6 horizontal bars on rows 3, 6, ... .

    The outer ring (rows 0/H-1, cols 0/W-1) is left free, which also
    guarantees the free space stays one connected component.
    """
    grid = np.zeros((H, W), dtype=np.int8)
    for r in range(3, H - 2, 3):
        c = 2
        while True:
            bar = int(rng.choice((4, 6)))
            end = min(c + bar - 1, W - 3)
            if end - c + 1 < 3:
                break
            grid[r, c : end + 1] = SHELF
            c = end + 1 + int(rng.choice((1, 2)))
            if c > W - 3:
                break
    return grid


def _free_cells(grid: np.ndarray) -> List[Cell]:
    H, W = grid.shape
    return [(r, c) for r in range(H) for c in range(W) if grid[r, c] == FREE]


def _adjacent_to_shelf(grid: np.ndarray, cell: Cell) -> bool:
    H, W = grid.shape
    r, c = cell
    for dr, dc in NB4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] == SHELF:
            return True
    return False


def _on_ring(H: int, W: int, cell: Cell) -> bool:
    r, c = cell
    return r in (0, H - 1) or c in (0, W - 1)


# ----------------------------------------------------------------------
# world
# ----------------------------------------------------------------------
class World:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.H, self.W = grid.shape
        # cell -> list of (t_start, t_end) blockage windows (inclusive)
        self.blocked_until: Dict[Cell, List[Tuple[int, int]]] = {}
        self._dist_cache: Dict[Cell, Dict[Cell, int]] = {}
        self._free = set(_free_cells(grid))

    # ---- geometry ----------------------------------------------------
    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.H and 0 <= c < self.W

    def is_shelf(self, cell: Cell) -> bool:
        return (not self.in_bounds(cell)) or self.grid[cell[0], cell[1]] == SHELF

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.grid[cell[0], cell[1]] == FREE

    def free_cells(self) -> List[Cell]:
        return list(self._free)

    def neighbours(self, cell: Cell) -> List[Cell]:
        r, c = cell
        return [n for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if self.is_free(n)]

    def moves(self, cell: Cell) -> List[Cell]:
        """Wait plus the four free neighbours (deterministic order)."""
        out = [cell]
        out.extend(self.neighbours(cell))
        return out

    # ---- dynamic blockages -------------------------------------------
    def add_blockage(self, cell: Cell, t0: int, t1: int) -> None:
        self.blocked_until.setdefault(cell, []).append((int(t0), int(t1)))

    def is_blocked(self, cell: Cell, t: int) -> bool:
        for (a, b) in self.blocked_until.get(cell, ()):
            if a <= t <= b:
                return True
        return False

    def blocked_remaining(self, cell: Cell, t: int) -> int:
        """Steps until the cell is free again (0 when not blocked)."""
        best = 0
        for (a, b) in self.blocked_until.get(cell, ()):
            if a <= t <= b:
                best = max(best, b - t + 1)
        return best

    def active_blockages(self, t: int) -> List[Cell]:
        return [c for c, wins in self.blocked_until.items()
                if any(a <= t <= b for (a, b) in wins)]

    # ---- distance maps -----------------------------------------------
    def dist_map(self, goal: Cell) -> Dict[Cell, int]:
        """BFS distances from ``goal`` over free cells (cached)."""
        cached = self._dist_cache.get(goal)
        if cached is not None:
            return cached
        dist: Dict[Cell, int] = {}
        if not self.is_free(goal):
            self._dist_cache[goal] = dist
            return dist
        dist[goal] = 0
        dq = deque([goal])
        while dq:
            cur = dq.popleft()
            d = dist[cur]
            r, c = cur
            for dr, dc in NB4:
                n = (r + dr, c + dc)
                if n not in dist and self.is_free(n):
                    dist[n] = d + 1
                    dq.append(n)
        self._dist_cache[goal] = dist
        return dist

    def distance(self, a: Cell, b: Cell) -> float:
        return self.dist_map(b).get(a, math.inf)

    def connected(self) -> bool:
        cells = self.free_cells()
        if not cells:
            return False
        return len(self.dist_map(cells[0])) == len(cells)

    def manhattan(self, a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ----------------------------------------------------------------------
# world + task construction
# ----------------------------------------------------------------------
def sample_from(pool, k: int, rng: np.random.Generator) -> List:
    """Sample ``k`` items; repeats are allowed only when the pool is smaller."""
    n = len(pool)
    if n == 0 or k <= 0:
        return []
    order = rng.permutation(n)
    out = []
    i = 0
    while len(out) < k:
        out.append(pool[int(order[i % n])])
        i += 1
    return out[:k]


def build_world(cfg, rng: np.random.Generator = None, *, grid=None):
    """Return ``(world, tasks, parking)`` for a scenario seeded by ``cfg.seed``.

    ``parking[i]`` is the dedicated parking cell of agent ``i``.  ``grid``
    overrides the generated aisle-and-shelf layout (the demo's congested
    warehouse walls one aisle off except for a single door, see
    :func:`scenarios.aisle_closure_grid`); every other step of the scenario
    build -- task sampling, parking, the connectivity check -- is unchanged.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    if grid is None:
        grid = generate_grid(cfg.H, cfg.W, rng)
    world = World(grid)
    if not world.connected():
        raise RuntimeError("generated map free space is not a single component")

    free = world.free_cells()
    shed = [c for c in free if _adjacent_to_shelf(grid, c)]
    ring = [c for c in free if _on_ring(cfg.H, cfg.W, c)]

    pickup_pool = shed or free
    delivery_pool = (ring + shed) or shed or free

    n = cfg.n_tasks
    pickups = sample_from(pickup_pool, n, rng)
    deliveries = sample_from(delivery_pool, n, rng)
    tasks: List[Task] = []
    for i in range(n):
        p = pickups[i]
        d = deliveries[i]
        guard = 0
        while d == p and guard < 8:
            d = delivery_pool[int(rng.integers(len(delivery_pool)))]
            guard += 1
        tasks.append(Task(id=i, pickup=p, delivery=d))

    task_cells = set()
    for tk in tasks:
        task_cells.add(tk.pickup)
        task_cells.add(tk.delivery)

    ring_park = [c for c in ring if c not in task_cells]
    taken = set(ring_park)
    other = [c for c in free if c not in task_cells and c not in taken]
    ring_park = sample_from(ring_park, len(ring_park), rng)
    other = sample_from(other, len(other), rng)
    parking = (ring_park + other)[: cfg.n_agents]
    if len(parking) < cfg.n_agents:
        raise RuntimeError("not enough free cells for dedicated parking spots")
    return world, tasks, parking


def assign_tasks(tasks: List[Task], parking: List[Cell], rng: np.random.Generator):
    """Round-robin task assignment; returns ``list[list[Task]]`` per agent."""
    per = [[] for _ in parking]
    for tk in tasks:
        per[tk.id % len(parking)].append(tk)
    return per

