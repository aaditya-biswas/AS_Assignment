"""Space-time reservation table (``SPEC.md`` section 5).

Three maps:

* ``vertex[(cell, t)] = agent``  -- agent occupies ``cell`` at integer time ``t``.
* ``edge[(from, to, t)] = agent`` -- agent moves ``from -> to`` during ``t .. t+1``.
* ``park[cell] = (agent, t_from)`` -- agent sits on ``cell`` forever from ``t_from``.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]


class ReservationTable:
    __slots__ = ("vertex", "edge", "park")

    def __init__(self) -> None:
        self.vertex: Dict[Tuple[Cell, int], int] = {}
        self.edge: Dict[Tuple[Cell, Cell, int], int] = {}
        self.park: Dict[Cell, Tuple[int, int]] = {}

    # ---- snapshots ----------------------------------------------------
    def snapshot(self):
        return (dict(self.vertex), dict(self.edge), dict(self.park))

    def restore(self, snap) -> None:
        self.vertex = dict(snap[0])
        self.edge = dict(snap[1])
        self.park = dict(snap[2])

    def copy(self) -> "ReservationTable":
        t = ReservationTable()
        t.vertex = dict(self.vertex)
        t.edge = dict(self.edge)
        t.park = dict(self.park)
        return t

    # ---- mutation -----------------------------------------------------
    def add_path(self, agent_id: int, path: Iterable[Tuple[Cell, int]],
                 park: bool = True) -> None:
        """Reserve a time-indexed path ``[(cell, t), ...]`` (t increasing)."""
        path = list(path)
        for cell, t in path:
            self.vertex[(cell, t)] = agent_id
        for i in range(len(path) - 1):
            (c0, t0) = path[i]
            (c1, t1) = path[i + 1]
            if c0 != c1 and t1 == t0 + 1:
                self.edge[(c0, c1, t0)] = agent_id
        if park and path:
            cf, tf = path[-1]
            self.park[cf] = (agent_id, tf)

    def set_park(self, agent_id: int, cell: Cell, t_from: int) -> None:
        self.park[cell] = (agent_id, t_from)

    def remove_agent(self, agent_id: int) -> None:
        a = agent_id
        self.vertex = {k: v for k, v in self.vertex.items() if v != a}
        self.edge = {k: v for k, v in self.edge.items() if v != a}
        self.park = {k: v for k, v in self.park.items() if v[0] != a}

    def remove_agent_after(self, agent_id: int, t: int) -> None:
        a = agent_id
        self.vertex = {k: v for k, v in self.vertex.items()
                       if not (v == a and k[1] >= t)}
        self.edge = {k: v for k, v in self.edge.items()
                     if not (v == a and k[2] >= t)}
        self.park = {k: v for k, v in self.park.items()
                     if not (v[0] == a and v[1] >= t)}

    # ---- queries ------------------------------------------------------
    def clashes(self, agent_id: int, cell_from: Cell, cell_to: Cell,
                t: int) -> Set[int]:
        """Agents (other than ``agent_id``) blocking ``cell_from -> cell_to``.

        Covers the vertex clash at ``t+1``, the swap clash and the park clash.
        """
        out: Set[int] = set()
        v = self.vertex.get((cell_to, t + 1))
        if v is not None and v != agent_id:
            out.add(v)
        e = self.edge.get((cell_to, cell_from, t))
        if e is not None and e != agent_id:
            out.add(e)
        pk = self.park.get(cell_to)
        if pk is not None and pk[0] != agent_id and t + 1 >= pk[1]:
            out.add(pk[0])
        return out

    def occupants_at(self, cell: Cell, t: int, excluding: Optional[int] = None) -> Set[int]:
        out: Set[int] = set()
        v = self.vertex.get((cell, t))
        if v is not None and v != excluding:
            out.add(v)
        pk = self.park.get(cell)
        if pk is not None and pk[0] != excluding and t >= pk[1]:
            out.add(pk[0])
        return out

    def last_reserved_time(self, cell: Cell,
                           excluding_agent: Optional[int] = None) -> float:
        """Latest ``t`` at which another agent holds ``cell`` (park = +inf)."""
        best: float = -1.0
        for (c, t), a in self.vertex.items():
            if c == cell and a != excluding_agent and t > best:
                best = float(t)
        pk = self.park.get(cell)
        if pk is not None and pk[0] != excluding_agent:
            return math.inf
        return best

    def is_free(self, cell: Cell, t: int, excluding: Optional[int] = None) -> bool:
        return not self.occupants_at(cell, t, excluding)

    def time_horizon(self) -> int:
        if not self.vertex:
            return -1
        return max(t for (_, t) in self.vertex.keys())

    def agents(self) -> Set[int]:
        out = set(self.vertex.values())
        out.update(v[0] for v in self.park.values())
        return out

    # ---- validation ---------------------------------------------------
    def find_conflicts(self) -> List[str]:
        """Return a list of human-readable conflicts (empty when consistent)."""
        problems: List[str] = []
        for (cell, t), a in self.vertex.items():
            pk = self.park.get(cell)
            if pk is not None and pk[0] != a and t >= pk[1]:
                problems.append(
                    f"vertex {cell}@{t} by {a} overlaps park by {pk[0]}@{pk[1]}"
                )
        return problems
