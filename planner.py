"""Space-Time A* and prioritized grounding (``SPEC.md`` section 6).

The planner is the L2 (grounding) layer: it turns an ordered list of
waypoints into a time-indexed, reservation-respecting path.  It never decides
the *task order* (that is POCL's job in the v2 layer); ``greedy_task_order``
is provided only as a baseline helper.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from reservation import ReservationTable
from world import MV, Cell, Task, TaskStatus, World

MOVE, WAIT, PICK, DROP = "MOVE", "WAIT", "PICK", "DROP"

#: Instrumentation for ``RepairRecord`` (``st_astar`` calls / states expanded).
ST_ASTAR_CALLS = 0
ST_ASTAR_NODES = 0



@dataclass
class PlanStep:
    t: int
    cell: Cell
    action: str = MOVE
    item: Optional[int] = None      # item id for PICK / DROP steps


# ----------------------------------------------------------------------
# Space-Time A*
# ----------------------------------------------------------------------
def st_astar(
    world: World,
    res: ReservationTable,
    agent_id: int,
    start: Cell,
    goal: Cell,
    t0: int,
    *,
    soft_lambda: Optional[int] = None,
    extra_hard: Iterable[Tuple[Cell, int]] = frozenset(),
    T_limit: int = 600,
    strict_goal_time: bool = False,
    hold_after: int = 0,
):
    """Plan ``start -> goal`` in space-time.

    Returns ``(path, clashed_agents)`` where ``path`` is a list of
    ``(cell, t)`` pairs from ``(start, t0)`` to the goal, or ``(None, set())``.

    * ``soft_lambda is None`` -> hard mode: any clash is forbidden.
    * ``soft_lambda = L``      -> soft mode: a clash costs ``+L`` and the
      clashing agent ids are returned so a negotiation can be started.
    * ``strict_goal_time`` -> require ``t >= last_reserved_time(goal)`` (the
      final parking leg); waypoint legs need only a clash-free arrival.
    * ``hold_after = k`` -> require the goal cell to be free for ``k`` further
      ticks after arrival.  Pickup/delivery legs use ``k = 1`` so the action
      tick cannot collide with an already-reserved agent.
    """
    if not world.is_free(start) or not world.is_free(goal):
        return None, set()
    global ST_ASTAR_CALLS, ST_ASTAR_NODES
    ST_ASTAR_CALLS += 1
    h = world.dist_map(goal)
    if start not in h:
        return None, set()

    extra_hard = frozenset(extra_hard)
    start_state = (start, t0)
    g: Dict[Tuple[Cell, int], float] = {start_state: 0.0}
    came: Dict[Tuple[Cell, int], Optional[Tuple[Cell, int]]] = {start_state: None}
    clash: Dict[Tuple[Cell, int], frozenset] = {start_state: frozenset()}
    counter = itertools.count()
    heap = [(h[start] + 0.0, 0.0, next(counter), start_state)]

    while heap:
        f, gc, _, (cell, t) = heapq.heappop(heap)
        if gc > g.get((cell, t), math.inf):
            continue
        holds = hold_after <= 0 or all(
            not res.clashes(agent_id, goal, goal, t + k)
            and not world.is_blocked(goal, t + k)
            for k in range(hold_after)
        )
        if (
            cell == goal
            and holds
            and (not strict_goal_time or t >= res.last_reserved_time(goal, agent_id))
        ):
            states: List[Tuple[Cell, int]] = []
            s: Optional[Tuple[Cell, int]] = (cell, t)
            while s is not None:
                states.append(s)
                s = came[s]
            states.reverse()
            return [(c, tt) for (c, tt) in states], set(clash[(cell, t)])
        if t - t0 >= T_limit:
            continue
        ST_ASTAR_NODES += 1
        r, c = cell
        for dr, dc in MV:
            n = (r + dr, c + dc)
            if not world.is_free(n):
                continue
            nt = t + 1
            if world.is_blocked(n, nt):
                continue
            if (n, nt) in extra_hard:
                continue
            cl = res.clashes(agent_id, cell, n, t)
            if cl and soft_lambda is None:
                continue
            ng = gc + 1.0 + (soft_lambda if cl else 0.0)
            ns = (n, nt)
            if ng < g.get(ns, math.inf):
                g[ns] = ng
                came[ns] = (cell, t)
                clash[ns] = clash[(cell, t)] | cl
                heapq.heappush(heap, (ng + h.get(n, 0.0), ng, next(counter), ns))
    return None, set()


def st_astar_soft(world, res, agent_id, start, goal, t0, lam, **kw):
    """Convenience wrapper: ST-A* in soft mode with penalty ``lam``."""
    return st_astar(world, res, agent_id, start, goal, t0,
                    soft_lambda=lam, **kw)


# ----------------------------------------------------------------------
# leg sequencing for one agent
# ----------------------------------------------------------------------
def greedy_task_order(world: World, start: Cell, tasks: Sequence[Task]) -> List[Task]:
    """Nearest-pickup-first ordering (v1 baseline / fallback)."""
    remaining = list(tasks)
    order: List[Task] = []
    pos = start
    while remaining:
        best = min(remaining, key=lambda tk: (world.distance(pos, tk.pickup), tk.id))
        order.append(best)
        remaining.remove(best)
        pos = best.delivery
    return order


def plan_agent(
    world: World,
    res: ReservationTable,
    agent_id: int,
    start: Cell,
    parking: Cell,
    tasks: Sequence[Task],
    t0: int = 0,
    T_limit: int = 600,
    strict_park: bool = True,
    soft_lambda: Optional[int] = None,
    clash_sink: Optional[Set[int]] = None,
) -> Optional[List[PlanStep]]:
    """Ground ``start -> p1 -> d1 -> ... -> parking`` into ``PlanStep``s.

    One ``PICK`` wait step is inserted at each pickup and one ``DROP`` wait
    step at each delivery.  The caller is responsible for adding the result to
    the reservation table (``prioritized_ground`` does this).

    ``soft_lambda`` switches the legs to *soft* ST-A* (``SPEC.md`` 9.2): a
    clash is allowed at the price of ``soft_lambda`` and the clashed agents are
    collected in ``clash_sink`` -- this is the negotiation proposal.  With the
    default (``None``) nothing changes: every clash is forbidden.
    """
    steps: List[PlanStep] = [PlanStep(t0, start, WAIT)]
    cur, tc = start, t0
    for task in tasks:
        for leg_goal, act in ((task.pickup, PICK), (task.delivery, DROP)):
            # hold_after=1: reserve room for the 1-tick PICK/DROP action.
            path, clashed = st_astar(world, res, agent_id, cur, leg_goal, tc,
                                     T_limit=T_limit, strict_goal_time=False,
                                     hold_after=1, soft_lambda=soft_lambda)
            if path is None:
                return None
            if clash_sink is not None:
                clash_sink |= clashed
            for (c, tt) in path[1:]:
                steps.append(PlanStep(tt, c, MOVE))
            cur, tc = path[-1]
            tc += 1
            steps.append(PlanStep(tc, cur, act, item=task.id))
    path, clashed = st_astar(world, res, agent_id, cur, parking, tc,
                             T_limit=T_limit, strict_goal_time=strict_park,
                             soft_lambda=soft_lambda)
    if path is None:
        return None
    if clash_sink is not None:
        clash_sink |= clashed
    for (c, tt) in path[1:]:
        steps.append(PlanStep(tt, c, MOVE))
    return steps


def plan_to_path(steps: Sequence[PlanStep]) -> List[Tuple[Cell, int]]:
    return [(s.cell, s.t) for s in steps]


def prioritized_ground(
    world: World,
    res: ReservationTable,
    jobs: Sequence[Tuple[int, Cell, Cell, Sequence[Task]]],
    t0: int = 0,
    T_limit: int = 600,
):
    """Plan each ``(agent_id, start, parking, tasks)`` against a shared table.

    ``jobs`` must already be in priority order.  Returns
    ``(plans, failed_ids)`` where ``plans[aid]`` is a ``list[PlanStep]`` (or
    ``None`` when the agent is ``FAILED_INIT``).
    """
    plans: Dict[int, Optional[List[PlanStep]]] = {}
    failed: Set[int] = set()
    for (aid, start, parking, tasks) in jobs:
        steps = plan_agent(world, res, aid, start, parking, tasks, t0, T_limit)
        if steps is None:
            steps = plan_agent(world, res, aid, start, parking, tasks, t0,
                               2 * T_limit)
        if steps is None:
            plans[aid] = None
            failed.add(aid)
            continue
        res.add_path(aid, plan_to_path(steps), park=True)
        plans[aid] = steps
    return plans, failed


# v1 name kept for compatibility with the original specification/tests.
prioritized_plan = prioritized_ground


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------
def validate_paths(world: World, paths: Dict[int, List[Tuple[Cell, int]]]) -> List[str]:
    """Return a list of problems (vertex, swap, blocked-cell, shelf)."""
    problems: List[str] = []
    occ: Dict[Tuple[Cell, int], int] = {}
    edges: Dict[Tuple[Cell, Cell, int], int] = {}
    for aid in sorted(paths):
        p = paths[aid]
        for (cell, t) in p:
            if world.is_shelf(cell):
                problems.append(f"agent {aid} on shelf {cell} @{t}")
            if world.is_blocked(cell, t):
                problems.append(f"agent {aid} inside blockage {cell} @{t}")
            prev = occ.get((cell, t))
            if prev is not None and prev != aid:
                problems.append(f"vertex clash {cell}@{t}: agents {prev} & {aid}")
            occ[(cell, t)] = aid
        for i in range(len(p) - 1):
            (c0, t0) = p[i]
            (c1, t1) = p[i + 1]
            if c1 != c0 and t1 == t0 + 1:
                edges[(c0, c1, t0)] = aid
    for (c0, c1, t0), aid in edges.items():
        other = edges.get((c1, c0, t0))
        if other is not None and other != aid:
            problems.append(f"swap {c0} <-> {c1} @{t0}: agents {aid} & {other}")
    return problems


def make_static_grid(paths: Dict[int, List[Tuple[Cell, int]]]):
    """Helper for tests/visualisation: cell -> set of times it is occupied."""
    occ: Dict[Cell, Set[int]] = {}
    for aid, p in paths.items():
        for (cell, t) in p:
            occ.setdefault(cell, set()).add(t)
    return occ

