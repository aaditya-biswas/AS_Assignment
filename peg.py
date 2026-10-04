"""Plan Envelope Graph / grounding layer (``SPEC.md`` section 8).

Takes a POCL partial-order plan for one agent and *grounds* it: every
``Travel`` / ``Fetch`` / ``Haul`` / ``Emergency`` macro is expanded into
atomic, time-indexed moves via Space-Time A* against the shared reservation
table.  The result is a list of ``PlanStep`` whose times are the STN
solution.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from planner import DROP, MOVE, PICK, WAIT, PlanStep, plan_to_path, st_astar
from pocl import POP, Action, pop_waypoints
from reservation import ReservationTable
from world import Cell, Task, World


def _leg(world: World, res: ReservationTable, agent_id: int, cur: Cell,
         goal: Cell, tc: int, path: List[PlanStep], T_limit: int,
         hold_after: int):
    """Plan ``cur -> goal`` reserving ``hold_after`` extra free ticks at the goal.

    When ``cur == goal`` and the hold condition already holds, this is a no-op
    returning the unchanged position (``st_astar`` returns the trivial path).
    """
    leg, _ = st_astar(world, res, agent_id, cur, goal, tc, T_limit=T_limit,
                      strict_goal_time=False, hold_after=hold_after)
    if leg is None:
        return None
    for (c, tt) in leg[1:]:
        path.append(PlanStep(tt, c, MOVE))
    return leg[-1]


def ground_action(world: World, res: ReservationTable, agent_id: int,
                  cur: Cell, tc: int, path: List[PlanStep], act: Action,
                  T_limit: int) -> Optional[Tuple[Cell, int]]:
    """Ground one macro action, appending atomic ``PlanStep``s to ``path``.

    Every leg is planned with the ``hold_after`` it needs so the following
    action tick is guaranteed to be free of higher-priority reservations.
    """
    if act.kind == "travel":
        return _leg(world, res, agent_id, cur, act.cell_to, tc, path,
                    T_limit, hold_after=1)

    if act.kind == "fetch":
        pos = _leg(world, res, agent_id, cur, act.location, tc, path,
                   T_limit, hold_after=1)
        if pos is None:
            return None
        cur, tc = pos
        tc += 1
        path.append(PlanStep(tc, cur, PICK, item=act.item))
        return (cur, tc)

    if act.kind == "haul":
        pos = _leg(world, res, agent_id, cur, act.cell_to, tc, path,
                   T_limit, hold_after=1)
        if pos is None:
            return None
        cur, tc = pos
        tc += 1
        path.append(PlanStep(tc, cur, DROP, item=act.item))
        return (cur, tc)

    if act.kind == "emergency":
        hold = max(1, int(act.dur))
        pos = _leg(world, res, agent_id, cur, act.location, tc, path,
                   T_limit, hold_after=hold)
        if pos is None:
            return None
        cur, tc = pos
        for _ in range(hold):
            tc += 1
            path.append(PlanStep(tc, cur, WAIT))
        return (cur, tc)

    raise ValueError(f"cannot ground macro action of kind {act.kind!r}")


def ground_pop(world: World, res: ReservationTable, agent_id: int,
               start: Cell, parking: Cell, pop: POP, t0: int = 0,
               T_limit: int = 600, park: bool = True) -> Optional[List[PlanStep]]:
    """Ground ``pop`` for one agent; ``None`` on failure."""
    steps: List[PlanStep] = [PlanStep(t0, start, WAIT)]
    cur, tc = start, t0
    for act in pop_waypoints(pop):
        out = ground_action(world, res, agent_id, cur, tc, steps, act, T_limit)
        if out is None:
            return None
        cur, tc = out

    if park:
        leg, _ = st_astar(world, res, agent_id, cur, parking, tc,
                          T_limit=T_limit, strict_goal_time=True)
        if leg is None:
            return None
        for (c, tt) in leg[1:]:
            steps.append(PlanStep(tt, c, MOVE))
    return steps


def peg_solve(world: World, per_agent_tasks: Sequence[Sequence[Task]],
              parking: Sequence[Cell], res: Optional[ReservationTable] = None,
              t0: int = 0, T_limit: int = 600, max_nodes: int = 50_000,
              order: Optional[Sequence[int]] = None):
    """Full v2 pipeline: POCL (L1) then PEG grounded A* (L2), in priority order.

    Returns ``(plans, failed_ids)``; ``plans[aid]`` is ``None`` for agents that
    could not be grounded (``FAILED_INIT``).
    """
    res = ReservationTable() if res is None else res
    order = list(range(len(parking))) if order is None else list(order)
    plans: Dict[int, Optional[List[PlanStep]]] = {}
    failed = set()

    from planner import plan_agent  # local import to avoid a cycle at import time
    from pocl import pocl

    for aid in order:
        tasks = list(per_agent_tasks[aid])
        pop = pocl(aid, tasks, parking[aid], world, max_nodes=max_nodes)
        steps = None
        if pop is not None:
            steps = ground_pop(world, res, aid, parking[aid], parking[aid], pop,
                               t0, T_limit)
        if steps is None:
            # fall back to the greedy ordering + direct grounding
            from planner import greedy_task_order
            fallback = greedy_task_order(world, parking[aid], tasks)
            steps = plan_agent(world, res, aid, parking[aid], parking[aid],
                               fallback, t0, T_limit)
        if steps is None:
            plans[aid] = None
            failed.add(aid)
            continue
        res.add_path(aid, plan_to_path(steps), park=True)
        plans[aid] = steps
    return plans, failed
