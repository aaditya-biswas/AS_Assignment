"""Comparison baselines (``SPEC.md`` sections 10 and 12).

``global_plan`` is the **only** entry point allowed to invoke the global
prioritized planner.  ``modify.py`` (the repair engine) must never import it;
``tests/test_modify.py`` asserts that a repair run leaves
``GLOBAL_PLAN_CALLS`` at 0.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import planner
from planner import PlanStep, greedy_task_order
from reservation import ReservationTable
from world import Cell, Task, World

#: Times the global planner has been invoked.  Repair must leave this at 0.
GLOBAL_PLAN_CALLS = 0


def reset_counters() -> None:
    global GLOBAL_PLAN_CALLS
    GLOBAL_PLAN_CALLS = 0


def global_plan(world: World, res: ReservationTable,
                jobs: Sequence[Tuple[int, Cell, Cell, Sequence[Task]]],
                t0: int = 0, T_limit: int = 600):
    """Prioritized grounding over all remaining jobs (quality upper bound)."""
    global GLOBAL_PLAN_CALLS
    GLOBAL_PLAN_CALLS += 1
    return planner.prioritized_ground(world, res, jobs, t0, T_limit)
