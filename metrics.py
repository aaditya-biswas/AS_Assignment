"""Altered-agent metrics (``SPEC.md`` section 9.5).

Four views of "how much did a repair change agent ``a``'s plan", computed on
the *remaining* plan (everything strictly after the freeze tick):

* ``altered_plan``  (primary) -- the step sequence or its content changed.
* ``altered_path``            -- the cell route or the task list changed.
* ``delayed_only``            -- only the times changed; signature identical.
* ``altered_naive``  (ablation)-- any difference in the remaining ``(cell, t)``
  pairs, the original v1 definition.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from planner import PlanStep

UNCHANGED = "unchanged"
DELAYED_ONLY = "delayed_only"
ALTERED_PATH = "altered_path"
ALTERED_PLAN = "altered_plan"

_ORDER = {UNCHANGED: 0, DELAYED_ONLY: 1, ALTERED_PATH: 2, ALTERED_PLAN: 3}


def tail(steps: Optional[Sequence[PlanStep]], t_now: int) -> List[PlanStep]:
    """Steps that are *not yet executed* at tick ``t_now``."""
    if not steps:
        return []
    return [s for s in steps if s.t > t_now]


def remaining_pairs(steps: Optional[Sequence[PlanStep]],
                    t_now: int) -> Tuple[Tuple[int, int], ...]:
    """Relative ``(cell, dt)`` pairs of the remaining plan (delay-insensitive)."""
    return tuple((s.cell, s.t - t_now) for s in tail(steps, t_now))


def action_signature(steps: Optional[Sequence[PlanStep]],
                     t_now: int) -> Tuple[Tuple[str, Optional[int]], ...]:
    return tuple((s.action, s.item) for s in tail(steps, t_now))


def route_signature(steps: Optional[Sequence[PlanStep]],
                    t_now: int) -> Tuple[int, ...]:
    """The cell route of a ``PlanStep`` sequence, run-length encoded.

    ``[a a b b b a]`` -> ``[a, b, a]``: consecutive repeats (waiting / acting in
    place) are collapsed, so inserting a wait never changes the route.
    """
    out: List[int] = []
    for s in tail(steps, t_now):
        if not out or s.cell != out[-1]:
            out.append(s.cell)
    return tuple(out)


def leg_cells(steps: Optional[Sequence[PlanStep]],
              t_now: int) -> Tuple[Tuple[int, ...], ...]:
    """The route split at each action step (one tuple per leg)."""
    legs: List[Tuple[int, ...]] = []
    cur: List[int] = []
    for s in tail(steps, t_now):
        if s.cell != (cur[-1] if cur else None):
            cur.append(s.cell)
        if s.action in ("PICK", "DROP"):
            legs.append(tuple(cur))
            cur = []
    if cur:
        legs.append(tuple(cur))
    return tuple(legs)


def classify(old: Optional[Sequence[PlanStep]], new: Optional[Sequence[PlanStep]],
             t_now: int) -> str:
    """Label the magnitude of the change for one agent."""
    if not old and not new:
        return UNCHANGED
    if not old or not new:
        return ALTERED_PLAN
    if tail(old, t_now) == tail(new, t_now):
        return UNCHANGED
    if action_signature(old, t_now) != action_signature(new, t_now):
        return ALTERED_PLAN
    if leg_cells(old, t_now) != leg_cells(new, t_now):
        return ALTERED_PATH
    return DELAYED_ONLY


def completion_delta(old: Optional[Sequence[PlanStep]],
                     new: Optional[Sequence[PlanStep]]) -> int:
    """How much later the agent finishes (0 when it cannot finish at all)."""
    if not old or not new:
        return 0
    return max(0, new[-1].t - old[-1].t)


def altered_agent_sets(prev: Dict[int, Optional[List[PlanStep]]],
                       new: Dict[int, Optional[List[PlanStep]]],
                       t_now: int,
                       disrupted: Sequence[int] = ()) -> Dict[str, List[int]]:
    """The four SPEC 9.5 agent sets, plus ``altered_naive``."""
    a_plan: List[int] = []
    a_path: List[int] = []
    delayed: List[int] = []
    naive: List[int] = []
    for aid in sorted(set(prev) | set(new)):
        old, cur = prev.get(aid), new.get(aid)
        label = classify(old, cur, t_now)
        if label == ALTERED_PLAN:
            a_plan.append(aid)
        elif label == ALTERED_PATH:
            a_path.append(aid)
        elif label == DELAYED_ONLY:
            delayed.append(aid)
        if remaining_pairs(old, t_now) != remaining_pairs(cur, t_now):
            naive.append(aid)
    for aid in disrupted:            # the disrupted agent always counts
        if aid not in a_plan:
            a_plan.append(aid)
    return {
        "altered_plan": sorted(a_plan),
        "altered_path": sorted(set(a_path) | set(a_plan)),
        "delayed_only": sorted(set(delayed) - set(a_plan)),
        "altered_naive": sorted(set(naive) | set(a_plan)),
    }


def total_added_delay(prev: Dict[int, Optional[List[PlanStep]]],
                      new: Dict[int, Optional[List[PlanStep]]]) -> int:
    """Sum over agents of how much later they now finish."""
    return sum(completion_delta(prev.get(a), new.get(a))
               for a in set(prev) | set(new))
