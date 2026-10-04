"""Dynamic disruption events and injection (``SPEC.md`` section 10).

Two families are supported:

* **Exogenous** -- ``BlockageEvent`` maps a *cell + time window* onto the
  ``World`` (travel becomes time-dependent); this is the only disruption the
  planner can be aware of *a priori*.
* **Endogenous** -- ``AccidentEvent`` (an agent freezes for ``duration`` ticks)
  and ``EmergencyEvent`` (an agent must divert to a marshalling cell and hold
  for ``hold`` ticks).  Both are injected into an already-planned schedule by
  *plan surgery*, which is what creates the conflicts that must be repaired.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from planner import PlanStep, WAIT
from pocl import EmergencySpec
from world import Cell, Task, World


class ReplanMode(str, Enum):
    """The four repair strategies of ``SPEC.md`` section 11."""

    STATIC = "R1_static"          # freeze blockages at t=0 (no repair)
    RESTART = "R2_restart"        # replan every agent from its current state
    BFS_ORDER = "R3_bfs_order"    # restart, priority = BFS distance to hotspot
    ROBUST = "R4_robust"          # keep P0 when still feasible, else refine

    @property
    def is_reactive(self) -> bool:
        return self is not ReplanMode.STATIC


# ----------------------------------------------------------------------
# event types
# ----------------------------------------------------------------------
@dataclass
class AccidentEvent:
    agent: int
    at_t: int
    duration: int
    #: ``True`` for a breakdown that never clears: the agent is out of service
    #: for good, so its remaining goals must be handed to other agents (R4).
    permanent: bool = False


@dataclass
class EmergencyEvent:
    agent: int
    at_t: int
    hold: int = 5
    cell: Optional[Cell] = None       # chosen at injection time


@dataclass
class BlockageEvent:
    cell: Cell
    t0: int
    duration: int

    @property
    def t1(self) -> int:
        return self.t0 + self.duration


Event = object   # AccidentEvent | EmergencyEvent | BlockageEvent


# ----------------------------------------------------------------------
# plan surgery
# ----------------------------------------------------------------------
def plan_trace(steps: Sequence[PlanStep], horizon: Optional[int] = None) -> List[Cell]:
    """Cell-per-tick trace of a plan (padded at both ends)."""
    if not steps:
        return []
    end = steps[-1].t if horizon is None else horizon
    pos = {s.t: s.cell for s in steps}
    first, last, t0 = steps[0].cell, steps[-1].cell, steps[0].t
    return [pos[t] if t in pos else (first if t < t0 else last)
            for t in range(end + 1)]


def freeze_plan(steps: Sequence[PlanStep], at_t: int,
                duration: int) -> List[PlanStep]:
    """Freeze an agent for ``duration`` ticks at ``at_t`` (accident surgery).

    The agent holds the cell it occupies at ``at_t`` for ticks
    ``at_t .. at_t + duration - 1``; every later step shifts by ``duration``.
    """
    steps = list(steps)
    if not steps or duration <= 0 or at_t > steps[-1].t:
        return steps
    cell = plan_trace(steps)[at_t]
    out: List[PlanStep] = [s for s in steps if s.t < at_t]
    t = at_t - 1
    for _ in range(duration):
        t += 1
        out.append(PlanStep(t, cell, WAIT))
    for s in steps:
        if s.t >= at_t:
            out.append(PlanStep(s.t + duration, s.cell, s.action, s.item))
    return out


def pick_safe_cell(world: World, occupied: Sequence[Cell],
                   rng: Optional[np.random.Generator] = None) -> Optional[Cell]:
    """A free cell maximising the minimum BFS distance to any occupied cell."""
    occ = list(occupied)
    best, best_d = None, -1.0
    for cell in world.free_cells():
        if not occ:
            return cell
        d = min(world.distance(cell, o) for o in occ)
        if d > best_d or (d == best_d and best is not None and cell < best):
            best, best_d = cell, d
    return best


# ----------------------------------------------------------------------
# injection
# ----------------------------------------------------------------------
def inject_blockages(world: World, events: Sequence[BlockageEvent]) -> None:
    for ev in events:
        world.add_blockage(ev.cell, ev.t0, ev.t1)


def postpone_blockages(plans: Dict[int, Optional[List[PlanStep]]],
                       events: Sequence[BlockageEvent],
                       limit: int = 400) -> List[BlockageEvent]:
    """Delay a blockage until no agent occupies its cell (``SPEC.md`` 10).

    An event announced while a cell is occupied would have the occupying agent
    inside the blockage *at the freeze tick*, which no local repair can undo
    (the step is already executed).  Such events are shifted forward tick by
    tick; events that can never be applied -- an agent parked on the cell for
    good -- are dropped.
    """
    live = {aid: steps for aid, steps in plans.items() if steps}
    if not live or not events:
        return list(events)
    horizon = max(ev.t0 + ev.duration for ev in events) + limit
    traces = {aid: plan_trace(steps, horizon=horizon)
              for aid, steps in live.items()}
    out: List[BlockageEvent] = []
    for ev in events:
        t = max(0, ev.t0)
        while t <= ev.t0 + limit and any(tr[t] == ev.cell for tr in traces.values()):
            t += 1
        if t > ev.t0 + limit:
            continue                       # never applicable -> dropped
        out.append(BlockageEvent(ev.cell, t, ev.duration))
    return out


def inject_accidents(plans: Dict[int, Optional[List[PlanStep]]],
                     events: Sequence[AccidentEvent]) -> Dict[int, Optional[List[PlanStep]]]:
    """Plan surgery: freeze the affected agents' plans."""
    out = dict(plans)
    for ev in events:
        steps = out.get(ev.agent)
        if steps is None:
            continue
        out[ev.agent] = freeze_plan(steps, ev.at_t, ev.duration)
    return out


def agent_cell_at(plans: Dict[int, Optional[List[PlanStep]]], aid: int,
                  t: int, default: Optional[Cell] = None) -> Optional[Cell]:
    """Where agent ``aid`` is at tick ``t`` (``default`` if unplannable)."""
    steps = plans.get(aid)
    if not steps:
        return default
    trace = plan_trace(steps, horizon=max(t, steps[-1].t))
    if t >= len(trace):
        return trace[-1] if trace else default
    return trace[t]


def inject_emergencies(world: World,
                       plans: Dict[int, Optional[List[PlanStep]]],
                       events: Sequence[EmergencyEvent],
                       rng: Optional[np.random.Generator] = None
                       ) -> Dict[int, EmergencySpec]:
    """Choose a marshalling cell per emergency and return the planner specs.

    The current schedule is *not* modified -- the emergency is a planning
    requirement handed to the repair step (the agent must divert and hold).
    The marshalling cell maximises the minimum distance to where every other
    agent is at the emergency time.
    """
    specs: Dict[int, EmergencySpec] = {}
    for ev in events:
        cell = ev.cell
        if cell is None:
            occ: List[Cell] = []
            for aid, steps in plans.items():
                if aid == ev.agent or not steps:
                    continue
                c = agent_cell_at(plans, aid, ev.at_t)
                if c is not None:
                    occ.append(c)
            cell = pick_safe_cell(world, occ, rng)
        if cell is None:
            continue
        specs[ev.agent] = EmergencySpec(agent=ev.agent, cell=cell, hold=ev.hold)
    return specs


# ----------------------------------------------------------------------
# scenario generation
# ----------------------------------------------------------------------
def random_blockages(world: World, n: int, rng: np.random.Generator,
                     t_max: int = 80, dur_range: Tuple[int, int] = (4, 14),
                     avoid: Sequence[Cell] = ()
                     ) -> List[BlockageEvent]:
    """``n`` random cell-time blockages; ``avoid`` cells are never chosen.

    ``avoid`` is normally the agents' parking spots (a blockage there would
    trap a parked agent, which no repair can resolve).
    """
    skip = set(avoid)
    free = [c for c in world.free_cells() if c not in skip]
    rng.shuffle(free)
    events: List[BlockageEvent] = []
    for cell in free[:n]:
        t0 = int(rng.integers(0, max(1, t_max)))
        events.append(BlockageEvent(cell, t0, int(rng.integers(*dur_range))))
    return events


def random_accidents(n_agents: int, rng: np.random.Generator, t_max: int = 60,
                     dur_range: Tuple[int, int] = (4, 10)) -> List[AccidentEvent]:
    agents = rng.permutation(n_agents)
    return [AccidentEvent(int(a), int(rng.integers(1, max(2, t_max))),
                          int(rng.integers(*dur_range)))
            for a in agents[: min(n_agents, max(1, n_agents // 5))]]


def random_emergencies(n_agents: int, rng: np.random.Generator,
                       t_max: int = 60, hold_range: Tuple[int, int] = (3, 6)
                       ) -> List[EmergencyEvent]:
    agents = rng.permutation(n_agents)
    return [EmergencyEvent(int(a), int(rng.integers(1, max(2, t_max))),
                           int(rng.integers(*hold_range)))
            for a in agents[: min(n_agents, max(1, n_agents // 5))]]

