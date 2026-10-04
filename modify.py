"""Local plan modification / repair (``SPEC.md`` sections 8-9).

Design commitments taken from the specification:

* **The global planner is never called here.**  Repair only ever uses
  per-agent POCL refinement (``pocl.pocl``) and L2 grounding
  (``planner.st_astar`` / ``planner.plan_agent``).  ``tests/test_modify.py``
  monkeypatches ``planner.prioritized_ground`` to raise and asserts that a
  repair run never triggers it.
* **The executed prefix is frozen.**  A repair may only touch steps with
  ``t > t_freeze``.
* **Escalation ladder.**  For each affected agent we accept the first
  candidate that works, in the order

  ======  ==========================================================
  level   action
  ======  ==========================================================
  ``R0``  same route/order; only waits are inserted (pure delay)
  ``R1``  reroute the legs of the existing task order, delay <= ``delay_threshold``
  ``R2``  reroute the legs of the existing task order, longer detour accepted
  ``R3``  re-derive the agent's task order (its own POCL / greedy) and reground
  ``R4``  hand orphan goals to another agent (POCL from the receiver's state)
  ======  ==========================================================

  Levels are *labelled*, not separately searched: a hard ST-A* regrounding of
  the existing order yields waits where they suffice (``R0``) and detours
  otherwise (``R1``/``R2``); the label is derived from the candidate by
  ``metrics.classify`` and ``metrics.completion_delta``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import planner
import negotiation
from metrics import (UNCHANGED, altered_agent_sets, classify,
                     completion_delta, route_signature, tail,
                     total_added_delay)
from negotiation import Negotiation
from planner import (DROP, MOVE, PICK, WAIT, PlanStep, greedy_task_order,
                     plan_agent, plan_to_path)
from pocl import EmergencySpec, pocl
from reservation import ReservationTable
from world import Cell, Task, World

#: Number of times the global planner was invoked.  Repair must leave this at 0.
GLOBAL_PLAN_CALLS = 0


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
@dataclass
class Disruption:
    """An event that turns part of a plan into a *flaw*."""

    kind: str                       # accident | breakdown | emergency | blockage
    agent: int
    t: int
    duration: int = 0
    cell: Optional[Cell] = None
    permanent: bool = False
    hold: int = 0

    @property
    def is_emergency(self) -> bool:
        return self.kind == "emergency"


@dataclass
class RepairRecord:
    event_type: str
    t: int
    A0_size: int
    level_resolved: str
    altered_plan_ids: List[int] = field(default_factory=list)
    altered_path_ids: List[int] = field(default_factory=list)
    delayed_ids: List[int] = field(default_factory=list)
    naive_ids: List[int] = field(default_factory=list)
    added_delay_total: int = 0
    messages_by_type: Dict[str, int] = field(default_factory=dict)
    pocl_nodes: int = 0
    st_astar_calls: int = 0
    cpu_ms: float = 0.0
    success: bool = True


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def position_at(steps: Optional[Sequence[PlanStep]], t: int,
                default: Optional[Cell] = None) -> Optional[Cell]:
    """Cell occupied by a ``PlanStep`` sequence at tick ``t``."""
    if not steps:
        return default
    pos = {s.t: s.cell for s in steps}
    if t in pos:
        return pos[t]
    first, last, t0 = steps[0].cell, steps[-1].cell, steps[0].t
    if t < t0:
        return first
    return last


def freeze(steps: Optional[Sequence[PlanStep]], t_now: int) -> List[PlanStep]:
    """The executed prefix: steps with ``t <= t_now``.

    A step at exactly ``t_now`` is synthesised when the plan had none (the
    agent is simply sitting where it was).
    """
    if not steps:
        return []
    pre = [s for s in steps if s.t <= t_now]
    if not pre:
        return [PlanStep(t_now, steps[0].cell, WAIT)]
    if pre[-1].t < t_now:
        pre.append(PlanStep(t_now, pre[-1].cell, WAIT))
    return pre


def tail_steps(steps: Optional[Sequence[PlanStep]], t_now: int) -> List[PlanStep]:
    """Steps strictly after ``t_now``."""
    return list(tail(steps, t_now))


def merge(prefix: Sequence[PlanStep], rest: Sequence[PlanStep]) -> List[PlanStep]:
    return list(prefix) + list(rest)


def pending_tasks(steps: Optional[Sequence[PlanStep]], tasks: Sequence[Task],
                  t_now: int) -> Tuple[List[Task], Optional[Task]]:
    """Tasks still to do, plus the task whose item is currently being carried."""
    if not steps:
        return [t for t in tasks if not t.done], None
    delivered = {s.item for s in steps if s.action == DROP and s.t <= t_now}
    picked = {s.item for s in steps if s.action == PICK and s.t <= t_now}
    holding = next((t for t in tasks
                    if t.id in picked and t.id not in delivered), None)
    todo = [t for t in tasks if t.id not in delivered and t is not holding]
    return todo, holding


def _label(prev: Optional[Sequence[PlanStep]], new: Sequence[PlanStep],
           t_now: int, delay_threshold: int) -> str:
    """Classify a regrounding candidate onto the R0/R1/R2 rungs."""
    if prev is None:
        return "R2"
    if classify(prev, new, t_now) == UNCHANGED:
        return "R0"
    if route_signature(prev, t_now) == route_signature(new, t_now):
        return "R0"                       # same route, only extra waits
    if completion_delta(prev, new) <= delay_threshold:
        return "R1"
    return "R2"


LEVELS = ("R0", "R1", "R2", "R3", "R4")


# ----------------------------------------------------------------------
# per-agent regrounding
# ----------------------------------------------------------------------
def _plan_from(world: World, res: ReservationTable, aid: int, start: Cell,
               t_now: int, parking: Cell, tasks: Sequence[Task],
               emergency: Optional[EmergencySpec] = None, hold: int = 0,
               T_limit: int = 600, strict_park: bool = True,
               soft_lambda: Optional[int] = None,
               clash_sink: Optional[Set[int]] = None
               ) -> Optional[List[PlanStep]]:
    """``plan_agent`` with an optional up-front hold and/or emergency detour.

    ``hold = d > 0`` keeps the agent at ``start`` for ticks
    ``t_now .. t_now + d - 1`` (an accident/breakdown) and resumes at
    ``t_now + d`` -- the disruption is a *constraint* on the new plan rather
    than something baked into the frozen prefix.

    ``soft_lambda``/``clash_sink`` turn this into the *negotiation proposal*
    of ``SPEC.md`` 9.2: the route may cut through other agents' reservations
    (each clash costs ``soft_lambda``) and the clashed agents end up in
    ``clash_sink`` so they can be asked to yield.
    """
    waits: List[PlanStep] = []
    base = t_now
    if hold > 0:
        waits = [PlanStep(t_now + k, start, WAIT) for k in range(1, hold)]
        base = t_now + hold

    if emergency is None:
        rest = plan_agent(world, res, aid, start, parking, list(tasks),
                          t0=base, T_limit=T_limit, strict_park=strict_park,
                          soft_lambda=soft_lambda, clash_sink=clash_sink)
        return None if rest is None else waits + rest

    steps: List[PlanStep] = waits + [PlanStep(base, start, WAIT)]
    cur, tc = start, base
    hold_ticks = max(1, emergency.hold)
    leg, clashed = planner.st_astar(world, res, aid, cur, emergency.cell, tc,
                                    T_limit=T_limit, hold_after=hold_ticks,
                                    soft_lambda=soft_lambda)
    if leg is None:
        return None
    if clash_sink is not None:
        clash_sink |= clashed
    for (c, tt) in leg[1:]:
        steps.append(PlanStep(tt, c, MOVE))
    cur, tc = leg[-1]
    for _ in range(hold_ticks):
        tc += 1
        steps.append(PlanStep(tc, cur, WAIT))
    rest = plan_agent(world, res, aid, cur, parking, list(tasks),
                      t0=tc, T_limit=T_limit, strict_park=strict_park,
                      soft_lambda=soft_lambda, clash_sink=clash_sink)
    if rest is None:
        return None
    steps.extend(rest[1:])
    return steps


def previous_order(prev_steps: Optional[Sequence[PlanStep]],
                   tasks: Sequence[Task]) -> List[Task]:
    """The agent's task order as evidenced by its own plan (PICK order)."""
    seen: List[Task] = []
    for s in (prev_steps or ()):
        if s.action == PICK and s.item is not None:
            tk = next((t for t in tasks if t.id == s.item), None)
            if tk is not None and tk not in seen:
                seen.append(tk)
    for t in tasks:
        if t not in seen:
            seen.append(t)
    return seen


def reground_agent(world: World, res: ReservationTable, aid: int,
                   start: Cell, t_now: int, parking: Cell,
                   tasks: Sequence[Task], prev_steps: Optional[List[PlanStep]],
                   *, T_limit: int = 600, delay_threshold: int = 8,
                   emergency: Optional[EmergencySpec] = None,
                   allow_reorder: bool = True, hold: int = 0,
                   soft_lambda: Optional[int] = None,
                   clash_sink: Optional[Set[int]] = None
                   ) -> Tuple[Optional[List[PlanStep]], str]:
    """Try the R0-R3 rungs for one agent; return ``(steps, level)``.

    ``steps`` is the *remaining* plan (``t > t_now``); ``level`` is the rung
    that produced it, or ``"R4"`` when nothing worked (the caller then hands the
    orphan goals to another agent).  ``soft_lambda``/``clash_sink`` are the
    negotiation proposal of ``SPEC.md`` 9.2 and are forwarded to
    :func:`_plan_from`.
    """
    order = previous_order(prev_steps, tasks)
    cand = _plan_from(world, res, aid, start, t_now, parking, order,
                      emergency, hold, T_limit, soft_lambda=soft_lambda,
                      clash_sink=clash_sink)
    if cand is not None:
        return cand, _label(prev_steps, cand, t_now, delay_threshold)

    if allow_reorder and soft_lambda is None:
        reordered = greedy_task_order(world, start, tasks)
        cand = _plan_from(world, res, aid, start, t_now, parking, reordered,
                          emergency, hold, T_limit)
        if cand is not None:
            return cand, "R3"
    return None, "R4"




# ----------------------------------------------------------------------
# R4: hand an orphan goal to another agent
# ----------------------------------------------------------------------
def handoff_task(world: World, res: ReservationTable,
                 plans: Dict[int, Optional[List[PlanStep]]],
                 broke: int, orphan: Task, t_now: int, parking: Sequence[Cell],
                 owned: Dict[int, List[Task]],
                 *, comm_radius: int = 6, k_candidates: int = 3,
                 T_limit: int = 600, max_nodes: int = 3000
                 ) -> Optional[Tuple[int, List[PlanStep]]]:
    """Reassign ``orphan`` to the cheapest nearby agent.

    Return ``(receiver, receiver_tail)`` or ``None``.  ``owned`` is the
    current task list of every agent (so a receiver keeps the orphans it was
    given earlier in the same pass).  The receiver's plan is rebuilt with POCL
    started from *its own current state* (never the global planner), exactly
    as SPEC 9.3 requires.
    """
    from peg import ground_pop

    here = position_at(plans.get(broke), t_now)
    if here is None:
        return None
    cands = []
    for aid in sorted(plans):
        if aid == broke or plans.get(aid) is None:
            continue
        d = world.distance(here, position_at(plans[aid], t_now, parking[aid]))
        if d <= comm_radius:
            cands.append((d, aid))
    cands.sort()

    best: Optional[Tuple[int, List[PlanStep]]] = None
    best_cost: Optional[int] = None
    for _, aid in cands[:k_candidates]:
        start = position_at(plans[aid], t_now, parking[aid])
        extra = list(owned.get(aid, [])) + [orphan]

        pop = pocl(aid, extra, start, world, max_nodes=max_nodes)
        tail_new = None
        if pop is not None:
            tail_new = ground_pop(world, res, aid, start, parking[aid], pop,
                                  t_now, T_limit, park=True)
        if tail_new is None:
            tail_new = plan_agent(world, res, aid, start, parking[aid],
                                  greedy_task_order(world, start, extra),
                                  t_now, T_limit)
        if tail_new is None:
            continue
        cost = tail_new[-1].t - t_now
        if best_cost is None or cost < best_cost:
            best, best_cost = (aid, tail_new), cost
    return best




# ----------------------------------------------------------------------
# schedule-level repair: the entry point the simulator calls
# ----------------------------------------------------------------------
def freeze_tick(disruptions: Sequence[Disruption], t_sim: int) -> int:
    """Tick up to which plans are frozen.

    Repair always happens *at* the event tick: the executed prefix (``t``
    smaller than or equal to the event) is untouchable and an
    accident/breakdown hold is re-imposed as a constraint during regrounding
    (see :func:`_plan_from`), never baked into the frozen prefix.
    """
    t = t_sim
    for d in disruptions:
        t = max(t, d.t)
    return t


def _wait_forever(steps: Sequence[PlanStep], cell: Cell, t_from: int,
                  failure_wait: int) -> List[PlanStep]:
    """Agent gives up: it sits still for ``failure_wait`` ticks (a failure)."""
    return list(steps) + [PlanStep(t_from + i, cell, WAIT)
                          for i in range(1, failure_wait + 1)]


def _record(prev: Dict[int, Optional[List[PlanStep]]],
            new_plans: Dict[int, Optional[List[PlanStep]]],
            frozen: Dict[int, List[PlanStep]], t_freeze: int,
            A0: Sequence[int], event: str, level_resolved: str, success: bool,
            c0: float, calls0: int, nodes0: int,
            log: Optional[Negotiation] = None) -> RepairRecord:
    """Assemble the ``SPEC.md`` 9.5 repair record for one repair pass.

    ``event`` is the disruption kind(s) that triggered the pass (``accident``,
    ``emergency``, ``blockage``, ``conflict``, ...); ``log`` is the negotiation
    message log of the pass, counted by type in the record.
    """
    sets = altered_agent_sets(prev, new_plans, t_freeze, disrupted=A0)
    return RepairRecord(
        event_type=event,
        t=t_freeze,
        A0_size=len(A0),
        level_resolved=level_resolved,
        altered_plan_ids=sets["altered_plan"],
        altered_path_ids=sets["altered_path"],
        delayed_ids=sets["delayed_only"],
        naive_ids=sets["altered_naive"],
        added_delay_total=total_added_delay(prev, new_plans),
        messages_by_type=log.by_type() if log is not None else {},
        pocl_nodes=planner.ST_ASTAR_NODES - nodes0,
        st_astar_calls=planner.ST_ASTAR_CALLS - calls0,
        cpu_ms=(time.perf_counter() - c0) * 1000.0,
        success=success,
    )


def repair_schedule(
    world: World,
    plans: Dict[int, Optional[List[PlanStep]]],
    tasks_by_agent: Dict[int, List[Task]],
    parking: Sequence[Cell],
    *,
    t_now: int = 0,
    disruptions: Sequence[Disruption] = (),
    mode: str = "negotiate",
    comm_radius: int = 6,
    delay_threshold: int = 8,
    T_limit: int = 600,
    repair_max_nodes: int = 3000,
    failure_wait: int = 60,
    k_candidates: int = 3,
    lambda_soft: int = 15,
    max_depth: int = 3,
    beta_alter: int = 10,
    message_log: Optional[Negotiation] = None,
    emergency_specs: Optional[Dict[int, EmergencySpec]] = None,
    hold_until: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, Optional[List[PlanStep]]], List[RepairRecord]]:
    """Repair ``plans`` after ``disruptions`` -- locally, without the global planner.

    Modes (``Config.replan_mode``):

    * ``negotiate``  -- replan the disrupted agents plus every agent within
      ``comm_radius``; all others are frozen (SPEC 9.2 / 9.4).  The disrupted
      agents additionally *negotiate* their route with the neighbours they
      would cut through (:mod:`negotiation`), up to ``max_depth`` rungs of
      radius widening, and only when the saving pays for the extra altered
      agents (``beta_alter``).
    * ``self_only``  -- replan only the disrupted agents (R0-R2 + wait fallback).
    * ``global``     -- quality upper bound: re-plan every remaining goal.  This
      baseline is the one place allowed to call the global planner.
    * ``timeindexed`` / ``static`` -- ablation: no repair at all.
    * ``bfs``        -- like ``negotiate`` but priority order is BFS distance to
      the hotspot (nearest agent first).

    ``hold_until`` maps an agent to the absolute tick at which an
    accident/breakdown releases it; the remaining hold is re-imposed on the
    replanned tail.  ``message_log`` is the negotiation log of the pass (the
    simulator passes one per scenario) and is created when omitted; it ends up
    in ``RepairRecord.messages_by_type``.
    """
    c0 = time.perf_counter()
    calls0, nodes0 = planner.ST_ASTAR_CALLS, planner.ST_ASTAR_NODES
    n = len(parking)
    parking = list(parking)
    A0 = sorted({d.agent for d in disruptions})
    t_freeze = freeze_tick(disruptions, t_now)
    hold_until = dict(hold_until or {})
    event = "+".join(sorted({d.kind for d in disruptions})) or "none"
    log = message_log if message_log is not None else Negotiation(t=t_freeze)
    log.t = t_freeze        # stamp this pass' messages with the pass tick, so a
    #                         caller-supplied log replays correctly in the viz

    frozen = {aid: freeze(plans.get(aid), t_freeze) for aid in range(n)}
    rem = {aid: pending_tasks(plans.get(aid), tasks_by_agent.get(aid, []), t_freeze)
           for aid in range(n)}
    hot_window: Dict[Cell, int] = {}
    for d in disruptions:
        c = position_at(plans.get(d.agent), t_freeze, parking[d.agent])
        if c is not None:
            hot_window[c] = max(hot_window.get(c, t_freeze),
                                hold_until.get(d.agent, t_freeze))
    hot = sorted(hot_window)

    # --- ablation: nothing is touched ------------------------------------
    if mode in ("static", "timeindexed"):
        new_plans = {aid: merge(frozen[aid], tail_steps(plans.get(aid), t_freeze))
                     for aid in range(n)}
        rec = RepairRecord(event_type=event, t=t_freeze, A0_size=len(A0),
                           level_resolved="R1_static", success=False,
                           messages_by_type=log.by_type(),
                           cpu_ms=(time.perf_counter() - c0) * 1000.0)
        return new_plans, [rec]

    # --- seed a reservation table with everyone's frozen prefix ----------
    res = ReservationTable()
    for aid in range(n):
        if frozen[aid]:
            done = (not rem[aid][0]) and rem[aid][1] is None
            res.add_path(aid, plan_to_path(frozen[aid]), park=done)

    # Agents that cannot move (accident/breakdown hold) are pinned in place
    # for the whole hold, so every other agent must route around them --
    # else a low-id agent may claim the held cell at the release tick.
    for aid in range(n):
        end = hold_until.get(aid, t_freeze)
        if end <= t_freeze or not frozen[aid]:
            continue
        c = position_at(frozen[aid], t_freeze, parking[aid])
        pinned = [(c, t) for t in range(t_freeze, end + 1)]
        res.add_path(aid, pinned, park=False)

    def resume_order(aid: int) -> List[Task]:
        """The agent's remaining work, with a carried item put first.

        A task whose item was picked up before the freeze tick is *not* in
        ``rem``; it is re-expressed as a fresh task from the agent's current
        cell so it can still be delivered (and handed off if need be).
        """
        todo, holding = rem[aid]
        if holding is None:
            return list(todo)
        here = position_at(frozen[aid], t_freeze, parking[aid])
        return [Task(holding.id, here, holding.delivery)] + list(todo)

    # --- mode "global": full re-plan from the current state --------------
    if mode == "global":
        from baselines import global_plan
        from planner import greedy_task_order
        # An accident hold is physical: bake the remaining hold ticks into the
        # frozen prefix so the global re-plan must respect them.
        for aid in range(n):
            h = max(0, hold_until.get(aid, t_freeze) - t_freeze)
            if h <= 0 or not frozen[aid]:
                continue
            c = position_at(frozen[aid], t_freeze, parking[aid])
            extra = [PlanStep(t_freeze + k, c, WAIT) for k in range(1, h)]
            frozen[aid] = frozen[aid] + extra
            res.add_path(aid, plan_to_path(extra), park=False)
        jobs = []
        for aid in range(n):
            if plans.get(aid) is None:
                continue
            start = position_at(frozen[aid], t_freeze, parking[aid])
            jobs.append((aid, start, parking[aid],
                         greedy_task_order(world, start, resume_order(aid))))
        got, failed = global_plan(world, res.copy(), jobs, t_freeze, T_limit)
        out: Dict[int, Optional[List[PlanStep]]] = {}
        for aid in range(n):
            st = got.get(aid)
            if st is None:
                start = position_at(frozen[aid], t_freeze, parking[aid])
                out[aid] = (_wait_forever(frozen[aid], start, t_freeze, failure_wait)
                            if frozen[aid] else None)
            else:
                out[aid] = merge(frozen[aid], tail_steps(st, t_freeze))
        return out, [_record(plans, out, frozen, t_freeze, A0, event,
                             "R2_restart", not failed, c0, calls0, nodes0,
                             log)]

    # --- who gets replanned? --------------------------------------------
    near = set(A0)
    if hot:
        for aid in range(n):
            c = position_at(frozen[aid], t_freeze, parking[aid])
            if c is not None and any(world.distance(c, h) <= comm_radius for h in hot):
                near.add(aid)
                continue
            # An agent that drives *through* a held cell while it is held must
            # also be replanned, even if it is far away right now.
            for s in tail_steps(plans.get(aid), t_freeze):
                if hot_window.get(s.cell, -1) >= s.t:
                    near.add(aid)
                    break
    if mode == "self_only":
        replan_set = set(A0)
    elif mode in ("negotiate", "bfs"):
        replan_set = set(A0) | near
    else:
        raise ValueError(f"unknown repair mode {mode!r}")

    new_plans: Dict[int, Optional[List[PlanStep]]] = {}
    # everyone else is frozen: keep the original tail *and* reserve it, so the
    # replanned agents must route around it (SPEC 9.2 "others are frozen").
    for aid in range(n):
        if aid in replan_set:
            continue
        new_plans[aid] = merge(frozen[aid], tail_steps(plans.get(aid), t_freeze))
        tl = tail_steps(plans.get(aid), t_freeze)
        if tl:
            res.add_path(aid, plan_to_path(tl), park=True)

    def priority(aid: int):
        if mode == "bfs" and hot:
            c = position_at(frozen[aid], t_freeze, parking[aid])
            d = min((world.distance(c, h) for h in hot), default=0.0)
            return (0 if aid in A0 else 1, d, aid)
        return (0 if aid in A0 else 1, aid)

    order = sorted(replan_set, key=priority)
    #: priority rank inside this pass; the negotiation uses it to decide who
    #: can yield ("j is replanned after i, so j waits" -- SPEC 9.2)
    rank: Dict[int, int] = {aid: k for k, aid in enumerate(order)}
    levels: List[str] = []
    stalled: List[int] = []
    has_perm = {d.agent for d in disruptions if d.permanent}
    #: tasks currently owned by each agent (updated as orphans are handed over)
    owned: Dict[int, List[Task]] = {aid: resume_order(aid) for aid in range(n)}
    #: agents that can never move again during this pass (permanent breakdown,
    #: or an agent that has just been declared dead below)
    fixed: Set[int] = set(has_perm)
    negotiable = mode in ("negotiate", "bfs")

    def negotiate_move(aid: int, start: Cell, todo: Sequence[Task],
                       emg: Optional[EmergencySpec], hold: int,
                       hard: List[PlanStep]) -> Optional[List[PlanStep]]:
        """SPEC 9.2: ask the neighbours the initiator cuts through to yield.

        The proposal is a soft-ST-A* tour over a table that also carries the
        *pre-repair* tails of the agents replanned later in this pass -- they
        are exactly the agents that can yield, and they yield by being
        re-derived against the reservations the proposal books.  ``hard`` is
        the ladder's own candidate; a grant is only taken when it saves more
        than it costs (:func:`negotiation.grant_is_worthwhile`).
        """
        res_pre = res.copy()
        for later in order[rank[aid] + 1:]:
            tl = tail_steps(plans.get(later), t_freeze)
            if tl and later in replan_set and later not in fixed:
                res_pre.add_path(later, plan_to_path(tl), park=True)
        clash: Set[int] = set()
        prop = _plan_from(world, res_pre, aid, start, t_freeze, parking[aid],
                          todo, emg, hold, T_limit, soft_lambda=lambda_soft,
                          clash_sink=clash)
        if prop is None or not clash:
            return None
        #: SPEC 9.2 branch 2: the holders that reroute their own leg for ``aid``
        repaired: Dict[int, List[PlanStep]] = {}

        def reroute_cost(holder: int) -> Optional[int]:
            """Re-derive ``holder``'s tail around the proposal (others frozen)."""
            res_try = res.copy()
            res_try.remove_agent_after(holder, t_freeze + 1)
            res_try.add_path(aid, plan_to_path(prop), park=True)
            for h, tl in repaired.items():
                res_try.add_path(h, plan_to_path(tl), park=True)
            j_start = position_at(frozen[holder], t_freeze, parking[holder])
            j_tail, _ = reground_agent(
                world, res_try, holder, j_start, t_freeze, parking[holder],
                owned[holder], plans.get(holder), T_limit=T_limit,
                delay_threshold=delay_threshold,
                hold=max(0, hold_until.get(holder, t_freeze) - t_freeze))
            if j_tail is None:
                return None
            repaired[holder] = [s for s in j_tail if s.t > t_freeze]
            return completion_delta(plans.get(holder), j_tail)

        verdicts, extra = negotiation.negotiate(
            aid, clash,
            distance=lambda j: world.distance(
                start, position_at(frozen[j], t_freeze, parking[j])),
            replan_set=replan_set, order_index=rank,
            comm_radius=comm_radius, res=res_pre, path=prop,
            max_depth=max_depth, delay_threshold=delay_threshold,
            fixed=fixed, hold_end=hold_until, reroute_cost=reroute_cost,
            log=log)
        holders = negotiation.granted_holders(verdicts)
        if len(holders) != len(verdicts):
            log.abort(aid, [v.holder for v in verdicts if not v.granted],
                      "at least one holder rejected")
            return None
        loser = negotiation.first_clash(res, aid, prop, ignore=set(holders))
        if loser is None:
            for h in sorted(repaired):
                bad = negotiation.paths_clash(prop, repaired[h])
                if bad is not None:
                    loser = (bad[0], bad[1], h)
                    break
        if loser is not None:
            log.abort(aid, holders, f"clash with {loser[2]} at {loser[0]}")
            return None
        n_new = len((set(holders) | extra) - set(A0))
        if hard is not None and not negotiation.grant_is_worthwhile(
                hard_completion=hard[-1].t, prop_completion=prop[-1].t,
                n_new_altered=n_new, beta_alter=beta_alter,
                delay_threshold=delay_threshold):
            log.abort(aid, holders, "the saving does not pay for the agents "
                                    "it alters")
            return None
        # ``hard is None`` means the ladder found no candidate at all (it would
        # hand the task over): a granted proposal is then a plain win.
        log.commit(aid, holders, "the initiator goes first"
                   if hard is not None else "the ladder had no candidate")
        for h, tl in repaired.items():     # two-phase COMMIT: holders too
            res.remove_agent_after(h, t_freeze + 1)
            new_plans[h] = merge(frozen[h], tl)
            res.add_path(h, plan_to_path(tl), park=True)
        for j in sorted(extra):        # rung R3: the pulled-in holders
            if j in rank:
                continue
            res.remove_agent_after(j, t_freeze + 1)
            rank[j] = len(order)
            order.append(j)
            replan_set.add(j)
            owned.setdefault(j, resume_order(j))
        return [s for s in prop if s.t > t_freeze]

    cursor = 0
    while cursor < len(order):
        aid = order[cursor]
        cursor += 1
        if plans.get(aid) is None:
            new_plans[aid] = None
            stalled.append(aid)
            continue
        todo = owned[aid]
        start = position_at(frozen[aid], t_freeze, parking[aid])
        emg = (emergency_specs or {}).get(aid)
        hold = max(0, hold_until.get(aid, t_freeze) - t_freeze)

        if aid in has_perm:
            tail, level = None, "R4"
        else:
            tail, level = reground_agent(
                world, res, aid, start, t_freeze, parking[aid], todo,
                plans.get(aid), T_limit=T_limit, delay_threshold=delay_threshold,
                emergency=emg, allow_reorder=(mode != "self_only"), hold=hold)
            if tail is not None:
                tail = [s for s in tail if s.t > t_freeze]   # drop frozen dupes
            if negotiable and aid in A0:
                granted = negotiate_move(aid, start, todo, emg, hold, tail)
                if granted is not None:
                    level = "R2"     # SPEC 9.1: the negotiation rung
                    tail = granted

        if tail is not None:
            levels.append(level)
            new_plans[aid] = merge(frozen[aid], tail)
            res.add_path(aid, plan_to_path(tail), park=True)
            continue

        # ---- R4: hand the orphan goals to other agents ------------------
        # Reserve the failed agent first: it blocks its cell for the rest of
        # the run, so every receiver must route around it.
        fixed.add(aid)                # it is out of action for good
        dead = _wait_forever(frozen[aid], start, t_freeze, failure_wait)
        res.add_path(aid, plan_to_path([s for s in dead if s.t > t_freeze]),
                     park=True)
        orphans: List[Task] = list(todo)
        handed = True                 # nothing to hand over is not a failure
        for orph in orphans:
            got = handoff_task(world, res, plans, aid, orph, t_freeze, parking,
                               {a: owned.get(a, []) for a in range(n)},
                               comm_radius=comm_radius,
                               k_candidates=k_candidates, T_limit=T_limit,
                               max_nodes=repair_max_nodes)
            if got is None:
                handed = False
                break
            rid, rtail = got
            res.remove_agent(rid)
            new_plans[rid] = merge(frozen[rid], rtail)
            res.add_path(rid, plan_to_path(new_plans[rid]), park=True)
            owned.setdefault(rid, []).append(orph)

        levels.append("R4")
        if not handed:
            stalled.append(aid)
        new_plans[aid] = dead

    level_resolved = max(levels, key=LEVELS.index) if levels else "R0"
    rec = _record(plans, new_plans, frozen, t_freeze, A0, event, level_resolved,
                  not stalled, c0, calls0, nodes0, log)
    return new_plans, [rec]
