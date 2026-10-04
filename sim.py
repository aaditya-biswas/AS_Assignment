"""Discrete-time, time-expanded simulator and metrics (``SPEC.md`` section 9).

Time advances in integer ticks (one tick = one shortest path step, which is
the same abstraction the planner uses).  Every agent's ``PlanStep`` list is
padded with its final cell so the trace is defined for the whole makespan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from planner import DROP, WAIT, PlanStep
from world import Cell, Task, World


# ----------------------------------------------------------------------
# metrics container
# ----------------------------------------------------------------------
@dataclass
class SimResult:
    makespan: int
    completed: int
    total_tasks: int
    failed_init: int = 0
    failed_act: int = 0
    sum_service: float = 0.0
    violations: List[str] = field(default_factory=list)
    completion_times: Dict[int, int] = field(default_factory=dict)
    traces: Optional[Dict[int, List[Cell]]] = None
    stalled: List[int] = field(default_factory=list)
    repairs: int = 0

    @property
    def throughput(self) -> float:
        return self.completed / self.total_tasks if self.total_tasks else 1.0

    @property
    def mean_service(self) -> float:
        return self.sum_service / self.completed if self.completed else float("nan")

    @property
    def conflict_free(self) -> bool:
        return not self.violations


# ----------------------------------------------------------------------
# traces
# ----------------------------------------------------------------------
def build_traces(plans: Dict[int, Optional[List[PlanStep]]],
                 makespan: int) -> Dict[int, List[Cell]]:
    traces: Dict[int, List[Cell]] = {}
    for aid, steps in plans.items():
        if not steps:
            continue
        pos = {s.t: s.cell for s in steps}
        first = steps[0].cell
        last = steps[-1].cell
        start_t = steps[0].t
        traces[aid] = [
            pos[t] if t in pos else (first if t < start_t else last)
            for t in range(makespan + 1)
        ]
    return traces


def plan_makespan(plans: Dict[int, Optional[List[PlanStep]]]) -> int:
    real = [s[-1].t for s in plans.values() if s]
    return max(real) if real else 0


# ----------------------------------------------------------------------
# running
# ----------------------------------------------------------------------
def run_sim(world: World, plans: Dict[int, Optional[List[PlanStep]]],
            makespan: Optional[int] = None,
            stalled_agents: Sequence[int] = (),
            record_traces: bool = False,
            total_tasks: Optional[int] = None,
            failed_init: int = 0,
            repairs: int = 0) -> SimResult:
    """Execute ``plans`` and validate every tick for the six safety rules."""
    real = {a: s for a, s in plans.items() if s}
    if makespan is None:
        makespan = plan_makespan(plans)
    traces = build_traces(real, makespan)
    stalled = set(stalled_agents)

    violations: List[str] = []
    # --- per-tick safety ------------------------------------------------
    for t in range(makespan + 1):
        occupied: Dict[Cell, int] = {}
        for aid in sorted(traces):
            c = traces[aid][t]
            if world.is_shelf(c):
                violations.append(f"t={t}: agent {aid} on shelf {c}")
            if world.is_blocked(c, t):
                violations.append(f"t={t}: agent {aid} inside blockage {c}")
            other = occupied.get(c)
            if other is not None:
                violations.append(f"t={t}: vertex clash at {c} "
                                  f"(agents {other} & {aid})")
            occupied[c] = aid
        if t < makespan:
            for a in sorted(traces):
                if traces[a][t] == traces[a][t + 1]:
                    continue
                for b in sorted(traces):
                    if b <= a:
                        continue
                    if traces[b][t] == traces[a][t + 1] and \
                       traces[b][t + 1] == traces[a][t]:
                        violations.append(
                            f"t={t}: swap clash {traces[a][t]}<->{traces[a][t+1]} "
                            f"(agents {a} & {b})")

    # --- task accounting ------------------------------------------------
    completion: Dict[int, int] = {}
    failed_act = 0
    seen_drop: set = set()
    for aid in sorted(real):
        if aid in stalled:
            continue
        for s in real[aid]:
            if s.action == DROP and s.item is not None:
                if s.item in seen_drop:
                    failed_act += 1     # duplicated delivery of the same item
                seen_drop.add(s.item)
                completion[s.item] = s.t
    completed = len(completion)
    if total_tasks is None:
        total_tasks = completed + failed_init

    # --- stuck agents (livelock/lock detection) -------------------------
    stalled_out: List[int] = []
    for aid in sorted(real):
        steps = real[aid]
        if not steps:
            continue
        if steps[-1].t > makespan:
            stalled_out.append(aid)

    return SimResult(
        makespan=makespan,
        completed=completed,
        total_tasks=total_tasks,
        failed_init=failed_init,
        failed_act=failed_act,
        sum_service=float(sum(completion.values())),
        violations=violations,
        completion_times=completion,
        traces=traces if record_traces else None,
        stalled=stalled_out,
        repairs=repairs,
    )


def scan_violation(world: World, plans: Dict[int, Optional[List[PlanStep]]],
                   t_from: int = 0
                   ) -> Optional[Tuple[int, List[int]]]:
    """First tick at/after ``t_from`` where the schedule breaks a safety rule.

    Returns ``(tick, agents)`` or ``None``.  Used by the repair loop to detect
    the *residual* conflicts a local repair pass did not remove.
    """
    real = {a: s for a, s in plans.items() if s}
    makespan = plan_makespan(plans)
    traces = build_traces(real, makespan)
    for t in range(max(0, t_from), makespan + 1):
        occupied: Dict[Cell, int] = {}
        for aid in sorted(traces):
            c = traces[aid][t]
            if world.is_shelf(c) or world.is_blocked(c, t):
                return t, [aid]
            other = occupied.get(c)
            if other is not None:
                return t, [min(other, aid), max(other, aid)]
            occupied[c] = aid
        if t < makespan:
            for a in sorted(traces):
                if traces[a][t] == traces[a][t + 1]:
                    continue
                for b in sorted(traces):
                    if b <= a:
                        continue
                    if traces[b][t] == traces[a][t + 1] and \
                       traces[b][t + 1] == traces[a][t]:
                        return t, [a, b]
    return None


# ----------------------------------------------------------------------
# disruption scenario driver (SPEC section 10)
# ----------------------------------------------------------------------
def simulate_scenario(
    world: World,
    plans: Dict[int, Optional[List[PlanStep]]],
    tasks_by_agent: Dict[int, List[Task]],
    parking: Sequence[Cell],
    *,
    mode: str = "negotiate",
    blockages: Sequence[object] = (),
    accidents: Sequence[object] = (),
    emergencies: Sequence[object] = (),
    rng=None,
    max_repairs: int = 8,
    total_tasks: Optional[int] = None,
    record_traces: bool = False,
    **repair_kwargs,
) -> Tuple[SimResult, List[object], Dict[int, Optional[List[PlanStep]]]]:
    """Apply a disruption scenario, repair it, and simulate the result.

    Returns ``(SimResult, RepairRecords, final_plans)``.  ``record_traces``
    keeps the per-tick cell track of every agent in ``SimResult.traces`` (the
    visualisation of :mod:`viz` and the trace schema of ``SPEC.md`` 11).

    The loop follows ``SPEC.md`` section 10:

    1. exogenous blockages are written into the ``World`` (they make travel
       time-dependent for everyone) -- events that would trap an agent inside
       an already-executed step are postponed ("postpone block if occupied");
    2. the event ticks are processed in **time order**; each pass replans the
       disrupted agents plus their neighbourhood
       (:func:`modify.repair_schedule`) and then sweeps the residual safety
       violations belonging to that same time window (at most ``max_repairs``
       sweeps per window).  Only disruptions that have already happened at a
       pass's tick constrain that pass: an accident at t=18 does not freeze
       anyone during a repair at t=3;
    3. the schedule is executed and validated.

    ``mode in ("static", "timeindexed")`` is the v1 ablation: accidents are
    injected by plan surgery and nothing is repaired.
    """
    from disruptions import (inject_accidents, inject_blockages,
                             inject_emergencies, postpone_blockages)
    from modify import Disruption, position_at, repair_schedule

    if not isinstance(tasks_by_agent, dict):        # accept list-per-agent too
        tasks_by_agent = {i: list(v) for i, v in enumerate(tasks_by_agent)}

    # SPEC 10: "postpone block if occupied" -- never announce a blockage that
    # would trap an agent inside an already-executed step.
    blockages = postpone_blockages(plans, list(blockages)) if blockages else []
    inject_blockages(world, blockages)
    reactive = mode not in ("static", "timeindexed")
    if not reactive:                    # ablation: surgery only, no repair
        plans = inject_accidents(plans, accidents)
    specs = (inject_emergencies(world, plans, emergencies, rng)
             if reactive else {})

    #: absolute tick at which an accident starts / releases the agent
    acc_at: Dict[int, int] = {}
    acc_end: Dict[int, int] = {}
    for ev in accidents:
        acc_at[ev.agent] = min(acc_at.get(ev.agent, ev.at_t), ev.at_t)
        acc_end[ev.agent] = max(acc_end.get(ev.agent, 0),
                                ev.at_t + ev.duration)
    emg_at = {ev.agent: ev.at_t for ev in emergencies}
    #: tick at which a planned emergency detour + hold resumes the agent
    emg_end: Dict[int, int] = {}

    events: List[object] = []
    for ev in accidents:
        events.append(Disruption("breakdown" if ev.permanent else "accident",
                                 ev.agent, ev.at_t, ev.duration,
                                 permanent=ev.permanent))
    for ev in emergencies:
        events.append(Disruption("emergency", ev.agent, ev.at_t, hold=ev.hold))
    for ev in blockages:                # exogenous: whoever drives through it
        for aid, steps in plans.items():
            if steps and any(s.cell == ev.cell and ev.t0 <= s.t < ev.t1
                             for s in steps):
                events.append(Disruption("blockage", aid, ev.t0,
                                         duration=ev.duration, cell=ev.cell))

    by_t: Dict[int, List[object]] = {}
    for d in events:
        by_t.setdefault(d.t, []).append(d)

    def context(cur: Dict[int, Optional[List[PlanStep]]], t: int):
        """Constraints that are *valid* at repair tick ``t``.

        Returns ``(emergency_specs, hold_until)``.  An accident holds its agent
        only from ``at_t`` onwards; an emergency demands its detour until that
        detour (with its marshalling hold) has been executed inside the frozen
        prefix.
        """
        now_specs: Dict[int, object] = {}
        now_hold: Dict[int, int] = {}
        for a, end in acc_end.items():
            if acc_at[a] <= t:
                now_hold[a] = max(now_hold.get(a, 0), end)
        for a, spec in specs.items():
            if t < emg_at[a]:
                continue                        # not announced yet
            end = emg_end.get(a)
            if end is not None and t >= end:
                continue                        # detour already executed
            here = position_at(cur.get(a), t, parking[a])
            if end is not None and here == spec.cell:
                now_hold[a] = max(now_hold.get(a, 0), end)    # still holding
            else:
                now_specs[a] = spec             # (re)perform the detour
        return now_specs, now_hold

    def do_pass(cur, t: int, disruptions):
        now_specs, now_hold = context(cur, t)
        return repair_schedule(world, cur, tasks_by_agent, parking, t_now=t,
                               disruptions=disruptions, mode=mode,
                               emergency_specs=now_specs, hold_until=now_hold,
                               **repair_kwargs)

    def sweep(cur, ceiling: Optional[int]):
        """Repair the residual violations below ``ceiling`` (``None``: no cap)."""
        recs: List[object] = []
        for _ in range(max_repairs):
            found = scan_violation(world, cur)
            if found is None:
                break
            tv, agents = found
            if ceiling is not None and tv >= ceiling:
                break                           # a later event pass owns it
            t_rep = max(0, tv - 1)
            cur, rr = do_pass(cur, t_rep, [Disruption("conflict", a, t_rep)
                                           for a in agents])
            recs.extend(rr)
        return cur, recs

    records: List[object] = []
    if reactive:
        ticks = sorted(by_t)
        for i, te in enumerate(ticks):
            plans, recs = do_pass(plans, te, by_t[te])
            records.extend(recs)
            # how long does a just-planned emergency hold last?
            for d in by_t[te]:
                spec = specs.get(d.agent)
                if d.kind != "emergency" or spec is None:
                    continue
                waits = [s.t for s in (plans.get(d.agent) or [])
                         if s.cell == spec.cell and s.action == WAIT]
                if waits:
                    emg_end[d.agent] = max(waits) + 1
            ceiling = ticks[i + 1] if i + 1 < len(ticks) else None
            plans, recs = sweep(plans, ceiling)
            records.extend(recs)
        plans, recs = sweep(plans, None)     # anything left after the events
        records.extend(recs)

    if total_tasks is None:
        total_tasks = sum(len(v) for v in tasks_by_agent.values())
    res = run_sim(world, plans, total_tasks=total_tasks, repairs=len(records),
                  record_traces=record_traces)
    return res, records, plans


def active_blockages_at_end(world: World, makespan: int) -> int:
    return len(world.active_blockages(makespan))
