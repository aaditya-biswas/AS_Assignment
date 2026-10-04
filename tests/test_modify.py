"""Dynamic disruptions and the four replanning modes (``SPEC.md`` 10-11).

Two invariants dominate this file:

* **Repair never calls the global planner.**  ``test_repair_never_uses_the_global_planner``
  monkeypatches ``planner.prioritized_ground`` to raise, so any leak surfaces
  as an immediate error, and additionally asserts that
  ``baselines.GLOBAL_PLAN_CALLS`` stays 0.
* **Repair is safe.**  Every reactive scenario must end conflict-free with
  every task delivered; only the ``static`` ablation may leave violations.

The escalation ladder is *labelled* from the candidate a local regrounding
returns (``R0`` waits only, ``R1``/``R2`` detours, ``R3`` re-derived order,
``R4`` hand-off), so the rungs are tested directly on ``reground_agent`` and
``_label`` as well as end to end.
"""
from __future__ import annotations

import numpy as np
import pytest

import baselines
import planner
from config import Config
from disruptions import (AccidentEvent, BlockageEvent, EmergencyEvent,
                         inject_blockages, postpone_blockages)
from metrics import (ALTERED_PATH, DELAYED_ONLY, UNCHANGED, altered_agent_sets,
                     classify, completion_delta, route_signature)
from modify import (LEVELS, Disruption, _label, freeze, handoff_task,
                    pending_tasks, position_at, reground_agent,
                    repair_schedule)
from peg import peg_solve
from planner import DROP, MOVE, PICK, WAIT, PlanStep, plan_agent, st_astar
from reservation import ReservationTable
from sim import plan_makespan, run_sim, scan_violation, simulate_scenario
from world import Task, TaskStatus, World, assign_tasks, build_world


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _cfg(seed: int = 0, H: int = 20, W: int = 20, n: int = 6, k: int = 2):
    return Config(H=H, W=W, n_agents=n, tasks_per_agent=k, seed=seed)


def _scenario(seed: int = 0, **kw):
    """A small, feasible, disruption-free scenario."""
    cfg = _cfg(seed=seed, **kw)
    rng = np.random.default_rng(cfg.seed)
    world, tasks, parking = build_world(cfg, rng)
    per_agent = assign_tasks(tasks, parking, rng)
    plans, failed = peg_solve(world, per_agent, parking, max_nodes=10_000)
    assert not failed, f"seed {seed}: FAILED_INIT for {sorted(failed)}"
    return world, tasks, parking, per_agent, plans


def _cell_of(steps, t):
    return position_at(steps, t)


# ----------------------------------------------------------------------
# freeze / pending-task bookkeeping
# ----------------------------------------------------------------------
def test_freeze_keeps_the_executed_prefix():
    steps = [PlanStep(0, (0, 0)), PlanStep(2, (0, 1)), PlanStep(5, (0, 2))]
    frozen = freeze(steps, 3)
    assert [s.t for s in frozen] == [0, 2, 3]      # step at t=3 synthesised
    assert frozen[-1].cell == (0, 1)               # agent is still on (0, 1)
    assert freeze(steps, 5) == steps
    assert freeze(steps, 0) == [PlanStep(0, (0, 0))]
    assert freeze(None, 4) == []


def test_pending_tasks_tracks_a_carried_item():
    steps = [PlanStep(0, (0, 0)),
             PlanStep(1, (0, 1), PICK, item=0),
             PlanStep(2, (0, 1)),
             PlanStep(3, (0, 2), DROP, item=0),
             PlanStep(4, (0, 2))]
    tasks = [Task(0, (0, 1), (0, 2)), Task(1, (1, 1), (2, 2))]

    todo, holding = pending_tasks(steps, tasks, 2)
    assert holding is not None and holding.id == 0   # picked, not delivered
    assert [t.id for t in todo] == [1]

    todo, holding = pending_tasks(steps, tasks, 4)
    assert holding is None
    assert [t.id for t in todo] == [1]


def test_metrics_classify_delayed_vs_altered():
    base = [PlanStep(0, (0, 0)), PlanStep(1, (0, 1), PICK, item=0),
            PlanStep(2, (0, 1)), PlanStep(3, (0, 2), DROP, item=0)]
    # same cells & actions, every remaining tick shifted: a pure delay
    delayed = base[:1] + [PlanStep(s.t + 2, s.cell, s.action, s.item)
                          for s in base[1:]]
    assert classify(base, delayed, 0) == DELAYED_ONLY
    assert route_signature(base, 0) == route_signature(delayed, 0)
    assert completion_delta(base, delayed) == 2

    # same action sequence, different cells: the route itself changed
    rerouted = [PlanStep(0, (0, 0)), PlanStep(1, (1, 0), PICK, item=0),
                PlanStep(2, (1, 0)), PlanStep(3, (1, 1), DROP, item=0)]
    assert classify(base, rerouted, 0) == ALTERED_PATH
    assert route_signature(base, 0) != route_signature(rerouted, 0)
    assert classify(base, base, 0) == UNCHANGED
    assert completion_delta(base, rerouted) == 0


def test_label_rungs():
    base = [PlanStep(0, (1, 1)), PlanStep(1, (1, 2)), PlanStep(2, (1, 3)),
            PlanStep(3, (2, 3), DROP, item=0), PlanStep(4, (2, 3))]
    # R0: only extra waits on the same route
    same = base[:2] + [PlanStep(2, (1, 2))] + \
        [PlanStep(s.t + 1, s.cell, s.action, s.item) for s in base[2:]]
    assert _label(base, same, 0, 8) == "R0"
    # R1: detour that costs no extra completion time
    detour = [PlanStep(0, (1, 1)), PlanStep(1, (2, 1)), PlanStep(2, (2, 2)),
              PlanStep(3, (2, 3), DROP, item=0), PlanStep(4, (2, 3))]
    assert completion_delta(base, detour) == 0
    assert _label(base, detour, 0, 8) == "R1"
    # R2: detour that pushes completion beyond the threshold
    late = detour[:3] + [PlanStep(30, (2, 3), DROP, item=0),
                         PlanStep(31, (2, 3))]
    assert _label(base, late, 0, 8) == "R2"
    assert _label(None, base, 0, 8) == "R2"       # unknown previous plan


# ----------------------------------------------------------------------
# the R0-R4 ladder, on a controlled world
# ----------------------------------------------------------------------
def _open_world(h: int = 40, w: int = 40) -> World:
    return World(np.zeros((h, w), dtype=np.int8))


def _ladder_fixture():
    """A world where the *previous* order and the *greedy* order differ.

    The agent starts and parks at ``(0, 0)``;  with a leg budget of 30 ticks
    the pre-existing ``[B, A]`` order cannot finish (its return leg needs 31)
    while the greedy ``[A, B]`` order fits, so only a re-derived order repairs
    the plan (rung R3).
    """
    world = _open_world()
    task_a = Task(0, (1, 0), (1, 30))     # near pickup, far delivery
    task_b = Task(1, (0, 15), (0, 16))
    prev = plan_agent(world, ReservationTable(), 0, (0, 0), (0, 0),
                      [task_b, task_a], 0, 600)
    assert prev is not None
    return world, task_a, task_b, prev


def _picks(steps):
    return [s.item for s in steps if s.action == PICK]


def test_ladder_R0_when_the_previous_order_still_works():
    world, task_a, task_b, prev = _ladder_fixture()
    tail, level = reground_agent(world, ReservationTable(), 0, (0, 0), 0, (0, 0),
                                 [task_b, task_a], prev, T_limit=600)
    assert level == "R0"
    assert _picks(tail) == [task_b.id, task_a.id]
    assert classify(prev, tail, 0) == UNCHANGED


def test_ladder_R3_reorders_when_the_previous_order_fails():
    world, task_a, task_b, prev = _ladder_fixture()
    tail, level = reground_agent(world, ReservationTable(), 0, (0, 0), 0, (0, 0),
                                 [task_a, task_b], prev, T_limit=30)
    assert level == "R3"
    assert _picks(tail) == [task_a.id, task_b.id]     # nearest pickup first


def test_ladder_R4_when_no_order_can_reach_the_goals():
    world, task_a, task_b, prev = _ladder_fixture()
    res = ReservationTable()
    res.add_path(9, [(task_a.pickup, 0)], park=True)  # blocked forever
    tail, level = reground_agent(world, res, 0, (0, 0), 0, (0, 0),
                                 [task_a, task_b], prev, T_limit=30)
    assert tail is None and level == "R4"


def test_handoff_task_gives_the_orphan_to_the_cheapest_neighbour():
    world = _open_world()
    orphan = Task(7, (10, 10), (10, 12))
    plans = {0: [PlanStep(0, (10, 0))],
             1: [PlanStep(0, (0, 20))],
             2: [PlanStep(0, (5, 5))]}
    parking = [(10, 0), (0, 20), (5, 5)]
    res = ReservationTable()
    got = handoff_task(world, res, plans, 2, orphan, 0, parking,
                       {0: [], 1: [], 2: []}, comm_radius=30)
    assert got is not None
    rid, tail = got
    assert rid == 0                                   # (10, 0) is the cheapest
    assert _picks(tail) == [orphan.id]
    assert any(s.action == DROP and s.item == orphan.id for s in tail)
    assert not res.vertex          # the caller owns reservation bookkeeping


# ----------------------------------------------------------------------
# end-to-end scenarios: the four replanning modes
# ----------------------------------------------------------------------
REACTIVE_MODES = ("negotiate", "self_only", "bfs")


def test_repair_never_uses_the_global_planner(monkeypatch):
    """The whole point of the local repair engine (SPEC 9)."""

    def _boom(*args, **kwargs):
        raise AssertionError("repair must never call the global planner")

    monkeypatch.setattr(planner, "prioritized_ground", _boom)
    baselines.reset_counters()

    for mode in REACTIVE_MODES:
        world, tasks, parking, per_agent, plans = _scenario(seed=1)
        res, records, _ = simulate_scenario(
            world, dict(plans), per_agent, parking, mode=mode,
            accidents=[AccidentEvent(0, 10, 6)],
            emergencies=[EmergencyEvent(1, 14, 4)],
            blockages=[BlockageEvent(parking[0], 30, 5)],
            comm_radius=12)
        assert records, f"{mode}: no repair record was produced"
        assert res.conflict_free, (mode, res.violations)
        assert res.completed == res.total_tasks == len(tasks)

    assert baselines.GLOBAL_PLAN_CALLS == 0


def test_repair_records_describe_the_replan():
    world, tasks, parking, per_agent, plans = _scenario(seed=3)
    res, records, _ = simulate_scenario(world, dict(plans), per_agent, parking,
                                        mode="negotiate",
                                        accidents=[AccidentEvent(0, 12, 5)])
    assert res.conflict_free
    assert res.completed == len(tasks)
    assert len(records) == 1
    rec = records[0]
    assert rec.event_type == "accident"
    assert rec.t == 12
    assert rec.A0_size == 1
    assert rec.level_resolved in LEVELS
    assert 0 in rec.altered_plan_ids
    assert rec.st_astar_calls > 0
    assert rec.cpu_ms >= 0.0
    assert rec.success


def test_self_only_touches_no_more_agents_than_negotiate():
    world, tasks, parking, per_agent, plans = _scenario(seed=0)
    res_n, rec_n, _ = simulate_scenario(world, dict(plans), per_agent, parking,
                                        mode="negotiate",
                                        accidents=[AccidentEvent(0, 12, 5)])
    res_s, rec_s, _ = simulate_scenario(world, dict(plans), per_agent, parking,
                                        mode="self_only",
                                        accidents=[AccidentEvent(0, 12, 5)])
    for res in (res_n, res_s):
        assert res.conflict_free and res.completed == len(tasks)

    self_only_agents = set(rec_s[0].altered_plan_ids)
    assert self_only_agents <= {0}                     # only A0 is replanned
    assert set(rec_n[0].altered_plan_ids) >= {0}
    assert len(rec_n[0].altered_plan_ids) >= len(rec_s[0].altered_plan_ids)


def test_dynamic_blockage_is_respected_by_reactive_modes_only():
    world, tasks, parking, per_agent, plans = _scenario(seed=0)
    # a cell agent 0 drives through, blocked *just before* it gets there
    hit = next(s for s in plans[0] if s.t >= 8)
    events = [BlockageEvent(hit.cell, hit.t - 2, 6)]

    for mode in REACTIVE_MODES:
        w2, _, p2, _, _ = _scenario(seed=0)
        res, _, _ = simulate_scenario(w2, dict(plans), {a: list(v) for a, v in
                                                        enumerate(per_agent)},
                                      p2, mode=mode, blockages=events)
        assert res.conflict_free, (mode, res.violations)
        assert res.completed == res.total_tasks

    # the static ablation executes the original plan verbatim -> it drives
    # straight through the blockage
    w3, _, p3, _, _ = _scenario(seed=0)
    res_static, recs, _ = simulate_scenario(w3, dict(plans), per_agent, p3,
                                            mode="static", blockages=events)
    assert not recs
    assert not res_static.conflict_free


def test_blockage_is_postponed_when_its_cell_is_occupied():
    world, tasks, parking, per_agent, plans = _scenario(seed=0)
    busy = next(s for s in plans[0] if s.t >= 6)      # agent 0 sits there
    shifted = postpone_blockages(plans, [BlockageEvent(busy.cell, busy.t, 6)])
    assert len(shifted) == 1
    assert shifted[0].cell == busy.cell
    assert shifted[0].t0 > busy.t                     # announced after the visit
    # once a cell is occupied for good (an agent parked on it) the blockage can
    # never be applied -> the event is dropped
    parked = parking[0]
    after = plans[0][-1].t + 1
    assert postpone_blockages(plans, [BlockageEvent(parked, after, 6)]) == []


def test_postpone_blockages_is_a_noop_without_events():
    """An event list that is empty (or trivially so) must not raise."""
    assert postpone_blockages(plans={0: []}, events=[]) == []
    ev = BlockageEvent((1, 1), 3, 4)
    assert postpone_blockages({}, [ev]) == [ev]


def test_st_astar_never_accepts_a_goal_tick_inside_a_blockage():
    """A wait-in-place goal must respect the blockage too (SPEC 10)."""
    world = _open_world(10, 10)
    world.add_blockage((5, 5), 0, 3)
    path, _ = st_astar(world, ReservationTable(), 0, (5, 5), (5, 5), 0,
                       hold_after=1)
    assert path is not None
    assert path[-1][1] > 3                    # it waits the blockage out


def test_permanent_breakdown_hands_everything_over():
    world, tasks, parking, per_agent, plans = _scenario(seed=4)
    broken = 0
    res, records, final = simulate_scenario(
        world, dict(plans), per_agent, parking, mode="negotiate",
        accidents=[AccidentEvent(broken, 8, 400, permanent=True)],
        comm_radius=40)

    assert records and records[0].level_resolved == "R4"
    assert res.conflict_free, res.violations
    assert res.completed == res.total_tasks == len(tasks)
    # the broken agent only waits from the breakdown onwards ...
    rest = [s for s in final[broken] if s.t > 8]
    assert rest and all(s.action == WAIT for s in rest)


def test_global_mode_is_the_only_one_that_calls_the_global_planner():
    baselines.reset_counters()
    world, tasks, parking, per_agent, plans = _scenario(seed=2)
    res, records, _ = simulate_scenario(world, dict(plans), per_agent, parking,
                                        mode="global",
                                        accidents=[AccidentEvent(0, 10, 6)])
    assert baselines.GLOBAL_PLAN_CALLS >= 1
    assert baselines.GLOBAL_PLAN_CALLS == len(records)
    assert res.conflict_free
    assert res.completed == res.total_tasks

    base = {0: [PlanStep(0, (0, 0)), PlanStep(1, (0, 1), DROP, item=0)],
            1: [PlanStep(0, (2, 2)), PlanStep(1, (2, 3), DROP, item=1)]}
    same = {a: list(s) for a, s in base.items()}
    diff = {0: [PlanStep(0, (0, 0)), PlanStep(3, (0, 1), DROP, item=0)],
            1: list(base[1])}
    sets = altered_agent_sets(base, diff, 0, disrupted=[0])
    assert 0 in sets["altered_plan"] and 1 not in sets["altered_plan"]
    sets = altered_agent_sets(base, same, 0, disrupted=[1])
    assert sets["altered_plan"] == [1]        # disrupted agent is force-added


# ----------------------------------------------------------------------
# causality: a pass only sees the disruptions of its own tick
# ----------------------------------------------------------------------
def test_a_future_accident_does_not_freeze_agents_early():
    """``SPEC.md`` 10 replay is causal: an accident announced for t=30 must not
    hold its agent during the repair of a t=3 blockage.

    Regression: the repair passes used a flat ``hold_until`` table, so *every*
    pass froze the accident agent (agent 0) from its own tick onward.
    """
    world, tasks, parking, per_agent, plans = _scenario(seed=0)
    assert len({_cell_of(plans[0], t) for t in range(30)}) > 1   # it does move
    hit = next(s for s in plans[1] if s.t >= 4)   # a cell agent 1 crosses
    blockage = [BlockageEvent(hit.cell, hit.t, 6)]

    def run(accidents):
        w2, _, p2, _, _ = _scenario(seed=0)
        res, _, final = simulate_scenario(
            w2, dict(plans), per_agent, p2, mode="negotiate",
            blockages=blockage, accidents=accidents,
            comm_radius=1000)                 # everybody is affected
        assert res.conflict_free, res.violations
        assert res.completed == res.total_tasks
        return final

    plain, late = run([]), run([AccidentEvent(0, 30, 5)])
    for t in range(30):                       # untouched before the accident
        assert _cell_of(late[0], t) == _cell_of(plain[0], t), t
    held = _cell_of(late[0], 30)
    assert all(_cell_of(late[0], t) == held for t in range(30, 35))


# ----------------------------------------------------------------------
# a held agent guards its cell for the whole hold
# ----------------------------------------------------------------------
def test_a_held_agent_cell_is_reserved_for_everyone_else():
    """The pinned cell of a held agent must be in the reservation table.

    ``_plan_from`` cannot refuse the resume step at the release tick (the agent
    is physically there), so without the pin a neighbour simply walks through
    the broken-down agent -- here through the only corridor cell ``(3, 3)``.
    The held agent has the *higher* id, so it is replanned after the neighbour:
    only a pre-seeded reservation can keep the neighbour off that cell.
    """
    def _corridor() -> World:
        grid = np.zeros((8, 8), dtype=np.int8)
        grid[[r for r in range(8) if r != 3], 3] = 1     # only (3, 3) connects
        return World(grid)

    world = _corridor()
    parking = [(0, 0), (7, 7)]
    task0, task1 = Task(0, (0, 2), (6, 5)), Task(1, (6, 6), (6, 7))
    plans = {0: plan_agent(world, ReservationTable(), 0, (0, 2), parking[0],
                           [task0], 0, T_limit=600),
             1: [PlanStep(0, (3, 3))]}     # broken down, held on the corridor
    assert plans[0] is not None
    tasks = {0: [task0], 1: [task1]}
    # The held agent has the higher id, so it is replanned *after* the
    # neighbour (this is a residual-conflict pass: both agents are in A0).
    events = lambda end: [Disruption("accident", 1, 0, end),
                          Disruption("conflict", 0, 0)]

    def crossings(hold_end: int):
        out, _ = repair_schedule(world, {a: list(s) for a, s in plans.items()},
                                 tasks, parking, t_now=0, comm_radius=100,
                                 mode="negotiate",
                                 disruptions=events(hold_end),
                                 hold_until={1: hold_end})
        return [s.t for s in (out[0] or []) if s.cell == (3, 3)]

    assert min(crossings(0)) < 6                 # no hold: through it at once
    assert all(t > 12 for t in crossings(12))    # held: it waits the hold out


def test_a_held_agent_owns_its_cell_for_the_whole_hold():
    """Nobody else may drive onto the cell of a broken-down agent.

    Regression: a low-id neighbour used to be routed straight through the held
    cell at the release tick, because the held agent's pinned cell was not in
    the seeded reservation table.
    """
    at_t, dur = 8, 7
    for mode in REACTIVE_MODES:
        world, tasks, parking, per_agent, plans = _scenario(seed=0)
        res, _, final = simulate_scenario(
            world, dict(plans), per_agent, parking, mode=mode,
            accidents=[AccidentEvent(0, at_t, dur)], comm_radius=1000)
        assert res.conflict_free, (mode, res.violations)
        assert res.completed == res.total_tasks
        cell = _cell_of(final[0], at_t)
        span = range(at_t, at_t + dur)
        assert all(_cell_of(final[0], t) == cell for t in span)
        for aid in range(len(parking)):
            if aid == 0:
                continue
            for t in range(at_t, at_t + dur):
                assert _cell_of(final[aid], t) != cell, (mode, aid, t)
