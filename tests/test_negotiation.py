"""Negotiation as POP operations (``SPEC.md`` section 9.2).

The protocol is tested on its own (verdicts, messages, radius widening, the
cost model) and end to end through ``repair_schedule``/``simulate_scenario``:
when the ladder cannot find a candidate at all, a granted negotiation still
lets the initiator through, and every message lands in
``RepairRecord.messages_by_type``.
"""
from __future__ import annotations

from disruptions import AccidentEvent
from negotiation import (ABORT, ACCEPT, COMMIT, MESSAGE_TYPES, PROPOSE_ORDER,
                         PROPOSE_REROUTE, REJECT, Negotiation, first_clash,
                         grant_is_worthwhile, paths_clash, resolve_clash)
from modify import Disruption, repair_schedule
from planner import PlanStep, plan_agent, plan_to_path
from reservation import ReservationTable
from scenarios import corridor_world, parked_holder_scenario
from sim import simulate_scenario


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _log(t: int = 0) -> Negotiation:
    return Negotiation(t=t)


def _sent(log: Negotiation) -> dict:
    return {k: v for k, v in log.by_type().items() if v}


def _accept(*, distance: int = 1, comm_radius: int = 6, replan=(0, 1),
            order=(0, 1), holder: int = 0, initiator: int = 1, **kw):
    """``resolve_clash`` with the usual happy-path arguments."""
    return resolve_clash(
        holder, initiator, distance=distance, replan_set=set(replan),
        order_index={a: i for i, a in enumerate(order)},
        comm_radius=comm_radius, **kw)


def _corridor():
    """An 8x8 world whose only crossing between the halves is ``(3, 3)``."""
    return corridor_world(8, 8, door_col=3, door_row=3)


def _parked_holder_scenario():
    """Agent 0 parks on the choke point; agent 1 is locked out of it.

    Agent 1 can only reach its parking cell through ``(3, 3)``, which agent 0
    books forever.  Both agents are disabled at t=2 (0 until t=6, 1 until
    t=10), so agent 1 needs a *negotiated* grant: there is no hard candidate.
    Shared with ``run_demo.py --choke`` via :mod:`scenarios`.
    """
    return parked_holder_scenario()


# ----------------------------------------------------------------------
# the message log
# ----------------------------------------------------------------------
def test_the_log_counts_every_message_type():
    log = _log(t=7)
    log.propose_order(1, 0, "before")
    log.accept(0, 1, "delay=1")
    log.reject(2, 1, "frozen")
    log.commit(1, [0, 2], "go")
    log.abort(1, [2], "rejected")

    counts = log.by_type()
    assert set(counts) == set(MESSAGE_TYPES)          # every column exists
    assert counts[PROPOSE_ORDER] == 1
    assert counts[ACCEPT] == 1
    assert counts[REJECT] == 1
    assert counts[COMMIT] == 2                        # one per holder
    assert counts[ABORT] == 1
    assert counts[PROPOSE_REROUTE] == 0
    assert log.rounds == 1 and log.committed == 1 and log.aborted == 1
    assert log.t == 7 and len(log) == 6
    assert (1, 0) in log.arrows() and (0, 1) in log.arrows()
    assert log.details(REJECT) == ["frozen"]


def test_an_empty_log_is_falsy_but_still_counts_zero():
    """``__len__`` must not be mistaken for "no log" by the protocol."""
    log = _log()
    assert not log
    assert log.by_type() == {k: 0 for k in MESSAGE_TYPES}


# ----------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------
def test_a_later_replan_set_holder_accepts_by_waiting():
    log = _log()
    # the holder is re-derived *after* the initiator in this pass
    v = _accept(order=(1, 0), holder=0, initiator=1, log=log)
    assert v.granted and v.how == "ordering"
    assert _sent(log) == {PROPOSE_ORDER: 1, ACCEPT: 1}


def test_a_frozen_holder_rejects():
    log = _log()
    v = _accept(replan=(1,), holder=0, initiator=1, log=log)
    assert not v.granted and v.how == "frozen"
    assert _sent(log) == {PROPOSE_ORDER: 1, REJECT: 1}


def test_an_ordering_cycle_is_rejected():
    log = _log()
    v = _accept(chain=(0,), holder=0, initiator=1, log=log)
    assert not v.granted and v.how == "cycle"
    assert REJECT in _sent(log)


def test_a_held_holder_rejects_only_inside_its_hold():
    """An agent that is held *now* can still yield after the hold ends."""
    kw = dict(hold_end={0: 12}, reroute_cost=lambda h: 2, log=_log())
    early = _accept(contested=5, **kw)
    assert not early.granted and early.how == "held"
    late = _accept(contested=14, **kw)
    assert late.granted and late.how == "reroute"


def test_a_fixed_holder_never_yields():
    v = _accept(fixed={0}, reroute_cost=lambda h: 0)
    assert not v.granted and v.how == "held"


def test_the_reroute_branch_prices_the_holder():
    """SPEC 9.2 branch 2: j reroutes only if its own delay stays small."""
    assert _accept(reroute_cost=lambda h: 2).how == "reroute"
    refused = _accept(reroute_cost=lambda h: 99)
    assert not refused.granted and refused.how == "delay"
    stuck = _accept(reroute_cost=lambda h: None)
    assert not stuck.granted and stuck.how == "blocked"


def test_rung_r3_pulls_a_far_holder_into_the_replan_set():
    """``comm_radius += 3`` (at most ``max_depth`` rounds) -- SPEC 9.1."""
    log = _log()
    far = _accept(distance=8, comm_radius=6, replan=(1,), log=log)
    assert far.granted and far.how == "reroute" and far.extra == {0}
    assert _sent(log) == {PROPOSE_REROUTE: 1}
    # ... but only while a rung is left, and only within reach
    assert not _accept(distance=8, comm_radius=6, replan=(1,),
                       depth=3, max_depth=3).granted
    assert not _accept(distance=40, comm_radius=6, replan=(1,)).granted


# ----------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------
def test_first_clash_ignores_the_granted_holders():
    res = ReservationTable()
    res.add_path(0, [((1, 1), 0), ((1, 1), 1), ((1, 1), 2)], park=True)
    path = [PlanStep(0, (0, 1)), PlanStep(1, (1, 1)), PlanStep(2, (1, 1))]
    assert first_clash(res, 1, path) == ((1, 1), 1, 0)
    assert first_clash(res, 1, path, ignore={0}) is None
    assert first_clash(res, 1, [PlanStep(0, (0, 0))]) is None


def test_paths_clash_finds_vertex_and_swap_clashes():
    a = [PlanStep(0, (0, 0)), PlanStep(1, (1, 0)), PlanStep(2, (2, 0))]
    same = [PlanStep(1, (1, 0)), PlanStep(2, (2, 0))]
    assert paths_clash(a, same) == ((1, 0), 1)
    swap = [PlanStep(0, (1, 0)), PlanStep(1, (0, 0))]
    assert paths_clash(a, swap) == ((0, 0), 0)
    other = [PlanStep(0, (5, 5)), PlanStep(1, (6, 5))]
    assert paths_clash(a, other) is None


def test_a_grant_must_pay_for_the_agents_it_alters():
    """SPEC 9.1 ``cost = added_delay + beta_alter*altered``."""
    # saves 20 ticks but alters one extra agent: 20 > 10 and > 8
    assert grant_is_worthwhile(hard_completion=40, prop_completion=20,
                               n_new_altered=1, beta_alter=10,
                               delay_threshold=8)
    # the same saving with three extra altered agents is not worth it
    assert not grant_is_worthwhile(hard_completion=40, prop_completion=20,
                                   n_new_altered=3, beta_alter=10,
                                   delay_threshold=8)
    # under the delay threshold the ladder's wait is cheaper (SPEC 9.4)
    assert not grant_is_worthwhile(hard_completion=10, prop_completion=5,
                                   n_new_altered=0, beta_alter=10,
                                   delay_threshold=8)
    assert not grant_is_worthwhile(hard_completion=10, prop_completion=11,
                                   n_new_altered=0, beta_alter=10,
                                   delay_threshold=8)


# ----------------------------------------------------------------------
# end to end: the rung that rescues a blocked agent
# ----------------------------------------------------------------------
def test_a_granted_negotiation_rescues_an_agent_the_ladder_cannot_plan():
    """The choke-point holder reroutes so the stuck agent can pass.

    Without the negotiation agent 1 has no candidate at all and is handed over
    (it just waits out ``failure_wait``); with it, both agents finish and the
    record shows the full ``PROPOSE_REROUTE``/``ACCEPT``/``COMMIT`` exchange.
    """
    world, plans, tasks, parking = _parked_holder_scenario()
    accidents = [AccidentEvent(0, 2, 4), AccidentEvent(1, 2, 8)]

    results = {}
    for mode in ("negotiate", "self_only"):
        sim, recs, _ = simulate_scenario(
            _corridor(), {a: list(s) for a, s in plans.items()}, tasks,
            parking, mode=mode, comm_radius=100, accidents=accidents,
            total_tasks=1)
        assert sim.conflict_free, sim.violations
        assert sim.completed == 1
        results[mode] = (sim, recs)

    sim, recs = results["negotiate"]
    assert [r.level_resolved for r in recs] == ["R2"]       # the R2 rung
    counts = recs[0].messages_by_type
    assert counts[PROPOSE_REROUTE] == 1 and counts[ACCEPT] == 1
    assert counts[COMMIT] == 1 and counts[ABORT] == 0
    assert sim.makespan < results["self_only"][0].makespan


def test_the_negotiation_is_repair_only_and_never_uses_the_global_planner(
        monkeypatch):
    import baselines
    import planner

    def boom(*a, **kw):                       # pragma: no cover - must not run
        raise AssertionError("repair called the global planner")

    monkeypatch.setattr(planner, "prioritized_ground", boom)
    before = baselines.GLOBAL_PLAN_CALLS
    world, plans, tasks, parking = _parked_holder_scenario()
    sim, recs, _ = simulate_scenario(
        _corridor(), plans, tasks, parking, mode="negotiate", comm_radius=100,
        accidents=[AccidentEvent(0, 2, 4), AccidentEvent(1, 2, 8)],
        total_tasks=1)
    assert sim.conflict_free and sim.completed == 1
    assert baselines.GLOBAL_PLAN_CALLS == before
    assert recs[0].st_astar_calls > 0


def test_a_plain_repair_reports_no_negotiation_messages():
    """No clash -> no messages: the counters must not invent a protocol."""
    world, plans, tasks, parking = _parked_holder_scenario()
    res = ReservationTable()
    res.add_path(0, plan_to_path(plans[0]), park=True)
    res.add_path(1, plan_to_path(plans[1]), park=True)
    log = _log(t=3)
    out, recs = repair_schedule(
        _corridor(), {a: list(s) for a, s in plans.items()}, tasks, parking,
        t_now=3, mode="negotiate", comm_radius=100,
        disruptions=[Disruption("conflict", 1, 3)], message_log=log)
    assert all(v == 0 for v in recs[0].messages_by_type.values())
    assert log.messages == []
    assert out[1] is not None


def test_a_rejected_proposal_is_recorded_but_changes_nothing():
    """``self_only`` never negotiates: nothing is logged, nothing moves."""
    world = _corridor()
    parking = [(0, 0), (0, 1)]
    tasks = {0: [], 1: []}
    res = ReservationTable()
    plans = {0: plan_agent(world, res, 0, (0, 0), parking[0], [], 0,
                           T_limit=600)}
    res.add_path(0, plan_to_path(plans[0]), park=True)
    plans[1] = plan_agent(world, res, 1, (0, 1), parking[1], [], 0,
                          T_limit=600)
    res.add_path(1, plan_to_path(plans[1]), park=True)

    log = _log(t=0)
    out, recs = repair_schedule(
        world, {a: list(s) for a, s in plans.items()}, tasks, parking,
        t_now=0, mode="self_only", comm_radius=100,
        disruptions=[Disruption("conflict", 0, 0)], message_log=log)
    # ``self_only`` never negotiates: nothing is logged, nothing moves
    assert log.messages == []
    assert out[0] == plans[0]


