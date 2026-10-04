"""The choke-point scenarios (``SPEC.md`` 9.2).

:mod:`scenarios` exists so the negotiation rungs can be exercised at all: the
random warehouse of :mod:`world` is open enough that a repair stays a *silent*
local detour and the protocol never speaks.  These tests pin the two properties
the demo and the report rely on -- a single-door corridor, and a repair that
actually negotiates (messages at rung ``R2``) instead of aborting (``R4``).
"""
from __future__ import annotations

import pytest

from negotiation import ACCEPT, COMMIT, PROPOSE_REROUTE, Negotiation
from scenarios import (choke_showcase, corridor_world, parked_holder_scenario,
                       parked_holder_showcase)
from sim import simulate_scenario
from world import World


def _run(sc, mode: str = "negotiate"):
    """Repair a scenario dict on a fresh world (events mutate it in place)."""
    log = Negotiation(t=0)
    sim, records, final = simulate_scenario(
        World(sc["world"].grid.copy()),
        {a: list(s) if s else None for a, s in sc["plans"].items()},
        sc["per_agent"], sc["parking"], mode=mode,
        blockages=list(sc["blockages"]), accidents=list(sc["accidents"]),
        emergencies=list(sc["emergencies"]), total_tasks=sc["total_tasks"],
        comm_radius=6, delay_threshold=8, lambda_soft=15, max_depth=3,
        beta_alter=10, message_log=log, record_traces=True)
    return sim, records, final, log


def test_corridor_world_has_exactly_one_door():
    world = corridor_world(8, 8, door_col=3, door_row=3)
    free = [(r, 3) for r in range(8) if world.is_free((r, 3))]
    assert free == [(3, 3)], "the wall must leave a single crossing"
    assert all(world.is_shelf((r, 3)) for r in range(8) if r != 3)
    assert world.is_free((3, 3))
    # the default door row is the middle of the wall
    assert corridor_world(9, 9, door_col=4).is_free((4, 4))


def test_parked_holder_scenario_shape():
    world, plans, tasks, parking = parked_holder_scenario()
    assert parking == [(3, 3), (0, 0)], "agent 0 parks on the choke point"
    assert tasks[0] and not tasks[1], "only agent 0 owns a task"
    assert plans[0] and plans[1], "both agents must be plannable"
    assert world.is_free(parking[0])


def test_parked_holder_scenario_negotiates():
    sc = parked_holder_showcase()
    assert sc["accidents"], "the pair needs an outage to be un-repairable"
    sim, records, final, log = _run(sc, "negotiate")
    assert sim.conflict_free and sim.completed == sim.total_tasks
    assert [r.level_resolved for r in records] == ["R2"]
    assert {m.kind for m in log.messages} == {PROPOSE_REROUTE, ACCEPT, COMMIT}
    counts = records[0].messages_by_type
    assert counts[PROPOSE_REROUTE] == 1 and counts[ACCEPT] == 1
    assert counts[COMMIT] == 1


def test_parked_holder_pair_speeds_up_with_negotiation():
    sc = parked_holder_showcase()
    sim, _records, _final, _log = _run(sc, "negotiate")
    slow, records2, _f, log2 = _run(sc, "self_only")
    assert log2.messages == [], "self_only never negotiates"
    assert any(r.level_resolved.startswith("R4") for r in records2)
    assert sim.makespan < slow.makespan, "the grant must beat waiting it out"


def test_choke_showcase_negotiates_and_self_only_aborts():
    sc = choke_showcase(4)
    assert sc["n_agents"] == 4 and sc["total_tasks"] == 3
    sim, records, final, log = _run(sc, "negotiate")
    assert sim.conflict_free and sim.completed == sim.total_tasks
    assert [r.level_resolved for r in records] == ["R2"]
    assert len(log.messages) == 3, "PROPOSE_REROUTE -> ACCEPT -> COMMIT"
    assert set(log.by_type()) >= {PROPOSE_REROUTE, ACCEPT, COMMIT}

    sim2, records2, _, log2 = _run(sc, "self_only")
    assert sim2.conflict_free and sim2.completed == sim2.total_tasks
    assert log2.messages == [], "without negotiation there is no dialogue"
    assert any(r.level_resolved.startswith("R4") for r in records2)


@pytest.mark.parametrize("n_agents", [2, 3, 5, 6])
def test_choke_showcase_scales_cleanly(n_agents):
    sc = choke_showcase(n_agents)
    sim, records, final, log = _run(sc, "negotiate")
    assert sim.conflict_free, f"{n_agents} agents must stay conflict-free"
    assert sim.completed == sim.total_tasks == n_agents - 1
    assert len(log.messages) == 3


def test_choke_showcase_needs_at_least_two_agents():
    with pytest.raises(ValueError):
        choke_showcase(1)


def test_choke_showcase_is_deterministic():
    a, b = choke_showcase(5, seed=7), choke_showcase(5, seed=7)
    assert a["parking"] == b["parking"] and a["starts"] == b["starts"]
