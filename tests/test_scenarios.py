"""The choke-point scenarios (``SPEC.md`` 9.2).

:mod:`scenarios` exists so the negotiation rungs can be exercised at all: the
random warehouse of :mod:`world` is open enough that a repair stays a *silent*
local detour and the protocol never speaks.  These tests pin the properties the
demo and the report rely on -- a single-door corridor, a repair that actually
negotiates (messages at rung ``R2``) instead of aborting (``R4``), and the
congested, disrupted warehouse shift whose five activity counters are all
non-zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import Config
from negotiation import ACCEPT, COMMIT, PROPOSE_REROUTE, Negotiation
from scenarios import (aisle_closure_grid, blockages_on_routes, choke_showcase,
                       corridor_world, parked_holder_scenario,
                       parked_holder_showcase, warehouse_profile,
                       warehouse_shift)
from sim import simulate_scenario
from world import FREE, SHELF, World

#: the shipped demo instance: a 16x16 warehouse with 12 robots, one closed
#: aisle, six closed lanes, two robots down at the same tick, one diversion --
#: and the seed the demo defaults to, whose every repair mode stays fast.
DEMO = dict(hw=16, n_agents=12, tasks_per_agent=2, seed=1, blockages=6,
            breakdowns=2, emergencies=1)

#: the five SPEC 11 activity counters, as :func:`viz.build_frames` reports them
COUNTERS = ("altered_so_far", "messages_so_far", "disruptions_so_far",
            "blocked_peak", "waiting_peak")


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


# ----------------------------------------------------------------------
# the congested, disrupted warehouse of the demo
# ----------------------------------------------------------------------
def test_aisle_closure_leaves_exactly_one_door():
    grid, door = aisle_closure_grid(16, 16, np.random.default_rng(0))
    r, c = door
    assert grid[r, c] == FREE
    assert all(grid[rr, c] == SHELF for rr in range(16) if rr != r)
    world = World(grid)
    assert world.connected(), "the single door must keep the map in one piece"
    # the two halves are two steps apart *only* through the door
    assert world.distance((r, c - 1), (r, c + 1)) == 2


def _shift(hw: int = 10, n_agents: int = 4, seed: int = 0):
    cfg = Config(H=hw, W=hw, n_agents=n_agents, tasks_per_agent=2, seed=seed)
    shift = warehouse_shift(cfg, np.random.default_rng(cfg.seed),
                            n_blockages=3, n_breakdowns=2, n_emergencies=1)
    assert not shift["failed"], f"unplannable agents {sorted(shift['failed'])}"
    return cfg, shift


def test_warehouse_shift_is_the_closed_aisle_instance():
    cfg, shift = _shift()
    grid, door = shift["grid"], shift["door"]
    assert grid is not None and door is not None
    assert all(grid[rr, door[1]] == SHELF
               for rr in range(cfg.H) if rr != door[0])
    assert shift["world"].is_free(door)
    owned = sorted(tk.id for lst in shift["per_agent"] for tk in lst)
    assert owned == sorted(tk.id for tk in shift["tasks"])
    assert len(shift["per_agent"]) == cfg.n_agents
    assert shift["parking"] and len(shift["parking"]) == cfg.n_agents


def test_warehouse_shift_is_deterministic_and_reproducible():
    a = warehouse_shift(Config(H=10, W=10, n_agents=4, tasks_per_agent=2,
                               seed=1),
                        np.random.default_rng(1), n_blockages=3,
                        n_breakdowns=2, n_emergencies=1)
    b = warehouse_shift(Config(H=10, W=10, n_agents=4, tasks_per_agent=2,
                               seed=1),
                        np.random.default_rng(1), n_blockages=3,
                        n_breakdowns=2, n_emergencies=1)
    assert a["parking"] == b["parking"] and a["door"] == b["door"]
    assert [e.cell for e in a["profile"]["blockages"]] == \
           [e.cell for e in b["profile"]["blockages"]]
    assert [e.agent for e in a["profile"]["accidents"]] == \
           [e.agent for e in b["profile"]["accidents"]]


def test_warehouse_profile_layers_blockages_breakdowns_and_diversions():
    cfg, shift = _shift()
    prof = shift["profile"]
    assert prof["blockages"] and len(prof["blockages"]) == 3
    assert len(prof["accidents"]) == 2 and len(prof["emergencies"]) == 1
    assert all(e.cell != shift["door"] for e in prof["blockages"]), \
        "the single door is never sealed: no repair could undo that"
    assert all(e.t1 > e.t0 for e in prof["blockages"])
    # the breakdowns share one tick: a charging-rail trip stops them together
    assert len({e.at_t for e in prof["accidents"]}) == 1


def test_blockages_on_routes_really_lie_on_a_route():
    cfg, shift = _shift()
    plans = shift["plans"]
    blk = blockages_on_routes(plans, 3, np.random.default_rng(4), avoid=[])
    assert len(blk) == 3
    for ev in blk:
        assert any(s.cell == ev.cell and ev.t0 <= s.t < ev.t1
                   for steps in plans.values() for s in (steps or []))
    assert not blockages_on_routes(plans, 0, np.random.default_rng(0))


def test_the_demo_shift_fills_all_five_counters():
    """The shipped instance must never fall back to five honest zeros.

    Parametrised over :data:`run_demo.VIZ_MODES`: the one GIF set the demo
    writes is exactly the modes whose panel has to be busy.
    """
    import viz
    from run_demo import VIZ_MODES

    assert VIZ_MODES, "the demo must animate at least one mode"
    for mode in VIZ_MODES:
        cfg = Config(H=DEMO["hw"], W=DEMO["hw"], n_agents=DEMO["n_agents"],
                     tasks_per_agent=DEMO["tasks_per_agent"], seed=DEMO["seed"])
        shift = warehouse_shift(cfg, np.random.default_rng(cfg.seed),
                                n_blockages=DEMO["blockages"],
                                n_breakdowns=DEMO["breakdowns"],
                                n_emergencies=DEMO["emergencies"])
        assert not shift["failed"], f"unplannable {sorted(shift['failed'])}"
        log = Negotiation(t=0)
        world = World(shift["grid"].copy())    # the events mutate this one
        sim, records, final = simulate_scenario(
            world,
            {a: (list(s) if s else None) for a, s in shift["plans"].items()},
            shift["per_agent"], shift["parking"], mode=mode,
            blockages=list(shift["profile"]["blockages"]),
            accidents=list(shift["profile"]["accidents"]),
            emergencies=list(shift["profile"]["emergencies"]),
            total_tasks=cfg.n_tasks, comm_radius=cfg.comm_radius,
            delay_threshold=cfg.delay_threshold, lambda_soft=cfg.lambda_soft,
            max_depth=cfg.max_depth, beta_alter=cfg.beta_alter, message_log=log,
            record_traces=True)
        assert sim.conflict_free, (mode, sim.violations)
        frames = viz.build_frames(
            world, final, shift["tasks"], shift["parking"],
            prev=shift["plans"], t_freeze=records[0].t if records else 0,
            disrupted=sorted({a for r in records for a in r.altered_plan_ids}),
            events=list(shift["profile"]["blockages"])
            + list(shift["profile"]["accidents"])
            + list(shift["profile"]["emergencies"]),
            records=records, log=log, traces=sim.traces)
        worst = {k: max(f["counters"][k] for f in frames) for k in COUNTERS}
        assert all(v > 0 for v in worst.values()), (mode, worst)
        assert worst["disruptions_so_far"] == 9, (mode, worst)   # 6 + 2 + 1
        assert worst["altered_so_far"] >= DEMO["n_agents"] - 1, (mode, worst)
        assert worst["messages_so_far"] > 0, (mode, worst)
        assert worst["blocked_peak"] > 0 and worst["waiting_peak"] > 0, \
            (mode, worst)


