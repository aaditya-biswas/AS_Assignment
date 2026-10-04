"""Tests for the PEG grounding layer and the simulator (M3)."""
from __future__ import annotations

import numpy as np
import pytest

from config import Config
from peg import ground_pop, peg_solve
from planner import DROP, MOVE, PICK, WAIT, PlanStep
from pocl import build_goals, build_init, is_valid_pop, pocl
from reservation import ReservationTable
from sim import build_traces, plan_makespan, run_sim
from world import SHELF, Task, World, assign_tasks, build_world


# ----------------------------------------------------------------------
# PEG grounding
# ----------------------------------------------------------------------
def test_ground_pop_expands_macros_to_atomic_steps():
    cfg = Config(H=16, W=16, n_agents=1, tasks_per_agent=2, seed=4)
    world, tasks, parking = build_world(cfg, np.random.default_rng(cfg.seed))
    pop = pocl(0, list(tasks), parking[0], world, max_nodes=10_000)
    assert pop is not None
    res = ReservationTable()
    steps = ground_pop(world, res, 0, parking[0], parking[0], pop)
    assert steps is not None
    # times are strictly increasing by exactly 1 (atomic ticks)
    for a, b in zip(steps, steps[1:]):
        assert b.t - a.t == 1
    # exactly one PICK and one DROP per task
    items_pick = [s.item for s in steps if s.action == PICK]
    items_drop = [s.item for s in steps if s.action == DROP]
    assert sorted(items_pick) == sorted(t.id for t in tasks)
    assert sorted(items_drop) == sorted(t.id for t in tasks)
    # every PICK precedes the matching DROP
    for item in items_pick:
        tp = next(s.t for s in steps if s.action == PICK and s.item == item)
        td = next(s.t for s in steps if s.action == DROP and s.item == item)
        assert tp < td
    # the agent ends parked on its parking cell
    assert steps[-1].cell == parking[0]


# ----------------------------------------------------------------------
# v2 end-to-end
# ----------------------------------------------------------------------
def test_v2_pipeline_is_conflict_free_over_seeds():
    for seed in range(10):
        cfg = Config(H=28, W=28, n_agents=12, tasks_per_agent=2, seed=seed)
        rng = np.random.default_rng(cfg.seed)
        world, tasks, parking = build_world(cfg, rng)
        per_agent = assign_tasks(tasks, parking, rng)

        plans, failed = peg_solve(world, per_agent, parking, max_nodes=10_000)
        assert not failed, f"seed {seed}: FAILED_INIT for {failed}"

        res = run_sim(world, plans, total_tasks=cfg.n_tasks)
        assert res.violations == [], f"seed {seed}: {res.violations[:3]}"
        assert res.completed == cfg.n_tasks, f"seed {seed}: {res.completed}"
        assert res.failed_init == 0 and res.failed_act == 0
        assert res.makespan > 0 and res.throughput == 1.0
        # everyone finishes parked on its own parking cell
        traces = build_traces({a: s for a, s in plans.items() if s}, res.makespan)
        for aid, tr in traces.items():
            assert tr[-1] == parking[aid]


def test_v2_respects_dynamic_blockages():
    for seed in range(5):
        cfg = Config(H=28, W=28, n_agents=10, tasks_per_agent=2, seed=seed)
        rng = np.random.default_rng(cfg.seed)
        world, tasks, parking = build_world(cfg, rng)
        free = world.free_cells()
        rng.shuffle(free)
        for cell in free[:25]:
            t0 = int(rng.integers(0, 40))
            world.add_blockage(cell, t0, t0 + int(rng.integers(3, 12)))
        per_agent = assign_tasks(tasks, parking, rng)

        plans, failed = peg_solve(world, per_agent, parking, max_nodes=8000)
        assert not failed, f"seed {seed}: FAILED_INIT for {failed}"
        res = run_sim(world, plans, total_tasks=cfg.n_tasks)
        assert res.violations == [], f"seed {seed}: {res.violations[:3]}"
        assert res.completed == cfg.n_tasks


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_v2_soak_20_agents_3_tasks(seed):
    cfg = Config(H=32, W=32, n_agents=20, tasks_per_agent=3, seed=seed)
    rng = np.random.default_rng(cfg.seed)
    world, tasks, parking = build_world(cfg, rng)
    per_agent = assign_tasks(tasks, parking, rng)
    plans, failed = peg_solve(world, per_agent, parking, max_nodes=20_000)
    assert not failed, f"seed {seed}: FAILED_INIT for {failed}"
    res = run_sim(world, plans, total_tasks=cfg.n_tasks)
    assert res.violations == [], f"seed {seed}: {res.violations[:3]}"
    assert res.completed == cfg.n_tasks


def test_v2_failed_init_is_reported():
    grid = np.zeros((9, 9), dtype=np.int8)
    grid[:, 4] = SHELF                      # unreachable right-hand region
    world = World(grid)
    parking = [(8, 0)]
    tasks = [[Task(0, (8, 0), (8, 8))]]     # delivery is across the wall
    plans, failed = peg_solve(world, tasks, parking, max_nodes=3000)
    assert failed == {0}
    assert plans[0] is None


# ----------------------------------------------------------------------
# simulator
# ----------------------------------------------------------------------
def test_sim_detects_vertex_and_swap_conflicts():
    world = World(np.zeros((3, 4), dtype=np.int8))
    # vertex clash: both agents on (1,1) at t=2
    clash = {
        0: [PlanStep(0, (1, 0)), PlanStep(1, (1, 1)), PlanStep(2, (1, 1))],
        1: [PlanStep(0, (1, 2)), PlanStep(1, (1, 1)), PlanStep(2, (1, 1))],
    }
    res = run_sim(world, clash)
    assert any("vertex clash" in v for v in res.violations)

    # swap clash: 0 goes (1,1)->(1,2) while 1 goes (1,2)->(1,1)
    swap = {
        0: [PlanStep(0, (1, 1)), PlanStep(1, (1, 2))],
        1: [PlanStep(0, (1, 2)), PlanStep(1, (1, 1))],
    }
    res2 = run_sim(world, swap)
    assert any("swap clash" in v for v in res2.violations)


def test_sim_flags_blockage_and_shelf_entry():
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[1, 1] = SHELF
    world = World(grid)
    world.add_blockage((0, 0), 1, 3)
    plans = {0: [PlanStep(0, (0, 0)), PlanStep(1, (0, 0)), PlanStep(2, (0, 1))]}
    res = run_sim(world, plans)
    assert any("blockage" in v for v in res.violations)


def test_sim_metrics():
    world = World(np.zeros((3, 5), dtype=np.int8))
    plans = {
        0: [PlanStep(0, (0, 0)), PlanStep(1, (0, 1), PICK, item=0),
            PlanStep(2, (0, 2), MOVE), PlanStep(3, (0, 3), DROP, item=0)],
        1: [PlanStep(0, (2, 0)), PlanStep(1, (2, 0), WAIT)],
    }
    res = run_sim(world, plans, total_tasks=1)
    assert res.makespan == 3
    assert res.completed == 1 and res.throughput == 1.0
    assert res.completion_times == {0: 3}
    assert res.sum_service == 3.0 and res.mean_service == 3.0
    assert plan_makespan(plans) == 3
