"""Tests for Space-Time A* and prioritized grounding (M1)."""
from __future__ import annotations

import numpy as np
import pytest

from config import Config
from planner import (
    DROP,
    PICK,
    greedy_task_order,
    plan_to_path,
    prioritized_ground,
    st_astar,
    validate_paths,
)
from reservation import ReservationTable
from world import SHELF, World, assign_tasks, build_world


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _empty_world(h=6, w=8):
    return World(np.zeros((h, w), dtype=np.int8))


def _corridor_with_pocket():
    """1-wide corridor (row 1, cols 0..4) with a single side pocket at (0,3).

    Agent 0 travels the corridor and parks at the far end on top of agent 1's
    start cell.  Agent 1 can only make progress by ducking into the pocket and
    letting agent 0 pass, so a correct space-time planner must use it.
    """
    grid = np.full((3, 5), SHELF, dtype=np.int8)
    grid[1, :] = 0          # corridor
    grid[0, 3] = 0          # pocket
    return World(grid)


# ----------------------------------------------------------------------
# ST-A*
# ----------------------------------------------------------------------
def test_shortest_path_on_empty_grid_is_manhattan():
    world = _empty_world()
    res = ReservationTable()
    path, clash = st_astar(world, res, 0, (0, 0), (5, 7), 0, T_limit=100)
    assert path is not None and not clash
    assert path[0] == ((0, 0), 0)
    assert path[-1][0] == (5, 7)
    assert len(path) - 1 == world.manhattan((0, 0), (5, 7))
    for (c0, t0), (c1, t1) in zip(path, path[1:]):
        assert t1 == t0 + 1
        assert abs(c0[0] - c1[0]) + abs(c0[1] - c1[1]) <= 1


def test_hard_mode_avoids_existing_reservations():
    world = _empty_world()
    res = ReservationTable()
    wall = [((r, 3), 1) for r in range(6)]
    res.add_path(0, wall, park=False)
    path, _ = st_astar(world, res, 1, (0, 0), (5, 7), 0, T_limit=200)
    assert path is not None
    assert all((cell, t) != ((r, 3), 1) for (cell, t) in path for r in range(6))


def test_strict_goal_time_waits_for_park_to_clear():
    world = _empty_world()
    res = ReservationTable()
    res.add_path(0, [((0, 3), 0), ((0, 4), 1)], park=True)  # 0 parks on (0,4)
    path, _ = st_astar(world, res, 1, (0, 0), (0, 4), 0, T_limit=50,
                       strict_goal_time=False)
    assert path is None  # parked cell is unavailable forever
    res2 = ReservationTable()
    path2, _ = st_astar(world, res2, 1, (0, 0), (0, 4), 0, T_limit=50,
                        strict_goal_time=True)
    assert path2 is not None


def test_soft_mode_returns_clashed_agents():
    world = _empty_world()
    res = ReservationTable()
    res.add_path(0, [((0, 1), 0)], park=True)   # agent 0 parks on (0,1) forever
    path, clash = st_astar(world, res, 1, (0, 0), (0, 1), 0, T_limit=50,
                           soft_lambda=10)
    assert path is not None
    assert 0 in clash


def test_blocked_cell_is_detoured():
    world = _empty_world(h=3, w=6)
    world.add_blockage((0, 1), 1, 1)          # blocked exactly at t=1
    res = ReservationTable()
    path, _ = st_astar(world, res, 0, (0, 0), (0, 5), 0, T_limit=50)
    assert path is not None
    assert ((0, 1), 1) not in path


# ----------------------------------------------------------------------
# swap in a 1-wide corridor
# ----------------------------------------------------------------------
def test_corridor_swap_resolves_without_conflicts():
    world = _corridor_with_pocket()
    res = ReservationTable()
    paths = {}

    # Agent 0 goes first (higher priority) and parks at the right end.
    a0, _ = st_astar(world, res, 0, (1, 0), (1, 4), 0, T_limit=100,
                     strict_goal_time=True)
    assert a0 is not None
    res.add_path(0, a0, park=True)
    paths[0] = a0

    # Agent 1 starts where agent 0 parks, so it must duck into the pocket.
    a1, _ = st_astar(world, res, 1, (1, 4), (1, 0), 0, T_limit=200,
                     strict_goal_time=True)
    assert a1 is not None, "planner must resolve the corridor swap via the pocket"
    res.add_path(1, a1, park=True)
    paths[1] = a1

    assert (0, 3) in [c for (c, _t) in a1], "agent 1 should use the side pocket"
    assert validate_paths(world, paths) == []


# ----------------------------------------------------------------------
# prioritized grounding
# ----------------------------------------------------------------------
def test_prioritized_plan_is_conflict_free_over_seeds():
    for seed in range(20):
        cfg = Config(H=30, W=30, n_agents=20, tasks_per_agent=2, seed=seed)
        rng = np.random.default_rng(cfg.seed)
        world, tasks, parking = build_world(cfg, rng)
        per_agent = assign_tasks(tasks, parking, rng)
        jobs = []
        for aid in range(cfg.n_agents):
            ordered = greedy_task_order(world, parking[aid], per_agent[aid])
            jobs.append((aid, parking[aid], parking[aid], ordered))
        res = ReservationTable()
        plans, failed = prioritized_ground(world, res, jobs, t0=0, T_limit=250)
        assert not failed, f"seed {seed} had FAILED_INIT agents"
        paths = {aid: plan_to_path(steps) for aid, steps in plans.items()}
        assert validate_paths(world, paths) == [], f"conflict at seed {seed}"


def test_validate_paths_detects_injected_collision():
    world = _empty_world()
    paths = {
        0: [((0, 0), 0), ((0, 1), 1)],
        1: [((0, 1), 1)],   # vertex clash with agent 0 at t=1
    }
    problems = validate_paths(world, paths)
    assert any("vertex clash" in p for p in problems)


def test_plan_steps_actions_and_pick_drop():
    cfg = Config(H=12, W=12, n_agents=1, tasks_per_agent=1, seed=2)
    world, tasks, parking = build_world(cfg)
    job = [(0, parking[0], parking[0], tasks)]
    res = ReservationTable()
    plans, failed = prioritized_ground(world, res, job, t0=0, T_limit=200)
    assert not failed
    steps = plans[0]
    actions = [s.action for s in steps]
    assert actions.count(PICK) == 1 and actions.count(DROP) == 1
    pick_step = next(s for s in steps if s.action == PICK)
    drop_step = next(s for s in steps if s.action == DROP)
    assert pick_step.cell == tasks[0].pickup
    assert drop_step.cell == tasks[0].delivery
    for a, b in zip(steps, steps[1:]):
        assert b.t - a.t == 1

