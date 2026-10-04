"""Tests for the POCL symbolic layer (M2)."""
from __future__ import annotations

import numpy as np
import pytest

from config import Config
from pocl import (
    EmergencySpec,
    build_goals,
    build_init,
    is_valid_pop,
    linearize,
    pocl,
    pop_task_order,
)
from world import SHELF, Task, World, build_world


def _open_world(h=10, w=10):
    return World(np.zeros((h, w), dtype=np.int8))


def _solve(agent, tasks, start, world, emergency=None, max_nodes=5000):
    plan = pocl(agent, tasks, start, world, max_nodes=max_nodes,
                emergency=emergency)
    assert plan is not None, "POCL failed to find a plan"
    init = build_init(agent, tasks, start)
    goals = build_goals(agent, tasks, emergency)
    assert is_valid_pop(plan, init, goals, n_samples=25)
    return plan


# ----------------------------------------------------------------------
# basic planning
# ----------------------------------------------------------------------
def test_pop_single_task():
    world = _open_world()
    tasks = [Task(0, (0, 0), (5, 5))]
    plan = _solve(0, tasks, (9, 9), world)
    assert pop_task_order(plan) == [0]
    kinds = [a.kind for a in linearize(plan)]
    assert kinds[0] == "travel"
    assert kinds.count("fetch") == 1 and kinds.count("haul") == 1
    fetch = next(a for a in linearize(plan) if a.kind == "fetch")
    haul = next(a for a in linearize(plan) if a.kind == "haul")
    assert fetch.location == tasks[0].pickup
    assert haul.cell_from == tasks[0].pickup and haul.cell_to == tasks[0].delivery


def test_pop_two_tasks_visits_both_pickups():
    world = _open_world()
    tasks = [Task(0, (0, 0), (0, 5)), Task(1, (5, 0), (5, 5))]
    plan = _solve(0, tasks, (9, 9), world)
    assert sorted(pop_task_order(plan)) == [0, 1]
    # hand must be empty before each pick and non-empty before each drop
    state = set(build_init(0, tasks, (9, 9)))
    for act in linearize(plan):
        assert act.pre <= state, f"unsatisfied precondition for {act.label}"
        state -= act.delete
        state |= act.add
    assert {("delivered", 0), ("delivered", 1)} <= state


def test_pop_plan_is_optimal_in_cost():
    world = _open_world()
    tasks = [Task(0, (0, 0), (5, 5))]
    plan = _solve(0, tasks, (9, 9), world)
    # start (9,9) -> pickup (0,0) = 18, +1 fetch tick,
    # pickup -> delivery = 10, +1 haul tick  ==> 30
    assert sum(a.dur for a in linearize(plan)) == 18 + 1 + 10 + 1


def test_pop_is_cheaper_than_worst_task_order():
    world = _open_world()
    # task 0's pickup is adjacent to the start; task 1's pickup is far away and
    # both deliveries coincide, so retrieving task 0 first is strictly cheaper.
    near = Task(0, (8, 9), (0, 0))
    far = Task(1, (0, 9), (0, 0))
    plan = _solve(0, [near, far], (9, 9), world)
    assert pop_task_order(plan)[0] == 0          # cheaper retrieval first
    # travel(1) + fetch(1) + haul(18) + travel(9) + fetch(1) + haul(10) = 40
    # (the other order costs 56)
    assert sum(a.dur for a in linearize(plan)) == 40


def test_pop_with_emergency_orders_emergency_before_finish():
    world = _open_world()
    tasks = [Task(0, (0, 0), (5, 5))]
    em = EmergencySpec(agent=0, cell=(8, 8), hold=4)
    plan = _solve(0, tasks, (9, 9), world, emergency=em)
    acts = linearize(plan)
    assert any(a.kind == "emergency" for a in acts), "emergency action missing"
    em_action = next(a for a in acts if a.kind == "emergency")
    assert em_action.location == (8, 8) and em_action.dur == 4
    # goals (delivered + emergency_done) are all achieved
    state = set(build_init(0, tasks, (9, 9)))
    for act in acts:
        assert act.pre <= state
        state -= act.delete
        state |= act.add
    assert ("emergency_done", 0) in state
    assert ("delivered", 0) in state


def test_pocl_is_deterministic():
    world = _open_world()
    tasks = [Task(0, (0, 0), (0, 6)), Task(1, (7, 0), (7, 6)),
             Task(2, (2, 3), (6, 3))]
    a = _solve(0, tasks, (9, 9), world)
    b = _solve(0, tasks, (9, 9), world)
    assert [x.label for x in linearize(a)] == [x.label for x in linearize(b)]


def test_unreachable_task_returns_none():
    grid = np.zeros((7, 7), dtype=np.int8)
    grid[:, 3] = SHELF                     # split the map into two components
    world = World(grid)
    tasks = [Task(0, (0, 0), (0, 6))]      # delivery is across the wall
    plan = pocl(0, tasks, (6, 0), world, max_nodes=3000)
    assert plan is None


def test_pocl_on_generated_warehouse():
    cfg = Config(H=24, W=24, n_agents=1, tasks_per_agent=4, seed=7)
    world, tasks, parking = build_world(cfg, np.random.default_rng(cfg.seed))
    plan = _solve(0, list(tasks), parking[0], world, max_nodes=8000)
    assert sorted(pop_task_order(plan)) == sorted(t.id for t in tasks)
