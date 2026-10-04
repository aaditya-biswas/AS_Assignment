"""Tests for the world model, config and reservation table (M0 + M1)."""
from __future__ import annotations

import numpy as np
import pytest

from config import Config
from reservation import ReservationTable
from world import (
    FREE,
    SHELF,
    Task,
    TaskStatus,
    World,
    build_world,
    generate_grid,
)


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_defaults_and_validation():
    cfg = Config()
    assert cfg.n_tasks == cfg.n_agents * cfg.tasks_per_agent
    assert cfg.n_blockages(1000) == 50
    with pytest.raises(ValueError):
        Config(replan_mode="nonsense")
    with pytest.raises(ValueError):
        Config(H=3)


def test_config_replace_is_a_copy():
    cfg = Config()
    other = cfg.replace(n_agents=7)
    assert other.n_agents == 7 and cfg.n_agents == 20


# ----------------------------------------------------------------------
# world
# ----------------------------------------------------------------------
def test_generated_world_is_connected_and_deterministic():
    cfg = Config(H=20, W=20, n_agents=5, seed=1)
    w1, t1, p1 = build_world(cfg, np.random.default_rng(cfg.seed))
    w2, t2, p2 = build_world(cfg, np.random.default_rng(cfg.seed))
    assert w1.connected()
    assert np.array_equal(w1.grid, w2.grid)
    assert [(t.pickup, t.delivery) for t in t1] == [(t.pickup, t.delivery) for t in t2]
    assert p1 == p2
    assert len(p1) == cfg.n_agents


def test_parking_and_tasks_are_free_and_distinct():
    cfg = Config(H=24, W=24, n_agents=10, tasks_per_agent=2, seed=3)
    world, tasks, parking = build_world(cfg)
    assert len(parking) == len(set(parking)) == cfg.n_agents
    for c in parking:
        assert world.is_free(c)
    for tk in tasks:
        assert world.is_free(tk.pickup) and world.is_free(tk.delivery)
        assert tk.pickup != tk.delivery
        assert tk.status is TaskStatus.PENDING


def test_grid_outer_ring_is_free():
    grid = generate_grid(15, 15, np.random.default_rng(0))
    assert np.all(grid[0] == FREE) and np.all(grid[-1] == FREE)
    assert np.all(grid[:, 0] == FREE) and np.all(grid[:, -1] == FREE)
    assert (grid == SHELF).any()


def test_bfs_distance_matches_manhattan_on_empty_grid():
    grid = np.zeros((6, 8), dtype=np.int8)
    world = World(grid)
    assert world.distance((0, 0), (5, 7)) == 12
    assert world.manhattan((0, 0), (5, 7)) == 12
    assert world.distance((2, 3), (2, 3)) == 0


def test_distance_infinite_when_walled_off():
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, :] = SHELF  # a full wall with no gap
    world = World(grid)
    assert world.distance((0, 0), (4, 4)) == float("inf")
    assert not world.connected()


def test_blockage_semantics_and_remaining():
    grid = np.zeros((4, 4), dtype=np.int8)
    world = World(grid)
    world.add_blockage((1, 1), 5, 9)
    assert not world.is_blocked((1, 1), 4)
    assert world.is_blocked((1, 1), 5)
    assert world.is_blocked((1, 1), 9)
    assert not world.is_blocked((1, 1), 10)
    assert world.blocked_remaining((1, 1), 7) == 3
    assert world.blocked_remaining((1, 1), 20) == 0
    assert world.active_blockages(6) == [(1, 1)]


# ----------------------------------------------------------------------
# reservation table
# ----------------------------------------------------------------------
def test_vertex_clash_detected():
    res = ReservationTable()
    res.add_path(0, [((0, 0), 0), ((0, 1), 1)], park=False)
    assert res.clashes(1, (1, 1), (0, 1), 0) == {0}


def test_swap_clash_detected():
    res = ReservationTable()
    res.add_path(0, [((0, 0), 0), ((0, 1), 1)], park=False)
    # agent 1 wants (0,1) -> (0,0) during 0..1: that is a swap with agent 0
    assert res.clashes(1, (0, 1), (0, 0), 0) == {0}


def test_park_persists_and_blocks_forever():
    res = ReservationTable()
    res.add_path(0, [((0, 0), 0), ((0, 1), 1)], park=True)
    assert res.park[(0, 1)] == (0, 1)
    assert res.clashes(1, (1, 1), (0, 1), 50) == {0}
    assert res.last_reserved_time((0, 1), excluding_agent=1) == float("inf")
    assert res.last_reserved_time((0, 1), excluding_agent=0) == -1.0


def test_remove_agent_and_remove_after():
    res = ReservationTable()
    res.add_path(0, [((0, 0), 0), ((0, 1), 1), ((0, 2), 2)], park=True)
    res.remove_agent_after(0, 2)
    assert ((0, 2), 2) not in res.vertex
    assert ((0, 1), 1) in res.vertex
    assert (0, 2) not in res.park
    res.remove_agent(0)
    assert res.vertex == {} and res.edge == {} and res.park == {}


def test_snapshot_restore_roundtrip():
    res = ReservationTable()
    res.add_path(3, [((0, 0), 0), ((0, 1), 1)], park=True)
    snap = res.snapshot()
    res.add_path(4, [((1, 0), 0), ((1, 1), 1)], park=False)
    res.restore(snap)
    assert 4 not in res.agents()
    assert res.agents() == {3}
