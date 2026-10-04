"""Hand-built scenarios that *force* negotiation (``SPEC.md`` 9.2).

The random warehouse of :mod:`world` is deliberately open, so a disruption
almost always resolves as a silent local detour (rung ``R1``/``R2`` hard) and
the protocol never speaks -- which is why the plain demo reports ``0
messages``.  These scenarios build a *choke point*: a walled corridor whose
halves are joined by a single door, with one agent parked on that door and a
neighbour locked out of it.  That makes a clash unavoidable, so the
negotiation rungs, the message log, the on-grid bubbles and ``sequence.png``
are all exercised.

:func:`parked_holder_scenario` is the minimal two-agent case asserted by
``tests/test_negotiation.py``; :func:`choke_showcase` scales it to a batch of
agents confined to the corridor.  ``run_demo.py --choke`` drives them, so the
demo GIF shows the protocol instead of a silent repair.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from disruptions import AccidentEvent
from planner import plan_agent, plan_to_path
from reservation import ReservationTable
from world import Cell, Task, World


# ----------------------------------------------------------------------
# the corridor
# ----------------------------------------------------------------------
def corridor_world(H: int = 8, W: int = 8, door_col: int = 3,
                   door_row: Optional[int] = None) -> World:
    """An ``H x W`` world split in two by a wall with a single door.

    Column ``door_col`` is shelf except at ``door_row``, so ``(door_row,
    door_col)`` is the only crossing between the left and the right half.
    """
    if door_row is None:
        door_row = H // 2
    grid = np.zeros((H, W), dtype=np.int8)
    grid[:, door_col] = 1
    grid[door_row, door_col] = 0
    return World(grid)


def _plan_in_order(world: World, starts: Sequence[Cell],
                   parking: Sequence[Cell],
                   tasks_by_agent: Dict[int, List[Task]],
                   *, T_limit: int = 600) -> Dict[int, List]:
    """Priority-plan agents ``0..n-1`` against one shared reservation table."""
    res = ReservationTable()
    plans: Dict[int, List] = {}
    for a in range(len(parking)):
        steps = plan_agent(world, res, a, starts[a], parking[a],
                           tasks_by_agent.get(a, []), 0, T_limit=T_limit)
        assert steps is not None, f"agent {a} cannot be planned"
        plans[a] = steps
        res.add_path(a, plan_to_path(steps), park=True)
    return plans


def parked_holder_scenario():
    """Agent 0 parks on the choke point; agent 1 is locked out of it.

    Agent 1 can only reach its parking cell through ``(3, 3)``, which agent 0
    books forever.  Both agents are disabled at t=2 (0 until t=6, 1 until
    t=10 -- see :data:`PARKED_HOLDER_ACCIDENTS`), so agent 1 needs a
    *negotiated* grant: there is no hard candidate.
    """
    world = corridor_world(8, 8, door_col=3, door_row=3)
    parking = [(3, 3), (0, 0)]
    tasks = {0: [Task(0, (1, 1), (2, 1))], 1: []}
    starts = [(0, 0), (7, 4)]
    plans = _plan_in_order(world, starts, parking, tasks)
    return world, plans, tasks, parking


#: the outage that makes the parked-holder pair un-repairable *without*
#: negotiation: both halves of the clash are down at ``t=2``, so the ladder
#: finds no hard candidate and only a negotiated grant (rung ``R2``) rescues
#: agent 1.  Shared by the demo, the tests and :func:`choke_showcase`.
PARKED_HOLDER_ACCIDENTS = (AccidentEvent(0, 2, 4), AccidentEvent(1, 2, 8))


def parked_holder_showcase() -> dict:
    """The minimal two-agent choke point as a :func:`choke_showcase` dict."""
    world, plans, tasks, parking = parked_holder_scenario()
    return {
        "world": world,
        "plans": plans,
        "per_agent": [tasks[0], tasks[1]],
        "parking": list(parking),
        "starts": [(0, 0), (7, 4)],
        "blockages": [],
        "accidents": list(PARKED_HOLDER_ACCIDENTS),
        "emergencies": [],
        "total_tasks": 1,
        "n_agents": 2,
        "door": (3, 3),
    }


# ----------------------------------------------------------------------
# the scaled showcase
# ----------------------------------------------------------------------
def choke_showcase(n_agents: int = 4, seed: int = 0) -> dict:
    """A corridor choke point that scales to ``n_agents`` robots.

    The first two agents reproduce :func:`parked_holder_scenario` exactly:
    agent 0 parks *on* the door and agent 1, starting on the far side, must
    cross it to reach its parking -- so agent 1 has to negotiate a grant (rung
    ``R2``, ``PROPOSE_REROUTE``/``ACCEPT``/``COMMIT``).  Agents ``2..`` live
    entirely in the right half, so the picture is a busy corridor without
    touching the negotiation plumbing.

    Returns the arguments :func:`sim.simulate_scenario` needs plus the metadata
    a report quotes: ``world``, ``plans``, ``per_agent`` (list per agent),
    ``parking``, ``starts``, ``blockages``, ``accidents``, ``emergencies``,
    ``total_tasks``, ``n_agents`` and ``door``.
    """
    if n_agents < 2:
        raise ValueError("choke_showcase needs at least 2 agents")
    rng = np.random.default_rng(seed)
    H = 8
    W = max(8, 6 + n_agents)          # the right half must seat the extras
    door_col, door_row = 3, 3
    world = corridor_world(H, W, door_col, door_row)
    door = (door_row, door_col)

    #: the proven pair: agent 0 parks on the door, agent 1 is locked out
    parking: List[Cell] = [door, (0, 0)]
    starts: List[Cell] = [(0, 0), (H - 1, door_col + 1)]
    used = {door, (0, 0), (H - 1, door_col + 1), (1, 1), (2, 1)}

    #: the extras never cross the door, so they only add life to the panel
    per_agent: List[List[Task]] = [[Task(0, (1, 1), (2, 1))], []]
    pool = [(r, c) for c in range(door_col + 1, W) for r in range(H)
            if (r, c) not in used]
    order = [pool[i] for i in rng.permutation(len(pool))]
    if len(order) < 4 * (n_agents - 2):
        raise ValueError("corridor too small for that many agents")
    k = 0
    for a in range(2, n_agents):
        starts.append(order[k])
        parking.append(order[k + 1])
        per_agent.append([Task(a, order[k + 2], order[k + 3])])
        k += 4
    total_tasks = sum(len(t) for t in per_agent)

    plans = _plan_in_order(world, starts, parking,
                           {a: t for a, t in enumerate(per_agent)})
    #: both halves of the pair are down early: no hard candidate -> negotiate
    accidents = list(PARKED_HOLDER_ACCIDENTS)
    return {
        "world": world,
        "plans": plans,
        "per_agent": per_agent,
        "parking": parking,
        "starts": starts,
        "blockages": [],
        "accidents": accidents,
        "emergencies": [],
        "total_tasks": total_tasks,
        "n_agents": n_agents,
        "door": door,
    }



# ----------------------------------------------------------------------
# blockages that are guaranteed to bite
# ----------------------------------------------------------------------
def blockages_on_routes(plans: Dict[int, List], n: int,
                        rng: np.random.Generator, *,
                        lead: int = 3, duration: int = 6,
                        t_min: int = 5) -> List:
    """``n`` blockages placed *on* an agent's route (``SPEC`` 10).

    :func:`disruptions.random_blockages` picks free cells at random, so most of
    them are never driven through and produce no disruption at all.  Here each
    cell is taken from an agent's own ``PlanStep`` sequence with a lead time
    ``lead`` before the agent arrives, so the window always contains the step
    and the repair loop really has to react -- which is what makes a
    multi-pass case study (and its four repair cards) reproducible.
    """
    from disruptions import BlockageEvent

    cand: List[tuple] = []
    for aid in sorted(plans):
        for s in (plans.get(aid) or []):
            if s.t >= t_min and s.action in ("MOVE", "WAIT", "PICK", "DROP"):
                cand.append((s.cell, s.t))
    seen: set = set()
    unique: List[tuple] = []
    for cell, t in cand:                     # deterministic order, deduplicated
        if cell in seen:
            continue
        seen.add(cell)
        unique.append((cell, t))
    if not unique:
        return []
    idx = rng.permutation(len(unique))
    out: List = []
    for i in idx:
        if len(out) >= n:
            break
        cell, t = unique[int(i)]
        out.append(BlockageEvent(cell, max(1, t - lead), duration))
    out.sort(key=lambda e: (e.t0, e.cell))
    return out

