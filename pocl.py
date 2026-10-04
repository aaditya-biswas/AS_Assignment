"""Partial-Order Causal-Link (POCL) planner over a STRIPS warehouse domain.

This is the L1 symbolic layer of ``SPEC.md`` section 7.  POCL is run
**per agent** (agents share no symbolic preconditions), which keeps branching
small; the cross-agent coupling is handled later by the PEG / reservation
layer.

Facts are tuples, e.g. ``("at", agent, cell)``, ``("holding", agent, item)``,
``("hand_empty", agent)``, ``("operational", agent)``,
``("at_item", item, cell)``, ``("delivered", item)``, ``("emergency_done", a)``.
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from world import Cell, Task, World

Fact = tuple


# ----------------------------------------------------------------------
# ground actions
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    kind: str                       # goto | pick | drop | emergency | start | finish
    agent: int
    pre: FrozenSet[Fact]
    add: FrozenSet[Fact]
    delete: FrozenSet[Fact]
    dur: int = 0
    label: str = ""
    cell_from: Optional[Cell] = None
    cell_to: Optional[Cell] = None
    item: Optional[int] = None
    location: Optional[Cell] = None

    @property
    def is_real(self) -> bool:
        return self.kind not in ("start", "finish")


@dataclass(frozen=True)
class EmergencySpec:
    agent: int
    cell: Cell
    hold: int = 3


def build_library(agent: int, tasks: Sequence[Task], start: Cell, world: World,
                  emergency: Optional[EmergencySpec] = None) -> List[Action]:
    """Instantiate the macro-action library for ``agent``.

    Three macro-operators keep the POP branching factor small while still
    exposing a genuine causal-link/threat structure (the single hand is what
    forces retrieval steps of *different* tasks to be ordered):

    * ``Travel(x -> y)``  -- move between two waypoints,
    * ``Fetch(o)``        -- arrive at ``pickup(o)`` and pick the item up
      (consumes the only hand),
    * ``Haul(o)``         -- carry the item to ``delivery(o)`` and drop it
      (releases the hand).
    * ``Emergency``       -- optional evacuation hold at a marshalling cell.

    ``Fetch``/``Haul`` are split (rather than fused into one ``Serve``) on
    purpose: a fused action could not *delete* ``hand_empty`` and therefore
    POCL would never order two tasks' retrieval steps, losing the very
    ordering decision this layer exists to make.
    """
    pickups = sorted({tk.pickup for tk in tasks})
    deliveries = sorted({tk.delivery for tk in tasks})
    em_cell = emergency.cell if emergency is not None else None

    # The agent is only ever "at rest" on its start cell, a delivery cell, or
    # the marshalling cell -- so Travel only needs those as sources.
    sources = {start} | set(deliveries)
    if em_cell is not None:
        sources.add(em_cell)
    targets = set(pickups)
    if em_cell is not None:
        targets.add(em_cell)

    acts: List[Action] = []
    for x in sorted(sources):
        for y in sorted(targets):
            if x == y:
                continue
            d = world.distance(x, y)
            if not math.isfinite(d):
                continue
            acts.append(Action(
                "travel", agent,
                pre=frozenset({("at", agent, x), ("operational", agent)}),
                add=frozenset({("at", agent, y)}),
                delete=frozenset({("at", agent, x)}),
                dur=int(d), label=f"Travel({agent},{x}->{y})",
                cell_from=x, cell_to=y))
    for tk in tasks:
        o = tk.id
        if not math.isfinite(world.distance(tk.pickup, tk.delivery)):
            continue
        acts.append(Action(
            "fetch", agent,
            pre=frozenset({("at", agent, tk.pickup),
                           ("at_item", o, tk.pickup),
                           ("hand_empty", agent), ("operational", agent)}),
            add=frozenset({("holding", agent, o)}),
            delete=frozenset({("hand_empty", agent), ("at_item", o, tk.pickup)}),
            dur=1, label=f"Fetch({agent},item{o})",
            location=tk.pickup, item=o))
        acts.append(Action(
            "haul", agent,
            pre=frozenset({("at", agent, tk.pickup), ("holding", agent, o),
                           ("operational", agent)}),
            add=frozenset({("delivered", o), ("hand_empty", agent),
                           ("at", agent, tk.delivery)}),
            delete=frozenset({("holding", agent, o), ("at", agent, tk.pickup)}),
            dur=int(world.distance(tk.pickup, tk.delivery)) + 1,
            label=f"Haul({agent},item{o})",
            cell_from=tk.pickup, cell_to=tk.delivery, item=o,
            location=tk.delivery))
    if emergency is not None:
        acts.append(Action(
            "emergency", agent,
            pre=frozenset({("at", agent, emergency.cell), ("operational", agent)}),
            add=frozenset({("emergency_done", agent)}),
            delete=frozenset(),
            dur=max(1, int(emergency.hold)), label=f"Emergency({agent})",
            location=emergency.cell))
    return acts


def build_init(agent: int, tasks: Sequence[Task], start: Cell) -> FrozenSet[Fact]:
    facts = {("at", agent, start), ("operational", agent), ("hand_empty", agent)}
    for tk in tasks:
        facts.add(("at_item", tk.id, tk.pickup))
    return frozenset(facts)


def build_goals(agent: int, tasks: Sequence[Task],
                emergency: Optional[EmergencySpec] = None) -> FrozenSet[Fact]:
    goals = {("delivered", tk.id) for tk in tasks}
    if emergency is not None:
        goals.add(("emergency_done", agent))
    return frozenset(goals)


# ----------------------------------------------------------------------
# partial plan
# ----------------------------------------------------------------------
@dataclass
class POP:
    steps: List[Action]
    order: Set[Tuple[int, int]]
    links: List[Tuple[int, int, Fact]]

    def copy_with(self, *, steps=None, order=None, links=None) -> "POP":
        return POP(
            self.steps if steps is None else steps,
            set(self.order if order is None else order),
            list(self.links if links is None else links),
        )


def _succ_map(order: Iterable[Tuple[int, int]]) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for (a, b) in sorted(order):
        adj.setdefault(a, []).append(b)
    return adj


def reachable(order: Set[Tuple[int, int]], src: int, dst: int) -> bool:
    if src == dst:
        return True
    adj = _succ_map(order)
    stack = [src]
    seen = {src}
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y == dst:
                return True
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


def add_edge(order: Set[Tuple[int, int]],
             i: int, j: int) -> Optional[Set[Tuple[int, int]]]:
    """Add ``i -> j`` unless it is a self-loop or would create a cycle."""
    if i == j or reachable(order, j, i):
        return None
    if (i, j) in order:
        return set(order)
    return set(order) | {(i, j)}


def is_acyclic(order: Iterable[Tuple[int, int]]) -> bool:
    adj = _succ_map(order)
    nodes = set(adj)
    for (_, b) in order:
        nodes.add(b)
    color: Dict[int, int] = {}

    def dfs(u: int) -> bool:
        color[u] = 1
        for v in adj.get(u, ()):
            c = color.get(v, 0)
            if c == 1:
                return False
            if c == 0 and not dfs(v):
                return False
        color[u] = 2
        return True

    return all(dfs(u) for u in sorted(nodes) if color.get(u, 0) == 0)


# ----------------------------------------------------------------------
# flaw detection and refinement
# ----------------------------------------------------------------------
def open_conditions(plan: POP) -> List[Tuple[int, Fact]]:
    """``(step_index, condition)`` pairs that no causal link supports."""
    out: List[Tuple[int, Fact]] = []
    for c, act in enumerate(plan.steps):
        if c == 0:
            continue
        supported = {cond for (_p, cc, cond) in plan.links if cc == c}
        for cond in sorted(act.pre):
            if cond not in supported:
                out.append((c, cond))
    return out


def find_threat(plan: POP):
    """Return ``(link_index, threat_step)`` for the first threat, else None."""
    for li, (p, c, cond) in enumerate(plan.links):
        for t, act in enumerate(plan.steps):
            if t in (p, c):
                continue
            if cond in act.delete:
                if not reachable(plan.order, t, p) and not reachable(plan.order, c, t):
                    return (li, t)
    return None


def pick_flaw(plan: POP):
    """Threats first; otherwise the open condition with the fewest achievers."""
    threat = find_threat(plan)
    if threat is not None:
        return ("threat", threat[0], threat[1])
    best_key = None
    best = None
    for c, cond in open_conditions(plan):
        n = sum(1 for i, a in enumerate(plan.steps) if i != c and cond in a.add)
        key = (n, c, repr(cond))
        if best_key is None or key < best_key:
            best_key = key
            best = (c, cond)
    if best is None:
        return None
    return ("open", best[0], best[1])


def _action_signature(act: Action):
    return (act.kind, act.cell_from, act.cell_to, act.item, act.location)


def pop_key(plan: POP):
    """Canonical key used to de-duplicate partial plans during search."""
    steps = tuple(_action_signature(a) for a in plan.steps)
    order = tuple(sorted(plan.order))
    links = tuple(sorted((p, c, repr(cond)) for (p, c, cond) in plan.links))
    return (steps, order, links)


def refine(plan: POP, flaw, lib: Sequence[Action]) -> List[POP]:
    """All one-step refinements of ``flaw`` (empty list = dead end)."""
    out: List[POP] = []
    if flaw[0] == "threat":
        _, li, t = flaw
        p, c, _cond = plan.links[li]
        for (i, j) in ((t, p), (c, t)):
            o = add_edge(plan.order, i, j)
            if o is not None:
                out.append(plan.copy_with(order=o))
        return out

    _, c, cond = flaw
    # (a) reuse an existing producer
    for i, act in enumerate(plan.steps):
        if i == c or cond not in act.add:
            continue
        if reachable(plan.order, c, i):
            continue
        o = add_edge(plan.order, i, c)
        if o is None:
            continue
        out.append(plan.copy_with(order=o, links=plan.links + [(i, c, cond)]))
    # (b) add a new library action (never duplicate an existing step)
    present = {_action_signature(a) for a in plan.steps}
    for act in lib:
        if cond not in act.add:
            continue
        if not act.add:
            continue
        if _action_signature(act) in present:
            continue
        idx = len(plan.steps)
        o = add_edge(plan.order, 0, idx)
        if o is None:
            continue
        o = add_edge(o, idx, 1)
        if o is None:
            continue
        o = add_edge(o, idx, c)
        if o is None:
            continue
        out.append(POP(plan.steps + [act], o, plan.links + [(idx, c, cond)]))
    return out


def _plan_cost(plan: POP) -> int:
    """Sum of the durations of the real macro-steps (a lower bound on makespan)."""
    return sum(act.dur for act in plan.steps if act.is_real)


def heuristic(plan: POP) -> int:
    return _plan_cost(plan) + len(open_conditions(plan))


# ----------------------------------------------------------------------
# search
# ----------------------------------------------------------------------
def pocl(agent: int, tasks: Sequence[Task], start: Cell, world: World,
         max_nodes: int = 20_000,
         emergency: Optional[EmergencySpec] = None,
         initial: Optional[POP] = None) -> Optional[POP]:
    """Best-first POCL search.  Returns a valid ``POP`` or ``None``.

    Pass ``initial`` to *reuse* an existing plan (R4 refinement from ``P0``).
    """
    lib = build_library(agent, tasks, start, world, emergency)
    init = build_init(agent, tasks, start)
    goals = build_goals(agent, tasks, emergency)

    if initial is not None:
        root = initial.copy_with(order=set(initial.order), links=list(initial.links),
                                 steps=list(initial.steps))
        # re-root: link FINISH to new goals
        finish = root.steps[1]
        root.steps = list(root.steps)
        root.steps[1] = Action("finish", agent, goals, frozenset(), frozenset(), 0, "FINISH")
    else:
        start_act = Action("start", agent, frozenset(), init, frozenset(), 0, "START")
        finish_act = Action("finish", agent, goals, frozenset(), frozenset(), 0, "FINISH")
        root = POP([start_act, finish_act], {(0, 1)}, [])

    if not goals:
        return root

    counter = itertools.count()
    heap = [(heuristic(root), next(counter), root)]
    seen = {pop_key(root)}
    nodes = 0
    while heap:
        _f, _c, plan = heapq.heappop(heap)
        nodes += 1
        if nodes > max_nodes:
            return None
        if not is_acyclic(plan.order):
            continue
        flaw = pick_flaw(plan)
        if flaw is None:
            if is_valid_pop(plan, init, goals):
                return plan
            continue
        for child in refine(plan, flaw, lib):
            key = pop_key(child)
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(heap, (heuristic(child), next(counter), child))
    return None


# ----------------------------------------------------------------------
# validation and views
# ----------------------------------------------------------------------
def _topo_order(plan: POP, rng: Optional[random.Random] = None) -> List[int]:
    n = len(plan.steps)
    indeg = [0] * n
    adj: List[List[int]] = [[] for _ in range(n)]
    for (i, j) in sorted(plan.order):
        adj[i].append(j)
        indeg[j] += 1
    avail = [i for i in range(n) if indeg[i] == 0]
    out: List[int] = []
    while avail:
        if rng is None:
            k = 0
        else:
            k = rng.randrange(len(avail))
        i = avail.pop(k)
        out.append(i)
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                avail.append(j)
    return out


def linearize(plan: POP) -> List[Action]:
    """A deterministic linearisation (real steps only, START/FINISH removed)."""
    return [plan.steps[i] for i in _topo_order(plan) if plan.steps[i].is_real]


def execute_order(plan: POP, order: Sequence[int]) -> Optional[Set[Fact]]:
    """Execute ``order`` STRIPS-style; returns the final state or None on failure."""
    state: Set[Fact] = set()
    for i in order:
        act = plan.steps[i]
        if act.kind == "start":
            state |= set(act.add)
            continue
        if act.kind == "finish":
            continue
        if not act.pre <= state:
            return None
        state -= act.delete
        state |= act.add
    return state


def is_valid_pop(plan: POP, init: FrozenSet[Fact], goals: FrozenSet[Fact],
                 n_samples: int = 20, rng_seed: int = 0) -> bool:
    """All preconditions supported, no threats, acyclic, and every sampled
    linearisation reaches the goals."""
    if not is_acyclic(plan.order):
        return False
    for c, act in enumerate(plan.steps):
        if c == 0:
            continue
        supported = {cond for (_p, cc, cond) in plan.links if cc == c}
        if not act.pre <= supported:
            return False
    if find_threat(plan) is not None:
        return False
    rng = random.Random(rng_seed)
    for _ in range(max(1, n_samples)):
        order = _topo_order(plan, rng)
        if len(order) != len(plan.steps):
            return False
        state = execute_order(plan, order)
        if state is None or not goals <= state:
            return False
    return True


def pop_to_path(plan: POP) -> List[Action]:
    return linearize(plan)


def pop_task_order(plan: POP) -> List[int]:
    """Item ids in the order the agent picks them up (a linearisation view)."""
    order: List[int] = []
    for act in linearize(plan):
        if act.kind == "fetch" and act.item is not None:
            order.append(act.item)
    return order


def pop_waypoints(plan: POP) -> List[Action]:
    """The macro actions of a deterministic linearisation (no START/FINISH)."""
    return linearize(plan)


def pop_summary(plan: POP) -> str:
    lines = [f"POP with {len(plan.steps)} steps, {len(plan.order)} orderings, "
             f"{len(plan.links)} causal links"]
    for i, act in enumerate(plan.steps):
        if act.kind == "start":
            label = "<START>"
        elif act.kind == "finish":
            label = "<FINISH>"
        else:
            label = act.label
        lines.append(f"  [{i}] {label}")
    for (i, j) in sorted(plan.order):
        lines.append(f"  order {i} -> {j}")
    for (p, c, cond) in plan.links:
        lines.append(f"  link {p} --{cond}--> {c}")
    return "\n".join(lines)


