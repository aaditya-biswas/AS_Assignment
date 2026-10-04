# Multi-Agent Warehouse with Local Plan Repair — Consolidated Specification

This document reconciles the **two** specification drafts ("Spec v1: executable
specification" and "Spec v2: POCL + plan modification") into a single,
implementable, self-consistent source of truth.

## 0. Reconciliation rules

1. **v1 is the base contract.** World model, config, `ReservationTable`, ST-A*
   internals, base visualization, the experiment harness and the report
   checklist are inherited from v1 unless overridden below.
2. **v2 overrides** wherever the two disagree (task planning, plan
   representation, repair, metrics, visualization extensions, new modules).
3. Where both are silent or where they need a unifying decision, the
   "Merged decision" tables in this document define the behavior.

### 0.1 Conflict resolution matrix

| # | Topic | v1 | v2 | Merged decision |
|---|---|---|---|---|
| 1 | Task ordering | greedy nearest-pickup (BFS) | POCL, order emerges from threats | **POCL** is canonical; greedy kept as a baseline helper |
| 2 | Plan representation | time-indexed `PlanStep(t,cell,action)` | Plan Execution Graph (partial order) | **PEG** is canonical; `PlanStep` is a view over PEG nodes |
| 3 | Repair module | `repair.py` / `RepairManager`, levels 0-2 | `modify.py` / `modify_plan`, R0-R4, 5 phases | **`modify.py`**; the v1 engine is the optional `timeindexed` ablation |
| 4 | Messaging module | `messaging.py`, `MessageBus`, `REQUEST_YIELD` | `negotiation.py`, `PROPOSE_ORDER`/`PROPOSE_REROUTE` | **`negotiation.py`** keeps `MessageBus`; new message types |
| 5 | Altered metric | one def: remaining `(cell,t)` diff | 4 variants | **`altered_plan` primary**; v1 def = `altered_naive` ablation |
| 6 | Objective | minimise agents whose remaining plan changes | delays do not count | minimise **`altered_plan`** (structural); report naive separately |
| 7 | Candidate choice | lexicographic `(n_altered, delay)` | `delay + beta_alter*n_altered` | **v2 weighted cost** (lexicographic = beta -> infinity limit) |
| 8 | Config | 18 fields | + 5 fields, optional `timeindexed` | **union**; mode enum gains `timeindexed` |
| 9 | Dependencies | numpy, matplotlib, pandas, pillow, pytest, reportlab, ffmpeg | + networkx | **union** |
| 10 | Planner symbol | `prioritized_plan` | "prioritized grounding" -> PEG | **`prioritized_ground`**; used as the mocked "global planner" |
| 11 | Global-planner ban | never call `prioritized_plan` in repair | never call global planner outside baseline | one **`global_plan`** symbol; test asserts 0 calls |
| 12 | Breakdown | shift remaining plan by D | `Break`/`Fix` exogenous steps | **v2 mechanism** (retime realizes the shift) |
| 13 | Permanent breakdown | reassign nearest; carried item -> pickup at cell | R4 via POCL, orphan goals | **v2 R4**; identical semantics |
| 14 | Validator | vertex, swap, blocked, shelf | + `peg.is_acyclic()` | **union** |
| 15 | Fallback | `T_max` unfinished = failure | `failure_wait` per-agent cap | **both**, defined separately |
| 16 | Viz | grid animation | + POP graph, Gantt, waiting arrows, repair freeze-frames | **union**; new `viz_plan.py` |
| 17 | Experiments | columns + plots | extra columns + plots | **union** |
| 18 | Report | 9 sections + checklist | 12 sections + citations | **v2 outline**, keep v1 checklist items |

## 1. Goal and hard constraints

Simulate robot agents on a 2D grid warehouse. Initial plans come from
prioritized planning over a reservation table; disruptions are repaired
**locally** without ever calling the global planner.

Hard requirements (union of both specs):

1. Initial plan comes from Space-Time A* + a reservation table
   (and, in the v2 layer, per-agent POCL over STRIPS tasks).
2. **Repair must never call the global planner** and must minimise the number
   of agents whose plan *structure* changes (`altered_plan`).
3. Disrupted agents negotiate by counted message passing within `comm_radius`.
4. Metrics: sum of individual completion times, makespan, altered agents per
   disruption, messages, repair CPU time, success rate (plus the v2 variants).
5. Experiments sweep number of agents and dynamic-obstacle density.
6. Rich visualization with GIF/MP4 export and screenshots.
7. Everything seeded and reproducible.
8. Dependencies: `numpy`, `matplotlib`, `pandas`, `pillow`, `pytest`,
   `reportlab`, `networkx`; optional `ffmpeg`.

## 2. Project layout

```
warehouse_repair/            (this repository root)
  SPEC.md  README.md  requirements.txt  pytest.ini  conftest.py
  config.py
  world.py          reservation.py    planner.py
  pocl.py           peg.py            agent.py      negotiation.py
  disruptions.py    modify.py         sim.py        baselines.py
  metrics.py        viz.py            viz_plan.py
  experiments.py    make_report.py    run_demo.py
  tests/
  outputs/
```

## 3. World model

* `H x W` numpy grid, `0` free, `1` shelf. Coordinates `(row, col)`.
* Moves: wait `(0,0)` and the four neighbours; 1 step per move/wait.
* Map generator: horizontal 1x4 / 1x6 shelf bars on rows 3, 6, 9, ...,
  leaving fully-free aisles between them; the outer ring is free. After
  generation a BFS must show the free space is one connected component.
* `Task(id, pickup, delivery, status)`; `status in {PENDING, PICKED, DONE}`.
* Dynamic blockages: `blocked_until: dict[cell] -> list[(t_start, t_end)]`.
  `world.is_blocked(cell, t)` is true iff `t` lies in a window. A blockage
  only prevents **entering** a cell at `t`; an agent already there may leave.
  A blockage may only spawn on a cell that is: shelf-free, currently
  unoccupied, not any agent's start/pickup/delivery, and on some agent's
  future planned path. If the chosen cell is occupied at spawn time, the
  blockage is postponed one tick (v2); if it is on no future path, it is
  skipped.

## 4. Config

`config.Config` (dataclass) fields — union of v1 and v2:

`H=30, W=30, n_agents=20, tasks_per_agent=3, seed=0, T_max=600,
comm_radius=6, max_depth=3, lambda_soft=15, delay_threshold=8,
obstacle_density=0.05, n_breakdowns=2, n_emergencies=2,
breakdown_duration_range=(5,15), p_permanent_breakdown=0.2,
block_duration_range=(5,25), replan_mode in
{"negotiate","self_only","global","timeindexed"}, pocl_max_nodes=20000,
repair_pocl_max_nodes=3000, k_candidates=3, beta_alter=10, failure_wait=60`.

`obstacle_density = n_blockage_events / n_free_cells`, blockage start times
uniform over `[1, T_est]` where `T_est` is the undisrupted makespan. Log the
"fraction of cell-time blocked" too.

## 5. Reservation table

```
vertex: (cell, t) -> agent_id
edge:   (from_cell, to_cell, t) -> agent_id      # move from->to during t..t+1
park:   cell -> (agent_id, t_from)               # sits forever from t_from
```
Methods: `add_path`, `remove_agent`, `remove_agent_after`, `clashes`,
`last_reserved_time`, `snapshot`/`restore`. Finished agents persist as `park`,
so every plan ends at a dedicated parking cell.

`clashes(agent_id, c_from, c_to, t)` returns the set of *other* agents found
via (a) vertex `(c_to, t+1)`, (b) swap edge `(c_to, c_from, t)`, (c) park
`c_to` with `t+1 >= t_from`.

## 6. Planner (L2 grounding)

* Per-goal BFS distance map on the static grid, cached.
* `st_astar(world, res, agent_id, start, goal, t0, *, soft_lambda=None,
  extra_hard=frozenset(), T_limit, strict_goal_time=False)`.
  State `(cell, t)`, `f = g + h(cell)`, `g` = steps + `soft_lambda` per clash.
  Invalid successors: shelf, blocked at `t+1`, in `extra_hard`, or (hard mode)
  a non-empty clash set. Reject `t - t0 > T_limit`.
  **Goal test:** `cell == goal` and, *only when `strict_goal_time` is true*
  (final parking leg), `t >= last_reserved_time(goal)`; intermediate waypoint
  legs need only a clash-free arrival (fixes v1's over-strict rule).
* `greedy_task_order` (baseline) and `prioritized_ground` (canonical) plan
  agents in priority order against a shared table; on failure the `T_limit`
  is doubled once, then the agent is marked `FAILED_INIT`.
* `validate_paths` checks vertex, swap, blocked-cell and shelf violations.

## 7. Three plan layers (v2)

```
L1  SYMBOLIC POP (per agent)   POCL over STRIPS: Goto, Pick, Drop, Emergency
L2  GROUNDED MICRO-PLAN        Move / Act steps, timed, via ST-A*
L3  PLAN EXECUTION GRAPH       one partial order for ALL agents
     edges: seq (within agent), logic (Pick->Drop), res (cell hand-over)
     schedule = earliest start by longest path ("retime")
```

POCL <-> warehouse mapping (report table): Step = Goto/Pick/Drop (L1) or
Move/Act (L2/L3); causal link = `at`/`holding`/`hand_empty` (L1) or
`Free(cell)` hand-over (L3); ordering constraint = precedence edge; threat =
a new visit inside another agent's hand-over slot; threat resolution =
promotion/demotion = "who waits for whom"; open condition = unsupported
precondition (`operational(a)`, `delivered(o)`); exogenous step =
`Block`/`Unblock`, `Break`/`Fix`; plan reuse = start from the existing plan.

**Payoff:** because the plan is a partial order, a delay propagates through
edges and downstream agents are only *delayed*, not *altered*. This is the
central claim of the report and is measured by comparing `altered_plan` with
`altered_naive`.

### 7.1 STRIPS domain

Facts are tuples (`("at", a, l)`, `("holding", a, o)`, ...).

| Action | Pre | Add | Delete | Dur |
|---|---|---|---|---|
| `Goto(a,x,y)` | at(a,x), operational(a) | at(a,y) | at(a,x) | BFS dist(x,y) |
| `Pick(a,o,l)` | at(a,l), at_item(o,l), hand_empty(a), operational(a) | holding(a,o) | hand_empty(a), at_item(o,l) | 1 |
| `Drop(a,o,l)` | at(a,l), holding(a,o), operational(a) | delivered(o), hand_empty(a) | holding(a,o) | 1 |
| `Emergency(a,E)` | at(a,E), operational(a) | emergency_done(a) | - | hold_steps |

Ground actions are instantiated only over each agent's own location set
(~2*tasks+2), keeping branching small. Per-agent POCL is sound because agents
share no symbolic preconditions.

### 7.2 POCL search

Best-first over plans. `pick_flaw` = threats first, then the open condition
with the fewest achievers. Open conditions are refined by reuse (an existing
producer) or by a new library action. Threats are resolved by promotion or
demotion (add an ordering). `f(P) = g(P) + len(open_conds)`,
`g(P) = sum(Goto durations) + number of non-Goto steps`.

`is_valid_pop(P, init)`: all preconditions supported, no threats, acyclic;
then sample 20 random topological orders and execute each STRIPS-style from
the initial state — all must reach the goals. `linearize(P)` gives a
per-agent chain.

### 7.3 PEG construction

Micro-steps have a never-reused `id` and a `parent_l1_step`. Edges
`src -> (dst, kind, weight)`:

* `seq`: consecutive micro-steps of one agent, weight = duration of src.
* `logic`: Pick -> Drop of the same item (also covered by seq).
* `res`: weight **0**, start-to-start. Per cell build the occupancy sequence
  `occ[c] = [(agent, enter_step, leave_step), ...]` sorted by arrival; for
  consecutive visits of different agents add `p.leave_step -> q.enter_step`.

An agent occupies a cell from `enter_start+1` until `leave_start`; weight 0
allows "following" (q enters as p leaves in the same tick). **Swap and
rotation conflicts appear as cycles**, so `peg.is_acyclic()` is the
collision-safety check. Theorem (report): any schedule respecting all edges is
free of vertex and swap collisions.

`retime(peg, t_now)`: STARTED/DONE steps keep actual times; other steps are
scheduled in topological order by `s.t = max([t_now] + [anchor] +
[p.t + w(p, s) for p in preds(s)])` where `s.anchor` is used only when it is
not `None` (**never `anchor or t0`**, which would drop an anchor of 0).

PEG ops used by repair: `insert_visit`, `remove_leg`,
`insert_exogenous_occupant`, `from_peg_to_reservations(peg, exclude, t_now)`,
`signature(agent)`. Deep-copy for tentative negotiation.

## 8. Disruptions as plan flaws

All events come from the seeded generator *before* any repair, so every mode
sees identical events.

| Event | Exogenous steps | Resulting flaws |
|---|---|---|
| Cell blockage (c,t,D) | `Block(c)`@t, `Unblock(c)`@t+D | later visits of c get an edge from `Unblock` (demotion = wait); if too costly, reroute |
| Breakdown (a,t,D) | `Break(a)`@t, `Fix(a)`@t+D, `Fix -> a.next` | `operational(a)` violated for D ticks; followers wait via `res`; reroute if delay > threshold |
| Permanent breakdown | `Break(a)` only | orphan `delivered(o)` goals; `at_item(o,cell)` if carrying; other visits to a's cell reroute |
| Emergency (a,E,hold) | `Emergency(a,E)` before a's remaining steps | open `emergency_done(a)`, `at(a,E)`; POCL inserts the detour and restores the old `at` link |

## 9. Plan modification (`modify.py`)

Five phases: **diagnose -> unrefine -> refine -> negotiate -> commit**
(Kambhampati-Hendler style: retract only the invalid parts, then refine with a
least-change search).

1. **DIAGNOSE:** copy the PEG, insert exogenous steps, `retime`, compute
   affected set `A0` and the flaw list (threatened `res` links, unsupported
   preconditions, unreachable legs, orphan goals).
2. **UNREFINE:** retract only what is invalid. Ordering-resolvable threats
   retract nothing; a leg through a blocked cell retracts that leg's unstarted
   micro-steps; a permanent breakdown retracts the agent's unstarted steps.
   Never touch STARTED/DONE steps; re-root the executed prefix as `START'`.
3. **REFINE + NEGOTIATE:** escalation ladder R0-R4 (below).
4. **COMMIT:** two-phase; assert `peg.is_acyclic()` and run the simulator
   validator, else ABORT and roll back.

### 9.1 Escalation ladder (stop at the first acceptable candidate)

A candidate is acceptable if total added delay <= `delay_threshold`; candidates
are compared by `cost = added_delay_total + beta_alter * altered_plan_count`.

| Rung | Operation | Who may change |
|---|---|---|
| R0 | retime only (new ordering edges = waiting) | agents gaining an incoming edge |
| R1 | self-reroute the affected Goto leg with **hard** ST-A* (fixed endpoints), then `insert_visit` | the agent + agents that now wait |
| R2 | negotiation (9.2) within `comm_radius`, depth <= `max_depth` | the negotiation chain |
| R3 | R2 with `comm_radius += 3` (at most twice) | same, wider |
| R4 | reassignment via POCL (9.3), triggered by permanent breakdown / unreachable target | broken agent + receivers |
| Fallback | keep R0 (agents wait), retry each tick; after `failure_wait` ticks record `success=False`. **Never** call the global planner | - |

### 9.2 Negotiation as POP operations (`negotiation.py`)

```
negotiate(i, depth, chain):
    path, C = st_astar_soft(...)          # C = clashed agents (threatened res links)
    if C empty: apply to a PEG copy; return success
    for j in C within comm_radius, j not in chain:
        send PROPOSE_ORDER(i->j, visit, link, 'before'|'after')     # cheap: j only waits
        j: 1. add ordering to a copy; if acyclic and delay(j) <= 2*delay_threshold -> ACCEPT(delay)
           2. else j reroutes its own leg with hard ST-A* (others frozen) -> ACCEPT(delay, new_leg)
           3. else if depth < max_depth: j negotiates as sub-initiator (chain+[i]) -> ACCEPT/REJECT
           4. else REJECT(reason)
    if all ACCEPT: COMMIT to all involved; check acyclic -> success
    else: ABORT; add the rejected clash as a hard constraint; retry <= 3 times
```

Messages: `PROPOSE_ORDER`, `PROPOSE_REROUTE(avoid=...)`, `ACCEPT`, `REJECT`,
`COMMIT`, `ABORT`; all counted by type and recorded in the trace. A proposal
that only adds an outgoing wait (others wait for me) does not alter my plan —
this is why most negotiations alter few agents. Ordering proposals that create
a cycle are rejected (deadlock handling).

### 9.3 Reassignment with POCL (R4)

Orphan goals (`delivered(o)`, plus `at_item(o, cell)` for a carried item);
candidates = `k_candidates` nearest operational agents within the radius
(widen if none); for each, reuse its existing POP as `P0` re-rooted at the
current state, add orphan goals as FINISH preconditions and call
`pocl(P0, lib, repair_pocl_max_nodes)`; ground new Gotos with hard ST-A*,
insert into the PEG, `retime`; pick the minimal-cost candidate. Altered agents
are the broken agent and the receivers. Emergency uses the same machinery.

### 9.4 Minimal-change guarantees

* The global planner and from-scratch POCL are never called in repair; R4
  starts from the receiver's existing POP.
* The executed prefix and all STARTED steps are frozen.
* A leg is rerouted only if waiting costs more than `delay_threshold`.
* Priority: emergency = 0, else lower id wins; a lower-priority agent must not
  force a higher-priority reroute when its own detour is under the threshold.

### 9.5 Altered-agent metrics (report all)

| Metric | Definition |
|---|---|
| `altered_plan` (primary) | step sequence/content changed, **or** a new cross-agent incoming edge not already implied (BFS on the pre-repair graph); includes the broken/emergency agent and reassignment receivers |
| `altered_path` | subset of `altered_plan` whose cell sequence / task list changed |
| `delayed_only` | times changed but signature identical |
| `altered_naive` | v1 comparison: any difference in remaining `(cell,t)` |

`RepairRecord = {event_type, t, A0_size, level_resolved, altered_plan_ids,
altered_path_ids, delayed_ids, naive_ids, added_delay_total, messages_by_type,
pocl_nodes, st_astar_calls, cpu_ms, success}`.

## 10. Simulator

```
build world/tasks; per agent: POCL -> POP; prioritized grounding -> PEG; retime
events = generate_events(cfg, peg)                 # identical for all modes
for t in range(T_max):
    apply events at t -> modify_plan(...)  (mode-dependent); postpone block if occupied
    start each agent's next unstarted step whose s.t == t; update pos/carrying/status
    VALIDATE: vertex, swap, blocked, shelf + peg.is_acyclic()
    record trace frame
    stop when all agents DONE
```

Modes: `negotiate` (R0-R4), `self_only` (R0-R1 + wait fallback), `global`
(re-run POCL for all remaining goals + prioritized ST-A* from current
positions — the quality upper bound), optional `timeindexed` (v1 ablation).
If `T_max` is hit, unfinished agents are counted as failures with completion
time `T_max` and flagged.

## 11. Visualization

Base grid animation (matplotlib, `figsize=(16,9)`, dark background): grid with
shelves/parking, interpolated agent motion, pickup/delivery markers, carried
item, dotted future paths (toggle `p`), red hatched blocked cells with
countdown, breakdown X, emergency ring, message arrows, metrics panel, event
log, cumulative line plot. Keys: space, arrows, `+/-`, `p`, `m`, `s`, and v2's
`g` (waiting arrows), `m` (message arrows), `c` (message bubbles), `d` (taskboard), `l` (log <->>
protocol dialogue), `n` (cycle agent), `o` (POP panel), `r` (replay phases),
`t` (Gantt).

v2 additions: waiting-for arrows (thin arrow to the `res` predecessor),
altered-category colours (altered_plan = thick yellow, delayed_only = thin
dashed orange), message arrows coloured by protocol type, **message bubbles**
(`viz.agent_message_labels`) — a rounded bubble over *both* endpoints of every
message visible this tick, the sender marked `»` and the receiver `«`, coloured
by `MSG_COLORS[kind]` and folded into `+N` when a tick is busy (`c` toggles
them), repair freeze-frames
for the five phases with a phase banner and rung label, a bottom **fleet
taskboard** (one cell per agent: phase, current task, target, delivery
progress), a **protocol-dialogue** panel (`viz.dialogue_lines`: the speech acts
with their `detail`, plus one repair line per disruption, so a locally-resolved
pass still shows its repair line), and plan-space views (`viz_plan.py`): POP
graph (networkx layered layout), Gantt before/after, 3-panel "repair cards" and
the message **sequence diagram** (`sequence_figure`).

Export: `render_gif` (PillowWriter), `render_mp4` (FFMpegWriter if present),
`screenshots` (t=0, before/at/after disruption, final), before/after panels,
and `run_demo.py --compare` (negotiate vs global side by side). `run_demo.py
--choke` runs the `scenarios.py` choke point — where the protocol is forced to
speak — instead of the (usually silent) random warehouse. Headless via
`matplotlib.use("Agg")`.

Trace frame schema: `{t, agents:[{id,pos,state,carrying_task_id,plan_future,
task:{task_id,phase,target,done,total},altered_flash,broken}],
blocked:[{cell,until}], tasks:[...], events:[...],
messages:[...], waiting_for:[(a,b)], categories:{id:cat}, repair_log:[...],
counters:{done_tasks,total_tasks,altered_so_far,messages_so_far,
disruptions_so_far}}`.

## 12. Experiments

Fixed 30x30 map, 3 tasks per agent. Sweep A: N in {5,10,20,30,40} at density
5% with 2 breakdowns + 2 emergencies. Sweep B: density in {0,2,5,10,15}% at
N=20. Grid sweep over all (N, density), >= 10 seeds, 3 modes, multiprocessing.

`outputs/results.csv` columns: `mode, N, density, seed, sum_of_costs, makespan,
mean_altered, median_altered, max_altered, success_rate, mean_messages,
mean_cpu_ms, n_disruptions, unfinished_agents` **plus** `mean/median/max
altered_plan, altered_path, delayed_only, altered_naive, rung_histogram,
pocl_nodes, st_astar_calls, messages_by_type`. `outputs/disruptions.csv` has
one row per disruption.

Plots: costs/makespan/altered vs N and vs density (error bars), success vs
density, messages vs N, heatmaps over (N, density), boxplot of altered per
disruption, grouped mode bar chart, altered_plan vs altered_naive vs global,
stacked rung histogram vs N and density, `comm_radius` ablation {3,6,9}, and
POCL nodes vs tasks per agent {1..6}.

## 13. Tests

* ST-A*: shortest path length = Manhattan in an empty grid.
* Prioritized plans: no vertex/swap conflicts for 20 random seeds, N=20.
* Validator raises on an injected collision.
* Swap in a 1-wide corridor with a side pocket resolves without conflicts.
* Reservation: `clashes`, `park` persistence, `remove_agent_after`.
* POCL: `is_valid_pop` on 20 random instances; tasks ordered by travel cost
  for <= 4 tasks; a forced `hand_empty` threat is resolved by ordering.
* PEG: initial PEG equals the ST-A* plan up to compression; a constructed swap
  yields a cycle; 20-seed runs collision-free; `retime` leaves DONE steps fixed.
* Modify: a detour blockage alters <= 1 agent (R0/R1); a corridor blockage
  needs R2 and alters >= 2; a permanent breakdown triggers R4 and orphan tasks
  complete; an emergency visits E and still finishes.
* **Repair never calls the global planner** (mock `global_plan`, assert 0).
* Same seed -> identical results twice.
* negotiate mean `altered_plan` <= global.
* Metrics: a delay-only chain yields `altered_plan=1, delayed_only=k`.

## 14. Build order

| M | Deliverable | Acceptance |
|---|---|---|
| M0 | venv, `requirements.txt`, `config.py` | clean install works |
| M1 | `world, reservation, planner` | planner tests (Section 13 bullets 1-2, 4-5) |
| M2 | `pocl.py` | `is_valid_pop` tests, print one POP |
| M3 | `peg.py` + grounding + `retime` + `sim.py` (no disruptions) | N<=40 finish, collision-free |
| M4 | base viz + Gantt | smooth GIF, headless |
| M5 | `disruptions.py` + R0/R1 + `metrics.py` | blockage repair card; <=1 altered |
| M6 | `negotiation.py` R2/R3 + message arrows + freeze-frames | corridor blockage -> R2, >=2 altered |
| M7 | R4 reassignment (permanent breakdown / emergency) via POCL | orphan tasks complete; emergency finishes |
| M8 | `baselines.py` | negotiate <= global on mean altered_plan |
| M9 | `experiments.py` (3-seed check, then full) | CSVs + plots |
| M10 | `make_report.py` + README + zip | report has POP fig, Gantt, 4 repair cards, altered table |

After every milestone: run `pytest` and a 20-seed stress run.

## 15. Report outline

Problem statement and assumptions; three-layer architecture + POCL mapping
table; L1 STRIPS + POCL pseudocode + example POP; L2/L3 grounding, PEG
construction, occupancy order, cycle = collision theorem, retiming;
disruptions as exogenous steps (Section 8 table); plan modification (5 phases,
ladder flowchart, message sequence diagram, reassignment, complexity); metrics
with a worked example; experimental setup; results tables/plots/heatmaps, 4
repair cards, altered-per-disruption table (mean, median, max under all three
definitions); discussion (effect of N and density, why partial orders reduce
altered agents, failure/deadlock cases, prioritized-planning incompleteness,
radius ablation) and future work (CBS-based repair, auction task allocation);
how to run + demo link; references (UCPOP 1992; Weld 1994; Kambhampati &
Hendler 1992; Fox et al. 2006; Silver 2005; Hoenig et al. MAPF-POST 2016).

## 16. Reproducibility notes

* One seeded RNG stream; events are generated from the undisrupted plan so all
  modes see identical events.
* Sort every set/dict iteration that feeds decisions and pin
  `PYTHONHASHSEED=0` so multiprocessing workers agree.
* `retime` must use `s.anchor is not None` (never `s.anchor or t0`).
