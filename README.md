# Multi-Agent Warehouse with Local Plan Repair

A staged multi-robot warehouse planner and simulator: POCL planning, a
dispatch/grounding layer (space-time A*, prioritized), and **local plan
repair** when the world changes.  The repair never calls the global planner —
it reasons about the *flaw* a disruption creates and patches the plans of the
agents involved, escalating through the rungs `R0`–`R4` of `SPEC.md` §9.1.

```
world.py ──▶ pocl.py ──▶ peg.py ──▶ planner.py ──▶ sim.py
                 │                       ▲
                 └── repair: modify.py ──┴── negotiation.py   (R0-R4)
                              │
                       disruptions.py / metrics.py
```

## Install & run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python run_demo.py                                  # plan + simulate 8 agents
python run_demo.py --agents 20 --tasks-per-agent 3  # bigger instance
python run_demo.py --disruptions 3 --accident 1 --emergency 2
python run_demo.py --disruptions 3 --accident 1 --emergency 2 --compare
python run_demo.py --compare --mode negotiate      # one mode only

PYTHONHASHSEED=0 python -m pytest -q               # full test suite
```

`--compare` runs every replanning mode on the *same* events and prints

| mode | meaning |
|---|---|
| `negotiate` | the algorithm: R0–R4, negotiation within `comm_radius` (`SPEC` §9.2) |
| `self_only` | only the disrupted agents are replanned (R0–R1 + wait fallback) |
| `bfs` | like `negotiate`, priority by BFS distance to the incident |
| `global` | quality upper bound: re-plan every remaining goal |
| `static` * | the v1 ablation: no repair at all (shown with `*`, not a failure) |

Columns: throughput/makespan, repair passes, the agents whose plan changed
(`altered_plan`, `SPEC` §9.5), the number of negotiation messages, and the
rungs the passes resolved at.

## Layout

| file | role |
|---|---|
| `config.py` | every tunable (`comm_radius`, `max_depth`, `lambda_soft`, `delay_threshold`, `beta_alter`, …) |
| `world.py` | grid, shelves, time-dependent blockages, task assignment |
| `reservation.py` | time-indexed vertex/edge/park reservations, `clashes` |
| `planner.py` | space-time A* (hard and soft), prioritized grounding |
| `pocl.py` | partial-order planner (L1): operators, threats, `pocl()` |
| `peg.py` | plan-execution graph: POP → time-indexed PlanSteps (`retime`) |
| `disruptions.py` | event types, injection, `postpone_blockages` |
| `modify.py` | the repair ladder `R0`–`R4` (`repair_schedule`) |
| `negotiation.py` | the message protocol of `SPEC` §9.2 used by rung `R2`/`R3` |
| `metrics.py` | `altered_plan` / `altered_path` / `delayed_only` / `altered_naive` |
| `sim.py` | executor + validator, `simulate_scenario`, `scan_violation` |
| `baselines.py` | the `global` baseline (the only global-planner caller) |
| `run_demo.py` | CLI demo / comparison table |
| `tests/` | 76 tests, including the stress-style multi-seed safety tests |

## Repair in one page

A disruption is a *plan flaw*: an interrupted move, a shelf that is now a wall,
a target that moved.  `modify.repair_schedule` freezes the executed prefix and
then re-derives the affected agents in priority order (emergency, then the
disrupted agents, then the neighbourhood):

* `R0` waiting only — the same route, the same task order;
* `R1`/`R2` the legs of the existing order are re-grounded with hard ST-A*;
  the rung is *labelled* from the candidate (`metrics.classify` +
  `completion_delta`), so a detour within `delay_threshold` is `R1` and a
  longer one is `R2`;
* `R2`/`R3` **negotiation** (`negotiation.py`): the disrupted agent proposes a
  soft-ST-A* route through the cells its neighbours have booked, and every
  holder answers with the protocol of `SPEC` §9.2 —
  `PROPOSE_ORDER`/`PROPOSE_REROUTE`, `ACCEPT`, `REJECT`, then
  `COMMIT`/`ABORT`.  A holder that is replanned later just waits for the
  initiator; a holder that is already committed reroutes *its own* leg (priced
  by the caller, accepted while its delay stays under `2*delay_threshold`); a
  holder outside `comm_radius` is pulled into the replan set (rung `R3`,
  `+3` per rung, at most `max_depth` times).  A grant is taken only if the
  proposal is verifiably clash-free against every non-yielding holder *and*
  pays for the agents it alters: `saving > delay_threshold` and
  `saving > beta_alter * newly_altered` (`SPEC` §9.1/§9.4).  Every message is
  counted in `RepairRecord.messages_by_type`.
* `R4` orphan goals (item carrying, a permanent breakdown) are handed to the
  `k_candidates` nearest agents, each re-planned with POCL **from its own
  current state** — never with the global planner.
* Fallback: agents that cannot be planned wait `failure_wait` ticks and are
  reported as failures; the schedule is still validated tick by tick.

Two invariants are enforced by tests: repair must never call the global
planner (`baselines.GLOBAL_PLAN_CALLS == 0`, `planner.prioritized_ground`
mocked to raise), and every reactive mode must end conflict-free with all
tasks delivered (a 20-seed stress run, plus the `simulate_scenario` validator:
vertex clash, swap, shelf, blockage, park).

## Disruption model

`disruptions.py` provides blockages (time windows written into the world),
accidents (temporary holds), permanent breakdowns (R4 hand-off) and
emergencies (a mandatory marshalling detour + hold).  `simulate_scenario`
replays the events in **tick order**: each pass only sees the events announced
at or before its tick, a blockage is postponed while it would trap an agent
inside an already-executed step, and residual safety violations are swept per
time window.  An accident hold is a *constraint* on the new plan, never baked
into the frozen prefix.

## Reproducibility

One seeded RNG stream; events are generated from the undisrupted plan so all
modes see identical events.  Run with `PYTHONHASHSEED=0` (the test suite
assumes it) and keep set/dict iteration sorted where it feeds decisions.

