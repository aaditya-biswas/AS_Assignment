# Multi-Agent Warehouse with Local Plan Repair

A staged multi-robot warehouse planner and simulator: POCL planning, a
dispatch/grounding layer (space-time A*, prioritized), and **local plan
repair** when the world changes.  The repair never calls the global planner —
it reasons about the *flaw* a disruption creates and patches the plans of the
agents involved, escalating through the rungs `R0`–`R4` of `SPEC.md` §9.1.

```
world.py ──▶ pocl.py ──▶ peg.py ──▶ planner.py ──▶ sim.py ──▶ viz.py / viz_plan.py
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
python run_demo.py --compare --viz --outdir outputs  # + the figures (SPEC §11)
python run_demo.py --choke --compare --viz         # corridor choke: the protocol speaks

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

`--choke` swaps the random warehouse for a hand-built **choke point**
(`scenarios.py`): a walled corridor with a single door, one agent parked on it
and its neighbours locked out. There a repair *cannot* be a silent local detour,
so the protocol really speaks and the demo shows `PROPOSE_REROUTE → ACCEPT →
COMMIT` (rung `R2`) with the message bubbles on the grid. `--choke-agents N`
sets the fleet size (2–6 stay conflict-free); the other disruption flags do not
apply because the showcase injects its own outage on both halves of the clash.

## Layout

| file | role |
|---|---|
| `config.py` | every tunable (`comm_radius`, `max_depth`, `lambda_soft`, `delay_threshold`, `beta_alter`, …) |
| `world.py` | grid, shelves, time-dependent blockages, task assignment |
| `scenarios.py` | hand-built choke points that force negotiation (the `--choke` showcase) |
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
| `viz.py` | trace frames, grid animation + GIF/MP4/snapshots, metrics panel, Gantt |
| `viz_plan.py` | plan-space views: POP graph, before/after Gantt, repair cards, ladder |
| `run_demo.py` | CLI demo / comparison table (+ `--viz` figures) |
| `tests/` | 85 tests, including the stress-style multi-seed safety tests |

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

## Visualisation

`viz.py` builds the trace frames of `SPEC` §11 from a *finished* schedule —
`viz.build_frames(world, plans, tasks, parking, prev=…, events=…, records=…,
log=…, traces=…)` returns one plain `dict` per tick with the SPEC fields
(`agents`, `blocked`, `tasks`, `events`, `messages`, `waiting_for`,
`categories`, `repair_log`, `counters`).  Every agent also carries a `task`
record (`task_id`, `phase` ∈ {`TO_PICKUP`, `TO_DELIVER`, `TO_PARK`, `IDLE`,
`DONE`}, `target`, `done`/`total`) so the panel can show *what each robot is
doing*.  The panel is the grid (shelves, interpolated agents, carried-item
marker, dotted remaining paths, red hatched blockages with countdown, breakdown
`X`, waiting arrows, negotiation arrows coloured by protocol type) plus a
metrics panel, the log panel and the cumulative line plot — and a bottom
**fleet taskboard**, one cell per agent (phase, current task, target, delivery
progress; wrapped over two rows past 20 agents).

```python
frames = viz.build_frames(world, final, tasks, parking, prev=cold,
                          events=events, records=records, log=log,
                          traces=sim.traces)          # record_traces=True
viz.render_gif(frames, world, "outputs/animation.gif", fps=8)
viz.render_mp4(frames, world, "outputs/animation.mp4", fps=8)   # needs ffmpeg
viz.screenshots(frames, world, "outputs/shots")      # t=0, around each event
viz.gantt(final, categories=frames[-1]["categories"], t_freeze=10)
```

`viz_plan.py` adds the plan-space views: `pop_figure` (layered networkx DAG of a
POP — solid `order` edges, dashed causal links), `gantt_before_after`,
`repair_card` / `repair_cards` (three panels: the `RepairRecord` — rung, altered
sets, message counts, budgets — and the routes before/after), `ladder_figure`
(the `R0`–`R4` flowchart with the rungs a run actually used lit up) and
`phase_frames` (the five freeze-frames `diagnose → unrefine → refine →
negotiate → commit`, each with a phase banner and the resolving rung) and
`sequence_figure` (the message-sequence diagram of a pass: one lifeline per
participant, one arrow per `PROPOSE`/`ACCEPT`/`REJECT`/`COMMIT` annotated with
the message `detail`, coloured by message type; a pass that resolved locally
draws the honest "no negotiation needed" note instead).

`viz.plot_overview(..., show_dialogue=True)` swaps the event log for the
**protocol dialogue** (`viz.dialogue_lines`): one `repair` line per disruption
(rung, altered set, message counts) plus one line per speech act with its
human-readable detail — so a silent run still shows *why* it stayed silent.
`show_taskbar=False` (and `show_waiting`/`show_messages`/`show_future`/
`show_message_labels`) hides each strip; `viz.interactive(frames, world)` gives
keyboard playback (`space`, arrows, `p`, `g`, `m`, `c`, `d`, `l`, `s`). Every
agent that *sends or receives* a protocol message this tick also gets a rounded
**speech bubble** over its disc (`viz.agent_message_labels`): the sender is
marked `»`, the receiver `«`, the bubble is coloured by
`viz.MSG_COLORS[kind]`, and a busy tick folds the surplus into a `+N` bubble.
The bubbles are drawn over *both* endpoints, so the initiator and the holder of
a grant are both visible in a still frame.

Everything is headless — `matplotlib.use("Agg")` on import of either module —
and `python run_demo.py --viz --outdir outputs` writes the whole set into
`outputs/<mode>/` (`animation.gif`, `shot_*.png`, `gantt.png`,
`final_panel.png`, `repair_card_*.png`, `phase_*.png`, `ladder.png`,
`sequence.png`, `dialogue_panel.png`, `pop.png`).
`simulate_scenario(..., record_traces=True)` hands the per-tick cell track to
`build_frames` so nothing is derived twice; without `--viz` no traces are
recorded.
Negotiation arrows and bubbles appear whenever the protocol actually speaks:
with an open grid the initiator usually detours for less than `lambda_soft`, so
proposals are only needed when a *holder parks in the corridor*. That is exactly
what `scenarios.py` builds — `parked_holder_scenario` (the case asserted by
`tests/test_negotiation.py`) and the `--choke` showcase — and why the plain demo
honestly reports `0 messages`.

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

