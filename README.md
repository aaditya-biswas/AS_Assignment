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
python experiments.py --quick                      # small SPEC §12 sweep
python make_report.py --data outputs --zip         # PDF + Markdown + zip bundle

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
| `run_demo.py` | CLI demo / comparison table (+ `--viz`, `--choke`) |
| `experiments.py` | the `SPEC` §12 sweeps -> `results.csv`, `disruptions.csv`, figures |
| `make_report.py` | the `SPEC` §15 report (PDF + Markdown) and the zip bundle |
| `tests/` | 110+ tests, including the stress-style multi-seed safety tests |

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

### What the symbols mean

The panel explains itself: every export (GIF, MP4, screenshots, before/after
view, the five freeze-frames and the repair cards) carries the on-grid legend
built by `viz.grid_symbol_handles()` from the single vocabulary
`viz.GRID_SYMBOLS` — `k` toggles it, `show_symbol_legend=False` hides it.

| glyph | meaning |
| --- | --- |
| grey square | **shelf** (static obstacle; `0`/`1` of `World.grid`) |
| coloured disc + id | **agent**; the ring colour is its altered category (`altered_plan` thick yellow, `altered_path` orange, `delayed_only` dotted, `unchanged` grey) |
| small white square | agent **carrying a tote** (`carrying_task_id`) |
| red `X` | **breakdown** (permanent, the R4 hand-off) |
| hollow cyan **diamond** `D` | **pickup cell** of a task (`viz.PICKUP_C`) |
| star `*` | **delivery cell** of a task, filled by status from `viz.TASK_STATUS_C`: grey `PENDING`, amber `PICKED`, green `DONE` |
| red hatched square + number | dynamic **blockage**, the number is the **ticks left** |
| pale solid arrow | **waiting for** — who is blocked by whom (`waiting_for`) |
| dashed coloured arrow | **protocol message** (`PROPOSE_ORDER`, `PROPOSE_REROUTE`, `ACCEPT`, `REJECT`, `COMMIT`, `ABORT`; colour = `viz.MSG_COLORS[kind]`) |
| rounded bubble over an agent | the **speech act** itself, sender `»` / receiver `«`, surplus folded into `+N` |

The stars and diamonds are *task* markers, so a grid can show several of each:
the diamond is where a tote must be picked up, the star where it must be
delivered, and the star's colour is the live status of that task.

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
`show_message_labels`/`show_symbol_legend`) hides each strip;
`viz.interactive(frames, world)` gives keyboard playback (`space`, arrows, `p`,
`g`, `m`, `c`, `d`, `k`, `l`, `s`). Every
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

## Experiments and report

`experiments.py` runs the `SPEC` §12 sweeps.  The unit of work is a *family*
(one `(N, density, seed)` instance); the world and the undisrupted plan are built
**once** per family and then every mode sees the *identical* event list, so a
`results.csv` row differs only by `mode`.  Families are independent, so
`--jobs N` fans them out over a `multiprocessing.Pool`; all writes are sorted by
`(mode, N, density, seed)`, so `--jobs 1` and `--jobs 8` produce byte-identical
CSVs (a test asserts exactly that).

* **Sweep A** — `N ∈ {5,10,20,30,40}` at 5% density; **sweep B** —
  `density ∈ {0,2,5,10,15}%` at `N = 20`; **grid** — every combination;
  the default is 10 seeds × 3 modes (`negotiate`, `self_only`, `global`).
  `--quick` is a two-seed corner that finishes in seconds.
* `density` scales the number of *dynamic blockages*
  (`Config.n_blockages`: `round(density × free cells)`), on top of the
  `n_breakdowns` accidents and `n_emergencies` emergencies.
* `results.csv` has the full `SPEC` §12 column list (the four `altered_*`
  mean/median/max triplets, `rung_histogram`, `messages_by_type`,
  `pocl_nodes`, `st_astar_calls`, `success_rate`, `unfinished_agents`, …);
  `disruptions.csv` has one row per repair pass.
* `--plots` writes the trend curves (cost/makespan/altered/messages vs `N` and
  vs density, with error bars), success-vs-density, the `(N, density)` heatmaps,
  the per-disruption boxplot, the grouped mode bars, the stacked rung
  histograms and `altered_plan` vs `altered_naive`.
* `--ablations` adds the two extra `SPEC` §12 studies:
  `radius_ablation.csv` + `comm_radius_ablation.png` (`comm_radius ∈ {3,6,9}`)
  and `pocl_nodes.csv` + `pocl_nodes_vs_tasks.png` (POP action nodes and
  grounding nodes vs tasks per agent).
* `run_meta.json` records the command, the seed/`N`/density lists, the package
  versions, the git commit and `PYTHONHASHSEED`, so a run can be audited — and
  the documentation is honest about wall-clock (`mean_cpu_ms` is the only
  non-reproducible column).

`make_report.py` builds the `SPEC` §15 report from a **fresh** run (so it always
works) and folds in the sweep artefacts when `--data outputs` holds a
`results.csv`.  Case study **A** is a random warehouse with blockages placed *on
the planned routes* (`scenarios.blockages_on_routes`, so each one really bites),
two accidents and an emergency; case study **B** is the corridor choke point
where the protocol must speak.  The bundle contains the POP figure, the Gantt
chart, **four repair cards**, the freeze-frames, the ladder, the message
sequence diagram and the altered table (mean/median/max under all four `SPEC`
§9.5 definitions) — as `report.pdf` (reportlab), `report.md` (same content,
GitHub-friendly) and `report_bundle.zip` (self-contained: figures included).

## Disruption model
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

