#!/usr/bin/env python3
"""End-to-end demo: build a world, plan with the v2 pipeline, simulate.

Usage::

    python run_demo.py                 # 8 agents x 2 tasks, seed 0
    python run_demo.py --agents 20 --tasks-per-agent 4 --seed 3 --print-plan
    python run_demo.py --disruptions 3 --accident 1 --emergency 2
    python run_demo.py --compare       # every replanning mode, same events
    python run_demo.py --choke --compare --viz   # corridor choke: the protocol speaks
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from config import Config
from negotiation import Negotiation
from peg import peg_solve
from pocl import pop_summary, pocl
from sim import run_sim, simulate_scenario
from world import World, assign_tasks, build_world

#: The four replanning modes of ``SPEC.md`` section 11 (+ the v1 ablation).
MODES = ("negotiate", "self_only", "bfs", "global", "static")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--tasks-per-agent", type=int, default=2)
    ap.add_argument("--height", type=int, default=28)
    ap.add_argument("--width", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=20_000)
    ap.add_argument("--print-plan", action="store_true",
                    help="print one agent's POCL plan and atomic steps")
    # ---- dynamic disruptions (M4) -------------------------------------
    ap.add_argument("--mode", default="negotiate", choices=MODES,
                    help="repair strategy used for the disruption demo")
    ap.add_argument("--disruptions", type=int, default=0,
                    help="number of random cell blockages to inject")
    ap.add_argument("--accident", type=int, default=-1,
                    help="agent id that breaks down (temporary hold)")
    ap.add_argument("--emergency", type=int, default=-1,
                    help="agent id that must divert to a marshalling cell")
    ap.add_argument("--compare", action="store_true",
                    help="run *every* mode on the same events and tabulate")
    ap.add_argument("--viz", action="store_true",
                    help="write the figures (GIF/MP4, snapshots, Gantt, "
                         "repair cards, ladder, POP) into --outdir")
    ap.add_argument("--outdir", default="outputs",
                    help="directory for --viz artefacts (default: outputs)")
    ap.add_argument("--fps", type=int, default=None,
                    help="frames per second of the GIF/MP4 export (default: "
                         "viz.ANIM_FPS = 4, i.e. 0.25 s per tick, so the "
                         "protocol dialogue can be read while it plays)")
    ap.add_argument("--choke", action="store_true",
                    help="run the corridor choke-point showcase instead of the "
                         "random warehouse: the single door forces the agents "
                         "to negotiate, so the demo shows protocol messages "
                         "(bubbles over both agents, arrows, sequence.png)")
    ap.add_argument("--choke-agents", type=int, default=4,
                    help="fleet size of the --choke showcase (2..6 are clean)")
    args = ap.parse_args()

    if args.choke:
        return _choke_demo(args)

    cfg = Config(H=args.height, W=args.width, n_agents=args.agents,
                 tasks_per_agent=args.tasks_per_agent, seed=args.seed)
    rng = np.random.default_rng(cfg.seed)
    world, tasks, parking = build_world(cfg, rng)
    per_agent = assign_tasks(tasks, parking, rng)

    print(f"world {cfg.H}x{cfg.W}  agents={cfg.n_agents}  "
          f"tasks={cfg.n_tasks}  shelves={int((world.grid == 1).sum())}  "
          f"connected={world.connected()}")

    t0 = time.perf_counter()
    plans, failed = peg_solve(world, per_agent, parking, max_nodes=args.max_nodes)
    plan_time = time.perf_counter() - t0

    res = run_sim(world, plans, total_tasks=cfg.n_tasks)
    print(f"planned in {plan_time:.2f}s   makespan={res.makespan}   "
          f"completed={res.completed}/{res.total_tasks}   "
          f"FAILED_INIT={res.failed_init}  FAILED_ACT={res.failed_act}")
    print(f"throughput={res.throughput:.2%}   "
          f"mean service time={res.mean_service:.1f}   "
          f"conflicts={len(res.violations)}")
    if failed:
        print(f"agents that could not be planned: {sorted(failed)}")
    if res.violations:
        for v in res.violations[:5]:
            print("  VIOLATION", v)

    if args.print_plan:
        aid = 0
        pop = pocl(aid, list(per_agent[aid]), parking[aid], world,
                   max_nodes=args.max_nodes)
        if pop is not None:
            print("\n" + pop_summary(pop))
        print(f"\natomic steps for agent {aid}:")
        for s in (plans.get(aid) or []):
            print(f"  t={s.t:3d} {s.cell} {s.action}"
                  + (f" item={s.item}" if s.item is not None else ""))

    disrupted = (args.disruptions or args.accident >= 0
                 or args.emergency >= 0 or args.compare)
    if disrupted:
        return _disruption_demo(args, cfg, world, per_agent, parking, plans,
                                res)
    return 0 if (res.conflict_free and res.failed_init == 0) else 1


def _disruption_demo(args, cfg, world, per_agent, parking, plans, base) -> int:
    """Run the M4 disruption scenario for one or for all repair modes."""
    from disruptions import (AccidentEvent, EmergencyEvent, postpone_blockages,
                             random_blockages)

    rng = np.random.default_rng(cfg.seed + 1)
    horizon = max(20, base.makespan)
    blockages = random_blockages(world, args.disruptions, rng,
                                 t_max=max(4, horizon // 2), avoid=parking)
    # SPEC 10: a blockage is never announced while an agent occupies its cell
    blockages = postpone_blockages(plans, blockages)
    accidents = ([AccidentEvent(args.accident, max(2, horizon // 6), 6)]
                 if args.accident >= 0 else [])
    emergencies = ([EmergencyEvent(args.emergency, max(4, horizon // 4), 4)]
                   if args.emergency >= 0 else [])

    print(f"\n-- disruptions: {len(blockages)} blockage(s), "
          f"{len(accidents)} accident(s), {len(emergencies)} emergency(ies) --")
    for ev in blockages:
        print(f"   block {ev.cell} t={ev.t0}..{ev.t1}")
    for ev in accidents:
        print(f"   accident agent {ev.agent} at t={ev.at_t} for {ev.duration}")
    for ev in emergencies:
        print(f"   emergency agent {ev.agent} at t={ev.at_t} hold={ev.hold}")

    modes = MODES if args.compare else (args.mode,)
    print(f"\n{'mode':10s} {'status':9s} {'done':>9s} {'makespan':>8s} "
          f"{'repairs':>7s} {'altered':>7s} {'msgs':>5s}  rungs")
    worst = 0
    for mode in modes:
        # a fresh world per mode: blockages mutate the world in place
        w2, _, p2 = build_world(cfg, np.random.default_rng(cfg.seed))
        cold = {a: (list(s) if s else None) for a, s in plans.items()}
        log = Negotiation(t=0)
        sim, records, final = simulate_scenario(
            w2, cold, per_agent, p2, mode=mode, blockages=blockages,
            accidents=accidents, emergencies=emergencies,
            total_tasks=cfg.n_tasks, comm_radius=cfg.comm_radius,
            delay_threshold=cfg.delay_threshold, lambda_soft=cfg.lambda_soft,
            max_depth=cfg.max_depth, beta_alter=cfg.beta_alter,
            message_log=log, record_traces=args.viz)
        altered = sorted({a for r in records for a in r.altered_plan_ids})
        rungs = ",".join(r.level_resolved for r in records) or "-"
        msgs = sum(sum(r.messages_by_type.values()) for r in records)
        ablation = mode in ("static", "timeindexed")
        status = "OK" if sim.conflict_free else f"{len(sim.violations)} viol"
        if not ablation:      # the v1 ablation never repairs -- not a failure
            worst = max(worst, len(sim.violations))
        print(f"{mode:10s} {status + ('*' if ablation else ''):9s} "
              f"{sim.completed:3d}/{sim.total_tasks:<5d} {sim.makespan:8d} "
              f"{len(records):7d} {len(altered):7d} {msgs:5d}  {rungs}")
        if args.viz:
            _write_figures(args, w2, cold, final, per_agent, p2, records,
                           list(blockages) + list(accidents)
                           + list(emergencies), log, sim, mode)
    if worst == 0 and any(m in ("static", "timeindexed") for m in modes):
        print("* ablation mode: repairs are disabled by design")
    return 0 if worst == 0 else 1


def _choke_demo(args) -> int:
    """The corridor choke-point showcase (``--choke``): the protocol *speaks*.

    A walled corridor with a single door, one agent parked on it and its
    neighbours locked out, so the ladder finds no hard candidate and the
    initiators negotiate a grant.  Unlike the random warehouse (where a repair
    is a silent local detour), this instance produces ``PROPOSE_REROUTE``/
    ``ACCEPT``/``COMMIT`` -- so the GIF, ``sequence.png`` and the on-grid
    bubbles all show the protocol.
    """
    from scenarios import choke_showcase

    sc = choke_showcase(args.choke_agents, seed=args.seed)
    world, plans = sc["world"], sc["plans"]
    per_agent, parking = sc["per_agent"], sc["parking"]
    cfg = Config(H=world.H, W=world.W, n_agents=sc["n_agents"],
                 tasks_per_agent=1, seed=args.seed)
    print(f"choke showcase: {sc['n_agents']} agents on a {world.H}x{world.W} "
          f"corridor, single door {sc['door']}, {sc['total_tasks']} task(s)")
    print("   (--disruptions/--accident/--emergency do not apply: the showcase "
          "injects its own outage on both halves of the clash)")

    modes = MODES if args.compare else (args.mode,)
    print(f"\n{'mode':10s} {'status':9s} {'done':>9s} {'makespan':>8s} "
          f"{'repairs':>7s} {'altered':>7s} {'msgs':>5s}  rungs")
    worst = 0
    for mode in modes:
        # a fresh world per mode: the events mutate the world in place
        w2 = World(world.grid.copy())
        cold = {a: (list(s) if s else None) for a, s in plans.items()}
        log = Negotiation(t=0)
        sim, records, final = simulate_scenario(
            w2, cold, per_agent, parking, mode=mode,
            blockages=sc["blockages"], accidents=sc["accidents"],
            emergencies=sc["emergencies"], total_tasks=sc["total_tasks"],
            comm_radius=cfg.comm_radius, delay_threshold=cfg.delay_threshold,
            lambda_soft=cfg.lambda_soft, max_depth=cfg.max_depth,
            beta_alter=cfg.beta_alter, message_log=log, record_traces=args.viz)
        altered = sorted({a for r in records for a in r.altered_plan_ids})
        rungs = ",".join(r.level_resolved for r in records) or "-"
        msgs = sum(sum(r.messages_by_type.values()) for r in records)
        ablation = mode in ("static", "timeindexed")
        status = "OK" if sim.conflict_free else f"{len(sim.violations)} viol"
        if not ablation:      # the v1 ablation never repairs -- not a failure
            worst = max(worst, len(sim.violations))
        print(f"{mode:10s} {status + ('*' if ablation else ''):9s} "
              f"{sim.completed:3d}/{sim.total_tasks:<5d} {sim.makespan:8d} "
              f"{len(records):7d} {len(altered):7d} {msgs:5d}  {rungs}")
        if args.viz:
            _write_figures(args, w2, cold, final, per_agent, parking, records,
                           list(sc["blockages"]) + list(sc["accidents"])
                           + list(sc["emergencies"]), log, sim, mode)
    if worst == 0 and any(m in ("static", "timeindexed") for m in modes):
        print("* ablation mode: repairs are disabled by design")
    return 0 if worst == 0 else 1



def _write_figures(args, world, before, after, per_agent, parking, records,
                   events, log, sim, mode) -> None:
    """The SPEC 11 artefacts of one mode (GIF/MP4, snapshots, cards, POP)."""
    from matplotlib import pyplot as plt

    import viz
    import viz_plan

    out = os.path.join(args.outdir, mode)
    os.makedirs(out, exist_ok=True)
    fps = int(args.fps or viz.ANIM_FPS)
    tasks = [tk for lst in per_agent for tk in lst]
    disrupted = sorted({a for r in records for a in r.altered_plan_ids})
    frames = viz.build_frames(world, after, tasks, parking, prev=before,
                              t_freeze=records[0].t if records else 0,
                              disrupted=disrupted, events=events,
                              records=records, log=log, traces=sim.traces)
    gif = viz.render_gif(frames, world, os.path.join(out, "animation.gif"),
                         fps=fps, max_frames=200)
    shots = viz.screenshots(frames, world, out, prefix="shot")
    fig = viz.gantt(after, categories=frames[-1]["categories"],
                    t_freeze=records[0].t if records else None)
    fig.savefig(os.path.join(out, "gantt.png"), facecolor=viz.BG)
    viz.plot_overview(world, frames, frames[-1]["t"]).savefig(
        os.path.join(out, "final_panel.png"), facecolor=viz.BG)
    if records:
        viz_plan.repair_cards(records, world=world, tasks=tasks,
                              parking=parking, prev=before, new=after,
                              outdir=out, max_cards=4)
        viz_plan.phase_frames(records[0], world=world, frames=frames,
                              t=records[0].t, outdir=out)
    used = [r.level_resolved for r in records]
    viz_plan.ladder_figure(used).savefig(os.path.join(out, "ladder.png"),
                                         facecolor=viz.BG)
    viz_plan.sequence_figure(log, title=f"{mode}: negotiation sequence").savefig(
        os.path.join(out, "sequence.png"), facecolor=viz.BG)
    viz.plot_overview(world, frames, records[0].t if records else 0,
                      show_dialogue=True).savefig(
        os.path.join(out, "dialogue_panel.png"), facecolor=viz.BG)
    aid = disrupted[0] if disrupted else 0
    pop = pocl(aid, list(per_agent[aid]), parking[aid], world,
               max_nodes=args.max_nodes)
    if pop is not None:
        viz_plan.pop_panel(world, pop, agent=aid).savefig(
            os.path.join(out, "pop.png"), facecolor=viz.BG)
    print(f"   figures -> {out}  ({len(shots)} snapshots, gif + gantt + cards)")
    plt.close("all")
    return gif


if __name__ == "__main__":
    raise SystemExit(main())
