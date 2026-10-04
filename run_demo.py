#!/usr/bin/env python3
"""End-to-end demo: build a world, plan with the v2 pipeline, simulate.

Usage::

    python run_demo.py                 # 8 agents x 2 tasks, seed 0
    python run_demo.py --agents 20 --tasks-per-agent 4 --seed 3 --print-plan
    python run_demo.py --disruptions 3 --accident 1 --emergency 2
    python run_demo.py --compare       # every replanning mode, same events
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from config import Config
from peg import peg_solve
from pocl import pop_summary, pocl
from sim import run_sim, simulate_scenario
from world import assign_tasks, build_world

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
    args = ap.parse_args()

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
        sim, records, _ = simulate_scenario(
            w2, cold, per_agent, p2, mode=mode, blockages=blockages,
            accidents=accidents, emergencies=emergencies,
            total_tasks=cfg.n_tasks, comm_radius=cfg.comm_radius,
            delay_threshold=cfg.delay_threshold, lambda_soft=cfg.lambda_soft,
            max_depth=cfg.max_depth, beta_alter=cfg.beta_alter)
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
    if worst == 0 and any(m in ("static", "timeindexed") for m in modes):
        print("* ablation mode: repairs are disabled by design")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
