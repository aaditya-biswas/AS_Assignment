"""Experiment sweeps (``SPEC.md`` section 12).

The unit of work is a *family*: one ``(N, density, seed)`` instance plus the
events injected into it.  A family builds the world **once**, derives the
undisrupted plan, then injects identical events for every mode (``SPEC`` 16: all
modes must see the same disruptions), so a row of ``results.csv`` differs only
by ``mode``.  Families are independent, so they run in a
:class:`multiprocessing.Pool`; every write is sorted by ``(mode, N, density,
seed)`` so a run is reproducible even with ``--jobs > 1``.

Artefacts, all under ``--outdir``:

* ``results.csv``     -- one row per ``(mode, N, density, seed)`` (SPEC 12);
* ``disruptions.csv`` -- one row per disruption (repair record);
* ``run_meta.json``   -- config, versions, seed list, git commit, timing;
* ``*.png``           -- the SPEC 12 plots (error bars, heatmaps, rung stacks).

Usage::

    python experiments.py --quick                 # tiny, seconds
    python experiments.py --sweep A --seeds 10 --jobs "$(nproc)"
    python experiments.py --sweep grid --seeds 10 --plots --outdir outputs
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import Config
from disruptions import (AccidentEvent, EmergencyEvent, postpone_blockages,
                         random_blockages)
from negotiation import Negotiation
from peg import peg_solve
from sim import plan_makespan, simulate_scenario
from world import World, assign_tasks, build_world

#: the deterministic column order of ``results.csv`` (``SPEC`` 12)
RESULT_COLUMNS = (
    "mode", "N", "density", "seed", "sum_of_costs", "makespan",
    "mean_altered", "median_altered", "max_altered", "success_rate",
    "mean_messages", "mean_cpu_ms", "n_disruptions", "unfinished_agents",
    "mean_altered_plan", "median_altered_plan", "max_altered_plan",
    "mean_altered_path", "median_altered_path", "max_altered_path",
    "mean_delayed_only", "median_delayed_only", "max_delayed_only",
    "mean_altered_naive", "median_altered_naive", "max_altered_naive",
    "rung_histogram", "pocl_nodes", "st_astar_calls", "messages_by_type",
    "conflict_free", "failed_init",
)

#: one row per disruption in ``disruptions.csv``
DISRUPTION_COLUMNS = (
    "mode", "N", "density", "seed", "t", "event_type", "level_resolved",
    "n_altered_plan", "n_altered_path", "n_delayed_only", "n_altered_naive",
    "added_delay_total", "messages", "pocl_nodes", "st_astar_calls", "cpu_ms",
    "success",
)

#: the abbreviated column every metric of ``SPEC`` 9.5 fills in
_METRIC_KEYS = ("altered_plan", "altered_path", "delayed_only", "altered_naive")


# ----------------------------------------------------------------------
# the pieces of a SPEC 12 sweep
# ----------------------------------------------------------------------
def sweep_jobs(name: str, seeds: int, *, quick: bool = False,
               tasks_per_agent: int = 3) -> List[dict]:
    """The ``(N, density, seed)`` families of a named sweep."""
    if quick:
        ns, dens = [5, 10], [0.0, 0.05]
    elif name == "A":                       # sweep A: N at 5% density
        ns, dens = [5, 10, 20, 30, 40], [0.05]
    elif name == "B":                       # sweep B: density at N=20
        ns, dens = [20], [0.0, 0.02, 0.05, 0.10, 0.15]
    else:                                   # the full (N, density) grid
        ns = [5, 10, 20, 30, 40]
        dens = [0.0, 0.02, 0.05, 0.10, 0.15]
    return [{"N": n, "density": d, "seed": s, "sweep": name,
             "tasks_per_agent": tasks_per_agent}
            for n in ns for d in dens for s in range(seeds)]


def build_instance(job: dict, rng: np.random.Generator):
    """World + tasks + undisrupted plans of one family (``SPEC`` 16 step 1).

    Returns ``(cfg, world, tasks, parking, per_agent, plans, meta)``; ``plans``
    holds ``None`` for agents that could not be grounded (``FAILED_INIT``).
    """
    cfg = Config(H=job.get("H", 30), W=job.get("W", 30), n_agents=job["N"],
                 tasks_per_agent=job["tasks_per_agent"], seed=job["seed"],
                 obstacle_density=job["density"],
                 n_breakdowns=job.get("n_breakdowns", 2),
                 n_emergencies=job.get("n_emergencies", 2))
    world, tasks, parking = build_world(cfg, rng)
    per_agent = assign_tasks(tasks, parking, rng)
    plans, failed = peg_solve(world, per_agent, parking,
                              max_nodes=cfg.pocl_max_nodes)
    meta = {"failed_init": len(failed), "n_tasks": cfg.n_tasks,
            "n_blockages": cfg.n_blockages(len(world.free_cells()))}
    return cfg, world, tasks, parking, per_agent, plans, meta


def make_events(cfg: Config, world: World, plans: dict, parking: Sequence,
                rng: np.random.Generator):
    """The exogenous blockages + endogenous accidents/emergencies (SPEC 10).

    Derived from the *undisrupted* plan, so every mode of the family sees the
    identical event list.  ``density`` scales the number of blocked cells
    (``Config.n_blockages``); the accidents and emergencies come from
    ``n_breakdowns`` / ``n_emergencies``.
    """
    horizon = max(20, plan_makespan(plans))
    n_block = cfg.n_blockages(len(world.free_cells()))
    blockages = random_blockages(world, n_block, rng,
                                 t_max=max(4, horizon // 2), avoid=parking)
    blockages = postpone_blockages(plans, blockages)

    picked = rng.permutation(cfg.n_agents)
    accidents = [AccidentEvent(int(picked[i]),
                               int(rng.integers(2, max(3, horizon // 3))),
                               int(rng.integers(*cfg.breakdown_duration_range)),
                               permanent=bool(rng.random()
                                              < cfg.p_permanent_breakdown))
                 for i in range(min(cfg.n_breakdowns, cfg.n_agents))]
    n_acc = len(accidents)
    emergencies = [EmergencyEvent(int(picked[n_acc + i]),
                                  int(rng.integers(2, max(3, horizon // 4))),
                                  int(rng.integers(3, 6)))
                   for i in range(min(cfg.n_emergencies,
                                      max(0, cfg.n_agents - n_acc)))]
    return list(blockages), accidents, emergencies




# ----------------------------------------------------------------------
# one family -> rows of both CSVs
# ----------------------------------------------------------------------
#: the ``RepairRecord`` list field behind each SPEC 9.5 metric short name
_ID_FIELD = {"altered_plan": "altered_plan_ids",
             "altered_path": "altered_path_ids",
             "delayed_only": "delayed_ids",
             "altered_naive": "naive_ids"}


def _stats(values: Sequence[float]) -> Tuple[float, float, float]:
    """``(mean, median, max)`` of a possibly empty sample."""
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    return float(a.mean()), float(np.median(a)), float(a.max())


def summary_row(family: dict, mode: str, cfg: Config, sim, records,
                per_agent, meta: dict, cpu_ms: float) -> dict:
    """One ``results.csv`` row: the SPEC 12 columns for ``(mode, family)``."""
    counts = {k: [len(getattr(r, _ID_FIELD[k])) for r in records]
              for k in _METRIC_KEYS}
    rungs: Dict[str, int] = {}
    for r in records:
        rungs[r.level_resolved] = rungs.get(r.level_resolved, 0) + 1
    msgs: Dict[str, int] = {}
    for r in records:
        for k, v in r.messages_by_type.items():
            msgs[k] = msgs.get(k, 0) + v
    done = set(sim.completion_times)
    unfinished = sum(1 for a in range(cfg.n_agents)
                     if any(tk.id not in done for tk in per_agent[a]))
    mean_a, med_a, max_a = _stats(counts["altered_plan"])
    row = {
        "mode": mode, "N": family["N"], "density": family["density"],
        "seed": family["seed"], "sum_of_costs": int(sim.sum_service),
        "makespan": sim.makespan, "mean_altered": mean_a,
        "median_altered": med_a, "max_altered": max_a,
        "success_rate": (float(np.mean([r.success for r in records]))
                         if records else 1.0),
        "mean_messages": (sum(msgs.values()) / len(records)) if records else 0.0,
        "mean_cpu_ms": cpu_ms / len(records) if records else 0.0,
        "n_disruptions": len(records), "unfinished_agents": unfinished,
        "rung_histogram": json.dumps(rungs, sort_keys=True),
        "pocl_nodes": sum(r.pocl_nodes for r in records),
        "st_astar_calls": sum(r.st_astar_calls for r in records),
        "messages_by_type": json.dumps(msgs, sort_keys=True),
        "conflict_free": bool(sim.conflict_free),
        "failed_init": meta.get("failed_init", 0),
    }
    for k in _METRIC_KEYS:
        mean_v, med_v, max_v = _stats(counts[k])
        row[f"mean_{k}"], row[f"median_{k}"], row[f"max_{k}"] = \
            mean_v, med_v, max_v
    return row


def disruption_rows(family: dict, mode: str, records) -> List[dict]:
    """One ``disruptions.csv`` row per repair record of the family."""
    out = []
    for r in records:
        out.append({
            "mode": mode, "N": family["N"], "density": family["density"],
            "seed": family["seed"], "t": r.t, "event_type": r.event_type,
            "level_resolved": r.level_resolved,
            "n_altered_plan": len(r.altered_plan_ids),
            "n_altered_path": len(r.altered_path_ids),
            "n_delayed_only": len(r.delayed_ids),
            "n_altered_naive": len(r.naive_ids),
            "added_delay_total": r.added_delay_total,
            "messages": sum(r.messages_by_type.values()),
            "pocl_nodes": r.pocl_nodes, "st_astar_calls": r.st_astar_calls,
            "cpu_ms": r.cpu_ms, "success": bool(r.success),
        })
    return out


def run_family(family: dict, modes: Sequence[str]) -> Tuple[List[dict],
                                                           List[dict]]:
    """Run every ``mode`` on one ``(N, density, seed)`` instance."""
    rng = np.random.default_rng(family["seed"])
    cfg, world, tasks, parking, per_agent, plans, meta = \
        build_instance(family, rng)
    ev_rng = np.random.default_rng(family["seed"] + 1)
    blockages, accidents, emergencies = make_events(cfg, world, plans, parking,
                                                    ev_rng)
    max_repairs = max(8, 2 * (len(blockages) + len(accidents)
                              + len(emergencies)))
    rows: List[dict] = []
    drows: List[dict] = []
    for mode in modes:
        # a fresh world per mode: blockages mutate the world in place
        w2 = World(world.grid.copy())
        cold = {a: (list(s) if s else None) for a, s in plans.items()}
        log = Negotiation(t=0)
        t0 = time.perf_counter()
        sim, records, final = simulate_scenario(
            w2, cold, per_agent, parking, mode=mode,
            blockages=list(blockages), accidents=list(accidents),
            emergencies=list(emergencies), total_tasks=cfg.n_tasks,
            max_repairs=max_repairs,
            comm_radius=family.get("comm_radius", cfg.comm_radius),
            delay_threshold=cfg.delay_threshold, lambda_soft=cfg.lambda_soft,
            max_depth=cfg.max_depth, beta_alter=cfg.beta_alter,
            message_log=log)
        cpu_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(summary_row(family, mode, cfg, sim, records, per_agent,
                                meta, cpu_ms))
        drows.extend(disruption_rows(family, mode, records))
    return rows, drows


# ----------------------------------------------------------------------
# the parallel driver
# ----------------------------------------------------------------------
def _family_worker(payload):
    """Pool unit: ``(family, modes) -> (rows, disruption_rows)``."""
    family, modes = payload
    return run_family(family, modes)


def run_sweep(families: Sequence[dict], modes: Sequence[str], *,
              jobs: int = 1, verbose: bool = True
              ) -> Tuple[List[dict], List[dict], float]:
    """Run every family and return deterministic, sorted row lists.

    ``jobs > 1`` uses a :class:`multiprocessing.Pool`; the rows are re-sorted
    afterwards, so the CSVs do not depend on completion order.
    """
    jobs = max(1, int(jobs))
    total = len(families)
    rows: List[dict] = []
    drows: List[dict] = []
    t0 = time.perf_counter()
    if jobs == 1 or total <= 1:
        for i, fam in enumerate(families, 1):
            r, d = run_family(fam, modes)
            rows += r
            drows += d
            _progress(i, total, fam, verbose)
    else:
        payloads = [(fam, modes) for fam in families]
        with Pool(processes=min(jobs, total)) as pool:
            for i, (r, d) in enumerate(pool.imap_unordered(_family_worker,
                                                           payloads), 1):
                rows += r
                drows += d
                _progress(i, total, None, verbose)
    secs = time.perf_counter() - t0
    rows.sort(key=lambda r: (r["mode"], r["N"], r["density"], r["seed"]))
    drows.sort(key=lambda d: (d["mode"], d["N"], d["density"], d["seed"],
                              d["t"], d["event_type"]))
    return rows, drows, secs


def _progress(done: int, total: int, family: Optional[dict],
              verbose: bool) -> None:
    if not verbose:
        return
    where = f"N={family['N']} d={family['density']:.2f}" if family else ""
    end = "\n" if done == total else ""
    print(f"\r  families {done}/{total} {where:<18s}", end=end, file=sys.stderr)


# ----------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------
def _write_csv(rows: Sequence[dict], columns: Sequence[str], path: str) -> int:
    """Write ``rows`` with a fixed column order (blank when empty)."""
    import pandas as pd

    df = pd.DataFrame(list(rows), columns=None) if rows else pd.DataFrame()
    for col in columns:                     # guarantee the schema either way
        if col not in df.columns:
            df[col] = None
    df = df[list(columns)]
    df.to_csv(path, index=False)
    return len(df)


def _version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:                       # pragma: no cover - env dependent
        return "unknown"


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:                       # pragma: no cover - env dependent
        return "unknown"


def run_meta(args, families: Sequence[dict], modes: Sequence[str],
             secs: float, n_rows: int) -> dict:
    """Everything needed to reproduce (and audit) a sweep (``SPEC`` 16)."""
    return {
        "command": " ".join([sys.executable] + sys.argv),
        "sweep": getattr(args, "sweep", "?"), "quick": bool(getattr(args, "quick", False)),
        "modes": list(modes), "jobs": max(1, int(getattr(args, "jobs", 1))),
        "n_families": len(families), "n_rows": n_rows,
        "Ns": sorted({f["N"] for f in families}),
        "densities": sorted({f["density"] for f in families}),
        "seeds": sorted({f["seed"] for f in families}),
        "tasks_per_agent": (families[0]["tasks_per_agent"] if families else None),
        "wall_seconds": round(secs, 3),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in ("numpy", "matplotlib", "pandas")},
        "git_commit": _git_commit(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }


def write_outputs(outdir: str, rows: Sequence[dict],
                  drows: Sequence[dict], meta: dict) -> Dict[str, str]:
    """Write ``results.csv``, ``disruptions.csv`` and ``run_meta.json``."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "results": os.path.join(outdir, "results.csv"),
        "disruptions": os.path.join(outdir, "disruptions.csv"),
        "meta": os.path.join(outdir, "run_meta.json"),
    }
    _write_csv(rows, RESULT_COLUMNS, paths["results"])
    _write_csv(drows, DISRUPTION_COLUMNS, paths["disruptions"])
    with open(paths["meta"], "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return paths



# ----------------------------------------------------------------------
# plots (SPEC 12)
# ----------------------------------------------------------------------
def _fig():
    """Import matplotlib headless and return ``pyplot``."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    return plt


def _save(fig, outdir: str, name: str) -> str:
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _line_png(df, xcol: str, ycol: str, outdir: str, name: str,
              xlabel: str, ylabel: str, title: str) -> Optional[str]:
    """``ycol`` vs ``xcol`` per mode, mean over seeds with a std error bar."""
    plt = _fig()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    drawn = False
    for mode, g in df.groupby("mode"):
        stats = g.groupby(xcol)[ycol].agg(["mean", "std"]).sort_index()
        if stats.empty:
            continue
        err = stats["std"].fillna(0.0).to_numpy()
        ax.errorbar(stats.index.to_numpy(), stats["mean"].to_numpy(), yerr=err,
                    marker="o", capsize=3, lw=1.6, label=str(mode))
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(title="mode")
    return _save(fig, outdir, name)


def _heatmap_png(df, xcol: str, ycol: str, value: str, outdir: str, name: str,
                 xlabel: str, ylabel: str, title: str) -> Optional[str]:
    """Mean of ``value`` over the ``(xcol, ycol)`` grid."""
    plt = _fig()
    piv = df.pivot_table(index=ycol, columns=xcol, values=value, aggfunc="mean")
    if piv.empty:
        plt.close("all")
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(piv.to_numpy(), aspect="auto", origin="lower",
                   cmap="viridis")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value)
    return _save(fig, outdir, name)


def _rung_stack_png(df, xcol: str, outdir: str, name: str,
                    xlabel: str, title: str) -> Optional[str]:
    """Stacked histogram of the resolved rungs, summed over seeds."""
    import pandas as pd

    rows: List[dict] = []
    for _, r in df.iterrows():
        try:
            hist = json.loads(r["rung_histogram"] or "{}")
        except (TypeError, ValueError):
            continue
        for rung, n in hist.items():
            rows.append({"x": r[xcol], "rung": rung, "n": n})
    plt = _fig()
    if not rows:
        plt.close("all")
        return None
    frame = (pd.DataFrame(rows).groupby(["x", "rung"])["n"].sum()
             .unstack(fill_value=0).sort_index())
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    frame.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("repair passes")
    ax.set_title(title)
    ax.legend(title="rung", fontsize=8)
    return _save(fig, outdir, name)



def make_plots(rows: Sequence[dict], drows: Sequence[dict], outdir: str
               ) -> List[str]:
    """Every SPEC 12 plot the data can support (missing sweeps are skipped)."""
    import pandas as pd

    if not rows:
        return []
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(list(rows))
    dd = pd.DataFrame(list(drows)) if drows else pd.DataFrame()
    written: List[str] = []

    def add(path: Optional[str]) -> None:
        if path:
            written.append(path)

    core = [
        (df, "N", "sum_of_costs", "cost_vs_N.png", "N", "sum of costs",
         "Cost vs fleet size"),
        (df, "N", "makespan", "makespan_vs_N.png", "N", "makespan",
         "Makespan vs fleet size"),
        (df, "N", "mean_altered", "altered_vs_N.png", "N", "mean altered",
         "Altered agents vs fleet size"),
        (df, "N", "mean_messages", "messages_vs_N.png", "N", "messages/pass",
         "Negotiation messages vs fleet size"),
        (df, "density", "sum_of_costs", "cost_vs_density.png", "density",
         "sum of costs", "Cost vs blockage density"),
        (df, "density", "mean_altered", "altered_vs_density.png", "density",
         "mean altered", "Altered agents vs blockage density"),
        (df, "density", "success_rate", "success_vs_density.png", "density",
         "success rate", "Repair success vs blockage density"),
    ]
    for frame, x, y, name, xl, yl, title in core:
        if y in frame.columns:
            add(_line_png(frame, x, y, outdir, name, xl, yl, title))

    add(_heatmap_png(df, "N", "density", "sum_of_costs", outdir,
                     "heatmap_cost.png", "N", "density",
                     "Sum of costs over (N, density)"))
    add(_heatmap_png(df, "N", "density", "mean_altered", outdir,
                     "heatmap_altered.png", "N", "density",
                     "Mean altered agents over (N, density)"))
    add(_rung_stack_png(df, "N", outdir, "rung_stack_vs_N.png", "N",
                        "Rungs used vs fleet size"))
    add(_rung_stack_png(df, "density", outdir, "rung_stack_vs_density.png",
                        "density", "Rungs used vs density"))

    if not dd.empty and "n_altered_plan" in dd.columns:
        add(_altered_box_png(dd, outdir, "altered_box.png"))
    add(_mode_bars_png(df, outdir, "mode_bars.png"))
    add(_altered_defs_png(df, outdir, "altered_defs.png"))
    return written


def _altered_box_png(dd, outdir: str, name: str) -> Optional[str]:
    """Boxplot of the per-disruption altered count, one box per mode."""
    plt = _fig()
    modes = sorted(str(m) for m in dd["mode"].unique())
    data = [dd.loc[dd["mode"] == m, "n_altered_plan"].to_numpy()
            for m in modes]
    if not any(len(d) for d in data):
        plt.close("all")
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.boxplot(data)
    ax.set_xticks(range(1, len(modes) + 1), modes)
    ax.set_ylabel("altered_plan per disruption")
    ax.set_title("Altered agents per disruption")
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, outdir, name)


def _mode_bars_png(df, outdir: str, name: str) -> Optional[str]:
    """Grouped bars: the four SPEC 9.5 definitions + messages, per mode."""
    metrics = [c for c in ("mean_altered_plan", "mean_altered_path",
                           "mean_delayed_only", "mean_altered_naive",
                           "mean_messages") if c in df.columns]
    plt = _fig()
    piv = df.groupby("mode")[metrics].mean() if metrics else None
    if piv is None or piv.empty:
        plt.close("all")
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("mean per pass")
    ax.set_title("Repair cost by mode")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return _save(fig, outdir, name)


def _altered_defs_png(df, outdir: str, name: str) -> Optional[str]:
    """``altered_plan`` vs ``altered_naive`` vs ``global`` (per mode)."""
    cols = [c for c in ("mean_altered_plan", "mean_altered_path",
                        "mean_altered_naive") if c in df.columns]
    plt = _fig()
    piv = df.groupby("mode")[cols].mean() if cols else None
    if piv is None or piv.empty:
        plt.close("all")
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for col in cols:
        ax.plot(range(len(piv.index)), piv[col].to_numpy(), marker="o",
                label=col)
    ax.set_xticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_ylabel("mean altered agents")
    ax.set_title("altered_plan vs altered_naive (v1)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, outdir, name)



# ----------------------------------------------------------------------
# the two extra mini sweeps of SPEC 12
# ----------------------------------------------------------------------
def radius_ablation(seeds: int = 3, radii: Sequence[int] = (3, 6, 9), *,
                    N: int = 20, density: float = 0.05, tasks_per_agent: int = 3,
                    H: int = 30, W: int = 30, jobs: int = 1) -> List[dict]:
    """``comm_radius`` ablation: the same instance at radii ``{3, 6, 9}``."""
    rows: List[dict] = []
    for radius in radii:
        families = [{"N": N, "density": density, "seed": s, "sweep": "radius",
                     "tasks_per_agent": tasks_per_agent, "H": H, "W": W,
                     "comm_radius": radius} for s in range(seeds)]
        part, _drows, _secs = run_sweep(families, ["negotiate"], jobs=jobs,
                                        verbose=False)
        for row in part:
            row["comm_radius"] = radius
        rows += part
    rows.sort(key=lambda r: (r["comm_radius"], r["seed"]))
    return rows


def plot_radius_ablation(rows: Sequence[dict], outdir: str,
                         name: str = "comm_radius_ablation.png") -> Optional[str]:
    """Altered agents (left) and messages (right) vs ``comm_radius``."""
    import pandas as pd

    if not rows:
        return None
    df = pd.DataFrame(list(rows))
    plt = _fig()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    stats = df.groupby("comm_radius")[["mean_altered_plan", "mean_messages"]].mean()
    ax.plot(stats.index, stats["mean_altered_plan"], marker="o",
            color="#ffd54f", label="mean altered_plan")
    ax.set_xlabel("comm_radius")
    ax.set_ylabel("mean altered_plan")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(stats.index, stats["mean_messages"], marker="s",
             color="#58a6ff", label="mean messages")
    ax2.set_ylabel("mean messages")
    ax.set_title("comm_radius ablation (negotiate)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="best")
    return _save(fig, outdir, name)


def pocl_nodes_sweep(seeds: int = 3, task_counts: Sequence[int] = (1, 2, 3, 4, 5, 6),
                     *, n_agents: int = 8, density: float = 0.05, H: int = 30,
                     W: int = 30) -> List[dict]:
    """POP size (action nodes) and grounding nodes vs tasks per agent."""
    import planner
    from pocl import pocl

    rows: List[dict] = []
    for k in task_counts:
        for seed in range(seeds):
            cfg = Config(H=H, W=W, n_agents=n_agents, tasks_per_agent=k,
                         seed=seed, obstacle_density=density)
            world, tasks, parking = build_world(cfg, np.random.default_rng(seed))
            per_agent = assign_tasks(tasks, parking, np.random.default_rng(seed))
            pops = [pocl(a, per_agent[a], parking[a], world,
                         max_nodes=cfg.pocl_max_nodes)
                    for a in range(cfg.n_agents)]
            nodes = [len(p.steps) for p in pops if p is not None]
            before = planner.ST_ASTAR_NODES
            peg_solve(world, per_agent, parking, max_nodes=cfg.pocl_max_nodes)
            grounding = planner.ST_ASTAR_NODES - before
            rows.append({
                "tasks_per_agent": k, "seed": seed,
                "mean_pocl_nodes": float(np.mean(nodes)) if nodes else 0.0,
                "max_pocl_nodes": float(np.max(nodes)) if nodes else 0.0,
                "pocl_failed": sum(1 for p in pops if p is None),
                "st_astar_nodes": grounding,
            })
    rows.sort(key=lambda r: (r["tasks_per_agent"], r["seed"]))
    return rows


def plot_pocl_nodes(rows: Sequence[dict], outdir: str,
                    name: str = "pocl_nodes_vs_tasks.png") -> Optional[str]:
    """Mean POP action nodes per agent vs tasks per agent (with error bars)."""
    import pandas as pd

    if not rows:
        return None
    df = pd.DataFrame(list(rows))
    stats = (df.groupby("tasks_per_agent")[["mean_pocl_nodes", "st_astar_nodes"]]
             .agg(["mean", "std"]))
    plt = _fig()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for col, colour in (("mean_pocl_nodes", "#ffd54f"),
                        ("st_astar_nodes", "#58a6ff")):
        ax.errorbar(stats.index.to_numpy(), stats[(col, "mean")].to_numpy(),
                    yerr=stats[(col, "std")].fillna(0.0).to_numpy(),
                    marker="o", capsize=3, color=colour, label=col)
    ax.set_xlabel("tasks per agent")
    ax.set_ylabel("nodes (mean per instance)")
    ax.set_title("POCL / grounding effort vs tasks per agent")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, outdir, name)



# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
DEFAULT_MODES = ("negotiate", "self_only", "global")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="experiments.py",
        description="Run the SPEC 12 sweeps and write results.csv + plots.")
    ap.add_argument("--sweep", choices=("A", "B", "grid"), default="grid",
                    help="A: N in {5..40} at 5%% density; B: density in "
                         "{0..15}%% at N=20; grid: every combination")
    ap.add_argument("--seeds", type=int, default=10,
                    help="seed count per (N, density) cell (SPEC: >= 10)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny sweep (2 seeds x 2 N x 2 densities) for CI")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                    help="multiprocessing pool size (default: nproc-1)")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES),
                    help="comma-separated repair modes")
    ap.add_argument("--tasks-per-agent", type=int, default=3)
    ap.add_argument("--height", type=int, default=30)
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--plots", action="store_true",
                    help="also write the SPEC 12 figures into --outdir")
    ap.add_argument("--ablations", action="store_true",
                    help="also run the comm_radius and POCL-node mini sweeps")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    families = sweep_jobs(args.sweep, args.seeds, quick=args.quick,
                          tasks_per_agent=args.tasks_per_agent)
    for family in families:
        family["H"], family["W"] = args.height, args.width
    print(f"experiments: sweep {args.sweep}"
          f"{' (quick)' if args.quick else ''}, {len(families)} families "
          f"x {len(modes)} modes, {args.jobs} job(s)")
    rows, drows, secs = run_sweep(families, modes, jobs=args.jobs,
                                  verbose=not args.quiet)
    meta = run_meta(args, families, modes, secs, len(rows))
    paths = write_outputs(args.outdir, rows, drows, meta)
    print(f"  {paths['results']}: {len(rows)} rows | "
          f"{paths['disruptions']}: {len(drows)} rows | {paths['meta']}  "
          f"({secs:.1f}s)")
    if args.plots:
        made = make_plots(rows, drows, args.outdir)
        print(f"  {len(made)} figure(s) written to {args.outdir}")
    if args.ablations:
        radius_rows = radius_ablation(seeds=min(3, args.seeds), jobs=args.jobs,
                                      H=args.height, W=args.width)
        _write_csv(radius_rows, tuple(radius_rows[0]) if radius_rows else (),
                   os.path.join(args.outdir, "radius_ablation.csv"))
        plot_radius_ablation(radius_rows, args.outdir)
        node_rows = pocl_nodes_sweep(seeds=min(3, args.seeds),
                                     H=args.height, W=args.width)
        _write_csv(node_rows, tuple(node_rows[0]) if node_rows else (),
                   os.path.join(args.outdir, "pocl_nodes.csv"))
        plot_pocl_nodes(node_rows, args.outdir)
        print(f"  ablations: comm_radius + POCL nodes -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

