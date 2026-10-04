"""Plan-space visualisation: POP graph, before/after Gantt, repair cards.

``SPEC.md`` section 11 asks ``viz_plan.py`` for three plan-space views:

* :func:`pop_figure` -- the partial-order plan of one agent as a layered DAG
  (networkx): ``order`` edges solid, causal links dashed and coloured, node
  colour = action kind, node label = action label / cell;
* :func:`viz.gantt_panels` (re-exported through :func:`gantt_before_after`) --
  the Gantt of the remaining plans before and after one repair pass;
* :func:`repair_card` / :func:`repair_cards` -- the three-panel "repair card"
  of a :class:`modify.RepairRecord`: the record text (rung, altered sets,
  messages, budgets), the pre-repair routes and the post-repair routes.

:func:`ladder_figure` draws the R0-R4 escalation ladder flowchart with the
rungs a scenario actually used highlighted (the report figure of section 15),
and :func:`phase_banner` renders the five-phase freeze-frame
(diagnose -> unrefine -> refine -> negotiate -> commit).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")                                  # headless

from matplotlib import pyplot as plt                   # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import viz
from metrics import (ALTERED_PATH, ALTERED_PLAN, DELAYED_ONLY, UNCHANGED,
                     altered_agent_sets)                    # noqa: E402
from world import World                                     # noqa: E402

#: the five phases of ``SPEC.md`` 9 (plan modification)
PHASES = ("diagnose", "unrefine", "refine", "negotiate", "commit")

#: escalation ladder of ``SPEC.md`` 9.1 (``modify.LEVELS``)
RUNGS = ("R0", "R1", "R2", "R3", "R4")

RUNG_COLORS = {"R0": "#3fb950", "R1": "#58a6ff", "R2": "#ffd54f",
               "R3": "#ff9f45", "R4": "#ff5c4d"}
KIND_COLORS = {"start": "#8b949e", "finish": "#8b949e", "goto": "#58a6ff",
               "travel": "#58a6ff", "pick": "#3fb950", "fetch": "#3fb950",
               "drop": "#d29922", "haul": "#d29922", "emergency": "#ff5c4d"}


def _node_label(act) -> str:
    label = act.label or act.kind
    cell = act.cell_to or act.location or act.cell_from
    return f"{label}\n{cell}" if cell else str(label)


# ----------------------------------------------------------------------
# POP graph
# ----------------------------------------------------------------------
def _layers(pop) -> Dict[int, int]:
    """Longest-path layer of every step (0 = the ``start`` action)."""
    order = sorted(pop.order)
    depth = {i: 0 for i in range(len(pop.steps))}
    for _ in range(len(pop.steps)):
        changed = False
        for a, b in order:
            if depth[b] < depth[a] + 1:
                depth[b] = depth[a] + 1
                changed = True
        if not changed:
            break
    return depth


def pop_figure(pop, *, title: str = "POP",
               figsize: Tuple[float, float] = (16, 9),
               ax=None) -> "plt.Figure":
    """Layered DAG of a POP: solid ``order`` edges, dashed causal links."""
    import networkx as nx

    depth = _layers(pop)
    buckets: Dict[int, List[int]] = {}
    for i, d in enumerate(depth.values()):
        buckets.setdefault(d, []).append(i)
    pos: Dict[int, Tuple[float, float]] = {}
    for d, nodes in sorted(buckets.items()):
        n = len(nodes)
        for k, i in enumerate(sorted(nodes)):
            pos[i] = (float(d), (k - (n - 1) / 2.0))

    g = nx.DiGraph()
    g.add_nodes_from(range(len(pop.steps)))
    g.add_edges_from(sorted(pop.order))
    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=figsize, facecolor=viz.BG)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    ax.set_facecolor(viz.PANEL)
    for (a, b) in sorted(pop.order):
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>",
                                     color="#8b949e", lw=1.0,
                                     shrinkA=16, shrinkB=18, zorder=1))
    for (a, b, fact) in pop.links:
        col = "#3fb950" if "res" in str(fact) else "#ffd54f"
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>",
                                     color=col, lw=1.2, alpha=0.85,
                                     linestyle=(0, (4, 2)), connectionstyle=
                                     "arc3,rad=0.18", shrinkA=16, shrinkB=18,
                                     zorder=2))
    for i, act in enumerate(pop.steps):
        x, y = pos[i]
        col = KIND_COLORS.get(act.kind, "#8b949e")
        ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.16), 0.84, 0.32,
                                    boxstyle="round,pad=0.02", fc=col,
                                    ec="#0b0f13", lw=0.8, alpha=0.9, zorder=3))
        ax.text(x, y, _node_label(act), color="#0b0f13", fontsize=7,
                ha="center", va="center", zorder=4)
    ax.set_title(f"{title}   (order: {len(pop.order)} edges, "
                 f"links: {len(pop.links)})", color=viz.FG, fontsize=11,
                 loc="left")
    ax.margins(0.08)
    ax.axis("off")
    handles = [plt.Line2D([], [], color="#8b949e", lw=1.2, label="order"),
               plt.Line2D([], [], color="#3fb950", lw=1.2, ls=(0, (4, 2)),
                          label="causal link (res)"),
               plt.Line2D([], [], color="#ffd54f", lw=1.2, ls=(0, (4, 2)),
                          label="causal link (other)")]
    ax.legend(handles=handles, fontsize=7, facecolor=viz.PANEL,
              labelcolor=viz.FG, framealpha=0.6, loc="best")
    if own_fig:
        fig.tight_layout()
    return fig


def pop_panel(world: World, pop, *, agent: int = 0, figsize=(16, 9)):
    """Grid + POP side by side (the ``o`` panel of the SPEC 11 keys)."""
    fig = plt.figure(figsize=figsize, facecolor=viz.BG)
    ax1 = fig.add_subplot(1, 2, 1)
    viz.grid_ax(ax1, world, f"agent {agent}")
    ax2 = fig.add_subplot(1, 2, 2)
    pop_figure(pop, title=f"POP of agent {agent}", ax=ax2)
    fig.tight_layout()
    return fig



# ----------------------------------------------------------------------
# Gantt before / after
# ----------------------------------------------------------------------
def gantt_before_after(prev, new, *, t_freeze: Optional[int] = None,
                       categories: Optional[Dict[int, str]] = None,
                       figsize: Tuple[float, float] = (16, 9)):
    """Remaining plans before and after one repair pass, stacked."""
    return viz.gantt_panels([("before repair", prev), ("after repair", new)],
                            categories=categories, t_freeze=t_freeze,
                            figsize=figsize)


def categories_of(prev, new, t_freeze: int, disrupted: Sequence[int] = ()):
    """Category of every agent (``SPEC.md`` 9.5) for the Gantt / cards."""
    sets = altered_agent_sets(prev, new, t_freeze, disrupted=disrupted)
    out = {}
    for a in sorted(set(prev) | set(new)):
        if a in sets[ALTERED_PLAN]:
            out[a] = ALTERED_PLAN
        elif a in sets[ALTERED_PATH]:
            out[a] = ALTERED_PATH
        elif a in sets[DELAYED_ONLY]:
            out[a] = DELAYED_ONLY
        else:
            out[a] = UNCHANGED
    return out


# ----------------------------------------------------------------------
# repair cards
# ----------------------------------------------------------------------
def _plan_frames(world: World, plans, tasks, parking, t_freeze: int,
                 horizon: Optional[int] = None) -> List[dict]:
    if horizon is None:
        horizon = max((s[-1].t for s in plans.values() if s), default=t_freeze)
    return viz.build_frames(world, plans, tasks, parking, t_freeze=t_freeze,
                            horizon=horizon)


def _scene(ax, world: World, frames: Sequence[dict], t: int,
           title: str) -> None:
    """One grid panel of a repair card (routes + tasks + events)."""
    frame = viz.frame_at(frames, t)
    viz.grid_ax(ax, world, title)
    viz.draw_tasks(ax, frame)
    viz.draw_blocked(ax, frame)
    viz.draw_agents(ax, frame)


def _record_text(rec) -> str:
    msgs = ", ".join(f"{k}:{v}" for k, v in
                     (rec.messages_by_type or {}).items() if v)
    return (
        f"event      {rec.event_type}\n"
        f"t          {rec.t}\n"
        f"A0 size    {rec.A0_size}\n"
        f"rung       {rec.level_resolved}\n"
        f"success    {rec.success}\n\n"
        f"altered_plan  {len(rec.altered_plan_ids)} "
        f"{rec.altered_plan_ids}\n"
        f"altered_path  {len(rec.altered_path_ids)}\n"
        f"delayed_only  {len(rec.delayed_ids)}\n"
        f"altered_naive {len(rec.naive_ids)}\n\n"
        f"added delay   {rec.added_delay_total}\n"
        f"messages      {msgs or '-'}\n"
        f"pocl nodes    {rec.pocl_nodes}\n"
        f"st_astar      {rec.st_astar_calls}\n"
        f"cpu ms        {rec.cpu_ms:.2f}")


def repair_card(record, *, world: World, tasks=(), parking=(), prev=None,
                new=None, t: Optional[int] = None,
                figsize: Tuple[float, float] = (16, 5)):
    """The three-panel repair card of one ``RepairRecord``.

    Left: the record itself (rung, altered sets, messages, budgets).  Middle /
    right: the remaining routes before and after the pass, seen at the freeze
    tick (dots are the futures, rings are the altered categories).
    """
    t = getattr(record, "t", 0) if t is None else t
    fig = plt.figure(figsize=figsize, facecolor=viz.BG)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.2, 1.2), wspace=0.12)
    ax_txt = fig.add_subplot(gs[0, 0])
    ax_txt.set_facecolor(viz.PANEL)
    ax_txt.axis("off")
    ax_txt.text(0.02, 0.98, _record_text(record), color=viz.FG, fontsize=8,
                va="top", family="monospace", transform=ax_txt.transAxes)
    colour = RUNG_COLORS.get(getattr(record, "level_resolved", ""), "#8b949e")
    ax_txt.add_patch(FancyBboxPatch((0.02, 0.01), 0.96, 0.05,
                                    boxstyle="round,pad=0.01",
                                    fc=colour, ec="none", alpha=0.75,
                                    transform=ax_txt.transAxes))
    if prev is not None:
        frames = _plan_frames(world, prev, tasks, parking, t)
        _scene(fig.add_subplot(gs[0, 1]), world, frames, t,
               f"before repair   t={t}")
    if new is not None:
        frames = _plan_frames(world, new, tasks, parking, t)
        _scene(fig.add_subplot(gs[0, 2]), world, frames, t,
               f"after repair   t={t}")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.04)
    return fig


def repair_cards(records: Sequence[object], *, world: World, tasks=(),
                 parking=(), prev=None, new=None, outdir: Optional[str] = None,
                 prefix: str = "repair_card", max_cards: int = 4,
                 dpi: int = 110) -> List[str]:
    """PNG repair cards for the first ``max_cards`` records; returns paths."""
    paths: List[str] = []
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    for k, rec in enumerate(list(records)[:max_cards]):
        t = getattr(rec, "t", 0)
        fig = repair_card(rec, world=world, tasks=tasks, parking=parking,
                          prev=prev, new=new, t=t)
        path = os.path.join(outdir or ".", f"{prefix}_{k:02d}_t{t}.png")
        fig.savefig(path, dpi=dpi, facecolor=viz.BG)
        plt.close(fig)
        paths.append(path)
    return paths



# ----------------------------------------------------------------------
# the escalation ladder and the five-phase freeze-frames
# ----------------------------------------------------------------------
#: one-line description of each rung (SPEC 9.1)
RUNG_DESC = {
    "R0": "keep plan,\nwait only",
    "R1": "local detour\n(R0-R1)",
    "R2": "widen radius,\nnegotiate (R2)",
    "R3": "re-derive\ntask order",
    "R4": "hand-off /\nre-assign",
}


def ladder_figure(used: Sequence[str] = (), *, figsize=(16, 4),
                  title: str = "repair ladder R0-R4"):
    """The escalation ladder flowchart, with the rungs in ``used`` lit up."""
    fig = plt.figure(figsize=figsize, facecolor=viz.BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(viz.PANEL)
    ax.axis("off")
    used = set(used)
    n = len(RUNGS)
    for k, rung in enumerate(RUNGS):
        x = k / (n - 1) if n > 1 else 0.0
        col = RUNG_COLORS[rung] if rung in used else "#39424a"
        ax.add_patch(FancyBboxPatch((x - 0.075, 0.42), 0.15, 0.22,
                                    boxstyle="round,pad=0.02", fc=col,
                                    ec="#0b0f13", lw=1.0, alpha=0.95,
                                    transform=ax.transAxes))
        ax.text(x, 0.53, rung, color="#0b0f13", fontsize=13, ha="center",
                va="center", transform=ax.transAxes, weight="bold")
        ax.text(x, 0.30, RUNG_DESC[rung], color=viz.FG, fontsize=8,
                ha="center", va="top", transform=ax.transAxes)
        if k < n - 1:
            ax.annotate("", xy=((k + 1) / (n - 1) - 0.085, 0.53),
                        xytext=(x + 0.085, 0.53), xycoords="axes fraction",
                        textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="#8b949e",
                                        lw=1.2, shrinkA=0, shrinkB=0))
    ax.set_title(title, color=viz.FG, fontsize=11, loc="left")
    ax.text(0.0, 0.85, "stop at the first acceptable candidate "
                       "(added delay <= delay_threshold)", color=viz.FG,
            fontsize=8, transform=ax.transAxes)
    fig.tight_layout()
    return fig


def sequence_figure(log, *, figsize=(16, 6),
                    title: str = "negotiation message sequence"):
    """Message-sequence diagram of one repair pass (``SPEC.md`` 14.6).

    One lifeline per agent that spoke, one arrow per protocol message with its
    ``detail``, time flowing downwards, coloured by message type.  A pass that
    resolved locally (no holder to ask) draws the honest "no negotiation"
    note: the protocol only speaks when somebody has to yield.
    """
    msgs = list(getattr(log, "messages", []) or [])
    fig = plt.figure(figsize=figsize, facecolor=viz.BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(viz.PANEL)
    ax.axis("off")
    ax.set_title(title, color=viz.FG, fontsize=11, loc="left")
    if not msgs:
        ax.text(0.5, 0.5, "no protocol messages\n"
                "(every repair resolved locally:\nno holder had to yield)",
                color=viz.FG, fontsize=12, ha="center", va="center",
                transform=ax.transAxes)
        return fig
    people = sorted({m.sender for m in msgs} | {m.receiver for m in msgs})
    x = {a: k / max(1, len(people) - 1) for k, a in enumerate(people)}
    top, bottom = 0.90, 0.06
    for a, xx in x.items():
        ax.plot([xx, xx], [bottom, top], color="#39424a", lw=0.9, ls="--",
                transform=ax.transAxes, zorder=1)
        ax.text(xx, top + 0.03, f"a{a}", color=viz.FG, fontsize=9,
                ha="center", transform=ax.transAxes)
    step = (top - bottom) / (len(msgs) + 1)
    for i, m in enumerate(msgs):
        y = top - step * (i + 1)
        x0, x1 = x[m.sender], x[m.receiver]
        col = viz.MSG_COLORS.get(m.kind, "#8b949e")
        ax.annotate("", xy=(x1, y), xytext=(x0, y), xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4,
                                    shrinkA=2, shrinkB=2))
        label = f"t={m.t}  {m.kind}" + (f"  ({m.detail})" if m.detail else "")
        ax.text((x0 + x1) / 2, y + 0.012, label, color=col, fontsize=7,
                ha="center", va="bottom", transform=ax.transAxes)
    handles = [plt.Line2D([], [], color=c, lw=2, label=k)
               for k, c in viz.MSG_COLORS.items()]
    ax.legend(handles=handles, fontsize=7, facecolor=viz.PANEL,
              labelcolor=viz.FG, framealpha=0.6, loc="lower right", ncol=3)
    return fig


def phase_banner(ax, phase: str, rung: Optional[str] = None) -> None:
    """Draw the phase banner + rung label of a freeze-frame."""
    kind = "refine / negotiate" if phase in ("refine", "negotiate") else phase
    colour = RUNG_COLORS.get(rung or "", "#58a6ff")
    ax.add_patch(FancyBboxPatch((0.01, 0.90), 0.98, 0.075,
                                boxstyle="round,pad=0.01", fc=viz.PANEL,
                                ec=colour, lw=1.4, alpha=0.95,
                                transform=ax.transAxes, zorder=20))
    text = f"phase {PHASES.index(phase) + 1}/5: {kind}"
    if rung:
        text += f"      rung {rung}"
    ax.text(0.5, 0.938, text, color=viz.FG, fontsize=9, ha="center",
            va="center", transform=ax.transAxes, zorder=21)


def phase_frames(record, *, world: World, frames: Sequence[dict], t: int,
                 outdir: Optional[str] = None, prefix: str = "phase",
                 dpi: int = 100) -> List[str]:
    """The five freeze-frames of a repair pass (banner + rung label).

    The schedule does not change *within* a pass -- the phases are the steps of
    ``modify.repair_schedule`` -- so all five frames show the freeze tick; the
    banner names the phase and the rung that resolved it.
    """
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    paths: List[str] = []
    for phase in PHASES:
        fig = viz.plot_overview(world, frames, t)
        ax = fig.axes[0]
        phase_banner(ax, phase, getattr(record, "level_resolved", None))
        path = os.path.join(outdir or ".", f"{prefix}_{phase}.png")
        fig.savefig(path, dpi=dpi, facecolor=viz.BG)
        plt.close(fig)
        paths.append(path)
    return paths

