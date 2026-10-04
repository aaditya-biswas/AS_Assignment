"""Grid animation, Gantt charts and figure export (``SPEC.md`` section 11).

Everything here is headless: ``matplotlib.use("Agg")`` is set on import, so the
module is safe to import from tests, from ``experiments.py`` and from a CI box
without a display.  The entry points are

* :func:`build_frames` -- one trace frame per tick, in the ``SPEC.md`` 11
  schema (agents, blocked cells, tasks, events, messages, ``waiting_for``,
  category colours, repair log, counters);
* :func:`plot_overview` -- grid + metrics panel + event log + cumulative line
  plot, the panel of the base animation;
* :func:`render_grid_frame` -- the grid alone (used by the freeze-frames);
* :func:`render_gif`, :func:`render_mp4`, :func:`screenshots` -- the exports;
* :func:`gantt` -- the per-agent Gantt chart;
* :func:`before_after_panels` -- the v2 before/after view of one repair pass.

Plan-space views (POP graph, before/after Gantt, repair cards) live in
:mod:`viz_plan`.  A frame is a plain ``dict`` (JSON-able apart from the cells,
which are ``(row, col)`` tuples), so the animation can be replayed without
re-running the planner.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")                    # headless: never needs a display

import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter   # noqa: E402
from matplotlib.colors import ListedColormap        # noqa: E402
from matplotlib.patches import Circle, Rectangle    # noqa: E402

from metrics import (ALTERED_PATH, ALTERED_PLAN, DELAYED_ONLY, UNCHANGED,
                     altered_agent_sets)
from planner import DROP, MOVE, PICK, WAIT, PlanStep
from world import Cell, Task, TaskStatus, World            # noqa: F401

# ---- dark palette of the SPEC ----------------------------------------
BG = "#0f1418"
PANEL = "#161d24"
FG = "#d7e0e8"
SHELF_C = "#43535f"
GRID_C = "#232d36"
BLOCK_C = "#c0392b"
PICKUP_C = "#35b7c9"
DELIVER_C = "#b07cd6"

#: ring style per altered category (SPEC 11: "altered-category colours")
CAT_STYLE = {
    ALTERED_PLAN: dict(ec="#ffd54f", lw=3.0, ls="-"),
    ALTERED_PATH: dict(ec="#ffb74d", lw=2.0, ls="-"),
    DELAYED_ONLY: dict(ec="#ff8a65", lw=1.2, ls=(0, (3, 2))),
    UNCHANGED: dict(ec="#5c6b73", lw=0.8, ls="-"),
}

ACTION_C = {MOVE: "#2f81f7", WAIT: "#4a5a66", PICK: "#3fb950", DROP: "#d29922"}

#: shelf map: 0 = free (background), 1 = shelf
GRID_CMAP = ListedColormap([BG, SHELF_C])

#: how long a negotiation arrow stays visible in the animation
MESSAGE_HOLD = 6
#: dotted future paths are truncated to this many ticks
FUTURE_TICKS = 30

#: colour of a protocol message by type (``negotiation.MESSAGE_TYPES``)
MSG_COLORS = {"PROPOSE_ORDER": "#ffd54f", "PROPOSE_REROUTE": "#ffb74d",
              "ACCEPT": "#3fb950", "REJECT": "#ff5c4d",
              "COMMIT": "#58a6ff", "ABORT": "#b23b2e"}
#: short on-grid label of a protocol message (``negotiation.MESSAGE_TYPES``)
MSG_ABBR = {"PROPOSE_ORDER": "ORDER", "PROPOSE_REROUTE": "REROUTE",
            "ACCEPT": "ACCEPT", "REJECT": "REJECT",
            "COMMIT": "COMMIT", "ABORT": "ABORT"}
#: at most this many speech bubbles per agent and tick (the rest fold into +N)
MSG_LABELS_MAX = 2
#: directional glyph of a bubble: the sender speaks, the receiver is told
MSG_GLYPH_OUT = "\u00bb"
MSG_GLYPH_IN = "\u00ab"
#: short label of every activity phase of the per-agent taskboard
PHASE_ABBR = {"TO_PICKUP": "get ", "TO_DELIVER": "haul", "TO_PARK": "park",
              "IDLE": "idle", "DONE": "done", "BROKEN": "BRKN"}


# ----------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------
def _trace(steps: Optional[Sequence[PlanStep]], horizon: int) -> List[Cell]:
    """Cell-per-tick trace of ``steps``, padded at both ends."""
    if not steps:
        return []
    pos = {s.t: s.cell for s in steps}
    t0, t1 = steps[0].t, steps[-1].t
    return [pos.get(t, steps[0].cell if t < t0 else steps[-1].cell)
            for t in range(horizon + 1)]


def _agent_timeline(steps: Optional[Sequence[PlanStep]]) -> Dict[str, object]:
    """Picked/dropped ticks and the carrying window of one agent."""
    picked: Dict[int, int] = {}
    dropped: Dict[int, int] = {}
    for s in steps or ():
        if s.action == PICK and s.item is not None:
            picked.setdefault(s.item, s.t)
        elif s.action == DROP and s.item is not None:
            dropped.setdefault(s.item, s.t)
    return {"picked": picked, "dropped": dropped}


def _carrying(steps: Optional[Sequence[PlanStep]], t: int) -> Optional[int]:
    item: Optional[int] = None
    for s in steps or ():
        if s.t > t:
            break
        if s.action == PICK:
            item = s.item
        elif s.action == DROP:
            item = None
    return item


def _activity_index(steps):
    """Pick/drop tick per item of one agent, ordered by plan position."""
    picks: Dict[int, int] = {}
    drops: Dict[int, int] = {}
    for s in steps or ():
        if s.action == PICK and s.item is not None:
            picks.setdefault(s.item, s.t)
        elif s.action == DROP and s.item is not None:
            drops.setdefault(s.item, s.t)
    order = sorted(set(picks) | set(drops),
                   key=lambda i: (picks.get(i, 10 ** 9),
                                  drops.get(i, 10 ** 9)))
    return picks, drops, order


def _activity_at(t: int, idx, task_by_id) -> dict:
    """What one agent is doing at ``t``: task, phase, target, progress.

    The phases are the ones a reviewer watches for -- heading to a pickup,
    hauling to a delivery, parking, idle/done -- plus the delivery progress.
    """
    picks, drops, order = idx
    done = sum(1 for i in order if drops.get(i, 10 ** 9) <= t)
    for item in order:
        if drops.get(item, 10 ** 9) > t:                 # still open
            tk = task_by_id.get(item)
            to_pickup = picks.get(item, 10 ** 9) > t
            target = (tk.pickup if to_pickup else tk.delivery) if tk else None
            return {"task_id": item,
                    "phase": "TO_PICKUP" if to_pickup else "TO_DELIVER",
                    "target": target, "done": done, "total": len(order)}
    return {"task_id": None, "phase": "TO_PARK" if order else "IDLE",
            "target": None, "done": done, "total": len(order)}


def _serialize_event(ev) -> dict:
    """``Event``/``Disruption`` -> plain dict, for the trace and event log."""
    if hasattr(ev, "kind"):          # modify.Disruption (already normalized)
        return {"kind": ev.kind, "agent": ev.agent, "cell": ev.cell,
                "t0": ev.t, "duration": ev.duration,
                "permanent": bool(ev.permanent), "hold": ev.hold}
    if hasattr(ev, "permanent"):     # disruptions.AccidentEvent
        return {"kind": "breakdown" if ev.permanent else "accident",
                "agent": ev.agent, "cell": None, "t0": ev.at_t,
                "duration": ev.duration, "permanent": bool(ev.permanent),
                "hold": 0}
    if hasattr(ev, "hold"):          # disruptions.EmergencyEvent
        return {"kind": "emergency", "agent": ev.agent, "hold": ev.hold,
                "cell": ev.cell, "t0": ev.at_t, "duration": ev.hold,
                "permanent": False}
    return {"kind": "blockage",      # disruptions.BlockageEvent
            "cell": ev.cell, "t0": ev.t0, "duration": ev.duration,
            "agent": None, "permanent": False, "hold": 0}


def _event_tick(ev) -> int:
    if hasattr(ev, "kind"):
        return int(ev.t)
    return int(getattr(ev, "at_t", getattr(ev, "t0", 0)))


def _event_end(tick: int, ev) -> int:
    """Last tick at which the event is still active (``tick`` inclusive)."""
    dur = int(getattr(ev, "duration", 0) or 0)
    hold = int(getattr(ev, "hold", 0) or 0)
    if getattr(ev, "permanent", False):
        return 10 ** 9
    return tick + max(0, max(dur, hold) - 1)


def build_frames(
    world: World,
    plans: Dict[int, Optional[List[PlanStep]]],
    tasks: Sequence[Task],
    parking: Sequence[Cell],
    *,
    prev: Optional[Dict[int, Optional[List[PlanStep]]]] = None,
    t_freeze: int = 0,
    disrupted: Sequence[int] = (),
    events: Sequence[object] = (),
    records: Sequence[object] = (),
    log: Optional[object] = None,
    traces: Optional[Dict[int, List[Cell]]] = None,
    horizon: Optional[int] = None,
) -> List[dict]:
    """Trace frames of a *finished* schedule, one ``dict`` per tick.

    ``prev`` is the pre-repair plan map (used for the altered-category
    colours), ``records`` the :class:`modify.RepairRecord` list and ``log`` a
    :class:`negotiation.Negotiation`; both feed the per-frame counters, the
    repair log and the message arrows (an arrow stays visible for
    ``MESSAGE_HOLD`` ticks).  ``traces`` overrides the derived cell tracks (the
    simulator already records them, so there is no reason to derive them
    twice).
    """
    ids = sorted(set(plans) | set(prev or {}))
    if horizon is None:
        horizon = 0
        for steps in plans.values():
            if steps:
                horizon = max(horizon, steps[-1].t)
        for tr in (traces or {}).values():
            horizon = max(horizon, len(tr) - 1)
    if traces is None:
        traces = {a: _trace(plans.get(a), horizon) for a in ids}
    categories = _categories(prev, plans, t_freeze, disrupted, ids)
    timeline = {a: _agent_timeline(plans.get(a)) for a in ids}
    task_by_id = {tk.id: tk for tk in tasks}
    act_idx = {a: _activity_index(plans.get(a)) for a in ids}
    finish = {a: (plans[a][-1].t if plans.get(a) else None) for a in ids}
    ev_tick = [_event_tick(e) for e in events]
    msgs = list(getattr(log, "messages", ()) or ())

    frames: List[dict] = []
    for t in range(horizon + 1):
        blocked = []
        for cell, windows in sorted(world.blocked_until.items()):
            live = [w1 for (w0, w1) in windows if w0 <= t <= w1]
            if live:
                blocked.append({"cell": cell, "until": max(live)})
        active = [_serialize_event(e) for i, e in enumerate(events)
                  if ev_tick[i] <= t <= _event_end(ev_tick[i], e)]
        held = {e["agent"] for e in active if e["kind"] in ("accident",
                                                            "breakdown",
                                                            "emergency")}
        broken = {e["agent"] for e in active if e["kind"] == "breakdown"}

        agents = []
        for a in ids:
            tr = traces.get(a) or []
            pos = tr[t] if t < len(tr) else (tr[-1] if tr else None)
            if pos is None:
                pos = parking[a] if a < len(parking) else (0, 0)
            if a in broken:
                state = "BROKEN"
            elif a in held:
                state = "HOLD"
            elif finish[a] is not None and t >= finish[a]:
                state = "DONE"
            elif t + 1 < len(tr) and tr[t + 1] != pos:
                state = "MOVING"
            else:
                state = "WAITING"
            nxt = pos
            for tt in range(t, min(len(tr), t + 1 + FUTURE_TICKS)):
                if tr[tt] != pos:
                    nxt = tr[tt]
                    break
            act = _activity_at(t, act_idx[a], task_by_id)
            if state == "DONE":
                act = dict(act, phase="DONE", task_id=None, target=None)
            agents.append({
                "id": a,
                "pos": pos,
                "state": state,
                "carrying_task_id": _carrying(plans.get(a), t),
                "plan_future": tr[t + 1: t + 1 + FUTURE_TICKS],
                "next": nxt,
                "task": act,
                "altered_flash": a in set(disrupted) and t >= t_freeze,
                "broken": a in broken,
            })

        waiting_for: List[Tuple[int, int]] = []
        for ag in agents:
            if ag["state"] != "WAITING" or ag["next"] == ag["pos"]:
                continue
            for other in agents:
                if other["id"] != ag["id"] and other["pos"] == ag["next"]:
                    waiting_for.append((ag["id"], other["id"]))

        done = sum(1 for tk in tasks if _dropped_at(timeline, tk.id) <= t)
        altered_so_far = sorted({a for r in records
                                 if getattr(r, "t", 0) <= t
                                 for a in getattr(r, "altered_plan_ids",
                                                  [])})
        frames.append({
            "t": t,
            "agents": agents,
            "blocked": blocked,
            "tasks": [{"id": tk.id, "pickup": tk.pickup,
                       "delivery": tk.delivery,
                       "status": _task_status(tk, t, timeline)}
                      for tk in tasks],
            "events": active,
            "messages": [],
            "waiting_for": waiting_for,
            "categories": dict(categories),
            "repair_log": [{"t": getattr(r, "t", 0),
                            "event": getattr(r, "event_type", ""),
                            "level": getattr(r, "level_resolved", ""),
                            "altered": getattr(r, "altered_plan_ids", []),
                            "messages": getattr(r, "messages_by_type", {})}
                           for r in records if getattr(r, "t", 0) <= t],
            "counters": {
                "done_tasks": done,
                "total_tasks": len(tasks),
                "altered_so_far": len(altered_so_far),
                "messages_so_far": sum(1 for m in msgs if m.t <= t),
                "disruptions_so_far": sum(1 for tick in ev_tick if tick <= t),
            },
        })

    for m in msgs:                      # arrows: visible for a few ticks
        for t in range(m.t, min(horizon, m.t + MESSAGE_HOLD) + 1):
            frames[t]["messages"].append({"t": m.t, "kind": m.kind,
                                          "sender": m.sender,
                                          "receiver": m.receiver,
                                          "detail": m.detail})
    return frames


def _categories(prev, plans, t_freeze, disrupted, ids) -> Dict[int, str]:
    """Category colour of every agent (``SPEC.md`` 9.5 / 11)."""
    if prev is None:
        return {a: UNCHANGED for a in ids}
    sets = altered_agent_sets(prev, plans, t_freeze, disrupted=disrupted)
    out: Dict[int, str] = {}
    for a in ids:
        if a in sets[ALTERED_PLAN]:
            out[a] = ALTERED_PLAN
        elif a in sets[ALTERED_PATH]:
            out[a] = ALTERED_PATH
        elif a in sets[DELAYED_ONLY]:
            out[a] = DELAYED_ONLY
        else:
            out[a] = UNCHANGED
    return out


def _dropped_at(timeline: Dict[int, dict], task_id: int) -> int:
    """Tick at which ``task_id`` is delivered (``inf`` while it is not)."""
    for info in timeline.values():
        if task_id in info.get("dropped", {}):
            return info["dropped"][task_id]
    return 10 ** 9


def _task_status(tk: Task, t: int, timeline: Dict[int, dict]) -> str:
    """Task status at ``t`` from the pickup/drop ticks of its owner."""
    drop = _dropped_at(timeline, tk.id)
    if drop <= t:
        return TaskStatus.DONE.value
    for info in timeline.values():
        if tk.id in info.get("picked", {}) and info["picked"][tk.id] <= t:
            return TaskStatus.PICKED.value
    return TaskStatus.PENDING.value


def frame_at(frames: Sequence[dict], t: int) -> dict:
    """Frame of tick ``t`` (clamped to the horizon)."""
    if not frames:
        raise ValueError("no frames")
    return frames[max(0, min(t, len(frames) - 1))]


# ----------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------
def agent_color(aid: int) -> tuple:
    return plt.get_cmap("tab20")(aid % 20)


def grid_ax(ax, world: World, title: Optional[str] = None) -> None:
    """Dark grid with shelves; row 0 on top (``(row, col)`` coordinates).

    The shelves are one ``imshow`` rather than a patch per cell: the animation
    re-lays the panel out every tick, so the per-cell loop dominated rendering.
    """
    H, W = world.H, world.W
    ax.set_facecolor(BG)
    ax.imshow(world.grid, cmap=GRID_CMAP, origin="upper",
              interpolation="nearest", extent=(-0.5, W - 0.5, H - 0.5, -0.5),
              zorder=1)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, W, 5))
    ax.set_yticks(range(0, H, 5))
    ax.tick_params(colors=FG, labelsize=6, length=2)
    ax.grid(color=GRID_C, lw=0.3, alpha=0.35, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    if title:
        ax.set_title(title, color=FG, fontsize=11, loc="left")


def draw_tasks(ax, frame: dict) -> None:
    """Pickup diamonds and delivery stars, coloured by task status."""
    stat_c = {TaskStatus.PENDING.value: "#7d8b96",
              TaskStatus.PICKED.value: "#d29922",
              TaskStatus.DONE.value: "#3fb950"}
    for tk in frame["tasks"]:
        col = stat_c.get(tk["status"], "#7d8b96")
        pr, pc = tk["pickup"]
        ax.plot(pc, pr, marker="D", ms=5, mfc="none", mec=PICKUP_C, mew=1.1,
                zorder=3)
        dr, dc = tk["delivery"]
        ax.plot(dc, dr, marker="*", ms=8, mfc=col, mec=DELIVER_C, mew=0.7,
                alpha=0.9, zorder=3)


def draw_blocked(ax, frame: dict) -> None:
    """Red hatched cells with their countdown (``until - t + 1``)."""
    t = frame["t"]
    for b in frame["blocked"]:
        r, c = b["cell"]
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=BLOCK_C,
                               alpha=0.35, hatch="///", ec=BLOCK_C, lw=0.8,
                               zorder=4))
        left = max(0, b["until"] - t + 1)
        ax.text(c, r, str(left), color="#ffdcd6", fontsize=6, ha="center",
                va="center", zorder=6)


def agent_message_labels(frame: dict) -> Dict[int, List[dict]]:
    """Speech bubbles of this tick, keyed by the agent that shows them.

    Every negotiation message is labelled over *both* of its endpoints: the
    sender with ``role="out"`` (glyph :data:`MSG_GLYPH_OUT`) and the receiver
    with ``role="in"`` (glyph :data:`MSG_GLYPH_IN`); the colour comes from
    :data:`MSG_COLORS`.  A frame without messages yields ``{}``, so a silent
    (R1/R2-hard) repair draws no bubble at all.

    At most :data:`MSG_LABELS_MAX` bubbles are kept per agent; the surplus is
    collapsed into a trailing ``+N`` entry so a busy tick stays readable.
    """
    out: Dict[int, List[dict]] = {}
    for m in frame.get("messages", ()):
        kind = m["kind"]
        sender, receiver = m["sender"], m["receiver"]
        base = {"kind": kind, "text": MSG_ABBR.get(kind, kind),
                "detail": m.get("detail", "")}
        out.setdefault(sender, []).append(
            dict(base, role="out", glyph=MSG_GLYPH_OUT))
        if receiver != sender:      # a self-message has only one bubble
            out.setdefault(receiver, []).append(
                dict(base, role="in", glyph=MSG_GLYPH_IN))
    trimmed: Dict[int, List[dict]] = {}
    for aid, labels in out.items():
        if len(labels) > MSG_LABELS_MAX:
            extra = len(labels) - (MSG_LABELS_MAX - 1)
            labels = labels[: MSG_LABELS_MAX - 1] + [
                {"kind": "_more", "text": f"+{extra}", "detail": "",
                 "role": "mix", "glyph": ""}]
        trimmed[aid] = labels
    return trimmed


def _draw_message_bubbles(ax, frame: dict, by_id: Dict[int, dict]) -> None:
    """One rounded bubble above every agent that speaks or is spoken to.

    Messages of the same kind are combined into a single bubble; multiple
    kinds stack outwards.  On the top row the stack grows *downwards* so it
    never clips out of the axes.  The bubble is a ``text`` with a rounded
    ``bbox``, which keeps it a single artist per agent and kind.
    """
    for aid, items in sorted(agent_message_labels(frame).items()):
        ag = by_id.get(aid)
        if ag is None:
            continue
        r, c = ag["pos"]
        groups: List[Tuple[str, List[dict]]] = []
        for it in items:                    # group consecutive same-kind
            if groups and groups[-1][0] == it["kind"]:
                groups[-1][1].append(it)
            else:
                groups.append((it["kind"], [it]))
        below = r <= 0                      # no room above the top row
        step = 0.46
        for gi, (kind, grp) in enumerate(groups):
            dy = (r + 0.62 + gi * step) if below else (r - 0.62 - gi * step)
            col = MSG_COLORS.get(kind, "#ffd54f")
            text = "\n".join(f"{g['glyph']} {g['text']}".strip() for g in grp)
            ax.plot([c, c], [dy, r + (0.42 if below else -0.42)], color=col,
                    lw=0.6, alpha=0.6, zorder=11)
            ax.text(c, dy, text, color=col, fontsize=5.0, ha="center",
                    va="center", family="monospace", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.22", fc=PANEL, ec=col,
                              lw=1.0, alpha=0.92))


def draw_agents(ax, frame: dict, *, show_future: bool = True,
                show_waiting: bool = True, show_messages: bool = True,
                show_message_labels: bool = True,
                carry_marker: bool = True) -> None:
    """Agent discs, category rings, dotted futures, waiting/message arrows.

    With ``show_message_labels`` every agent that sends or receives a protocol
    message this tick also gets a speech bubble over its disc (see
    :func:`agent_message_labels`); the arrows stay either way.
    """
    cats = frame["categories"]
    by_id = {ag["id"]: ag for ag in frame["agents"]}
    if show_future:
        for ag in frame["agents"]:
            if not ag["plan_future"]:
                continue
            xs = [c for (_, c) in ag["plan_future"]]
            ys = [r for (r, _) in ag["plan_future"]]
            ax.plot(xs, ys, ls=(0, (1, 2)), lw=0.8, alpha=0.6, zorder=2,
                    color=agent_color(ag["id"]))
    if show_waiting:
        for (a, b) in frame["waiting_for"]:
            if a not in by_id or b not in by_id:
                continue
            (r0, c0), (r1, c1) = by_id[a]["pos"], by_id[b]["pos"]
            ax.annotate("", xy=(c1, r1), xytext=(c0, r0),
                        arrowprops=dict(arrowstyle="->", color="#c9d1d9",
                                        lw=0.7, alpha=0.55,
                                        shrinkA=6, shrinkB=6), zorder=5)
    if show_messages:
        for m in frame["messages"]:
            src, dst = by_id.get(m["sender"]), by_id.get(m["receiver"])
            if src is None or dst is None or src is dst:
                continue
            (r0, c0), (r1, c1) = src["pos"], dst["pos"]
            col = MSG_COLORS.get(m["kind"], "#ffd54f")
            ax.annotate("", xy=(c1, r1), xytext=(c0, r0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0,
                                        alpha=0.8, linestyle=":",
                                        shrinkA=7, shrinkB=7), zorder=7)
    for ag in frame["agents"]:
        r, c = ag["pos"]
        style = CAT_STYLE.get(cats.get(ag["id"], UNCHANGED),
                              CAT_STYLE[UNCHANGED])
        ax.add_patch(Circle((c, r), 0.38, fc=agent_color(ag["id"]),
                            alpha=0.95, zorder=8, **style))
        if ag["broken"]:
            ax.plot(c, r, marker="X", ms=9, color="#ff5c4d", mew=1.6, zorder=9)
        elif carry_marker and ag["carrying_task_id"] is not None:
            ax.add_patch(Rectangle((c - 0.14, r - 0.14), 0.28, 0.28,
                                   fc="#f0f6fc", ec="none", zorder=9))
        ax.text(c, r, str(ag["id"]), color="#0b0f13", fontsize=6, ha="center",
                va="center", zorder=10)
    if show_message_labels:
        _draw_message_bubbles(ax, frame, by_id)



def draw_taskbar(ax, frame: dict, *, max_cols: int = 20) -> None:
    """Bottom strip: one cell per agent -- phase, task, target, progress.

    Cells wrap onto a new row above ``max_cols`` agents, so an ``N=40`` sweep
    still reads.  This is the fleet taskboard of an autonomous system: what
    every robot is doing right now.
    """
    ax.set_facecolor(PANEL)
    agents = frame["agents"]
    per_row = max(1, min(max_cols, len(agents))) if agents else 1
    rows = max(1, (len(agents) + per_row - 1) // per_row)
    ax.set_xlim(0, per_row)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.set_title("fleet taskboard", color=FG, fontsize=8, loc="left")
    for k, ag in enumerate(agents):
        row = rows - 1 - (k // per_row)
        col = k % per_row
        act = ag.get("task") or {}
        if ag["broken"]:
            phase = "BROKEN"
        elif ag["state"] == "DONE":
            phase = "DONE"
        else:
            phase = act.get("phase", "IDLE")
        ax.add_patch(Rectangle((col, row), 1.0, 1.0, fc="#101820", ec=GRID_C,
                               lw=0.6, zorder=1))
        ax.add_patch(Rectangle((col, row), 0.09, 1.0,
                               fc=agent_color(ag["id"]), ec="none", zorder=2))
        task = act.get("task_id")
        target = act.get("target")
        label = (f"a{ag['id']:<2d} {PHASE_ABBR.get(phase, phase):<4s} "
                 f"{('t' + str(task)) if task is not None else '--':>3s} "
                 f"{target if target else '-'}")
        ax.text(col + 0.12, row + 0.63, label, color=FG, fontsize=5.0,
                family="monospace", va="center", zorder=3)
        done, total = act.get("done", 0), act.get("total", 0)
        ax.text(col + 0.12, row + 0.27, f"{done}/{total} delivered",
                color="#8b949e", fontsize=4.6, family="monospace",
                va="center", zorder=3)
        if total:
            ax.add_patch(Rectangle((col + 0.12, row + 0.08),
                                   0.84 * (done / total), 0.06,
                                   fc="#3fb950", ec="none", zorder=3))


def _metrics_text(frame: dict) -> str:
    c = frame["counters"]
    return (f"t = {frame['t']:4d}\n"
            f"tasks {c['done_tasks']}/{c['total_tasks']}\n"
            f"altered so far {c['altered_so_far']}\n"
            f"messages {c['messages_so_far']}\n"
            f"disruptions {c['disruptions_so_far']}\n"
            f"blocked cells {len(frame['blocked'])}\n"
            f"waiting pairs {len(frame['waiting_for'])}")


def _log_text(frames: Sequence[dict], t: int, max_lines: int = 12) -> str:
    lines: List[str] = []
    for f in frames[:t + 1]:
        for ev in f["events"]:
            if ev["t0"] == f["t"]:
                who = f"a{ev['agent']}" if ev["agent"] is not None else "-"
                lines.append(f"t={f['t']:<4d} {ev['kind']:<9s} {who:<4s} "
                             f"{ev['cell'] if ev['cell'] else ''}")
        for r in f["repair_log"]:
            if r["t"] == f["t"]:
                lines.append(f"t={f['t']:<4d} repair   {r['level']:<3s} "
                             f"alt={len(r['altered'])}")
        for m in f["messages"]:
            if m["t"] == f["t"]:
                lines.append(f"t={f['t']:<4d} {m['kind']:<14s} "
                             f"{m['sender']}->{m['receiver']}")
    return "\n".join(lines[-max_lines:]) if lines else "(no events yet)"


def dialogue_lines(frames: Sequence[dict], t: int, max_lines: int = 12) -> str:
    """Protocol dialogue of the run so far (``SPEC.md`` 9.2 / 14.6).

    Two kinds of line, both in tick order: a ``repair`` line per disruption
    pass (rung, altered set, message counts) and one line per protocol message
    with its human-readable ``detail``.  A pass that needs no dialogue still
    shows its repair line -- which is the point: the protocol only speaks when
    a holder has to yield.
    """
    lines: List[str] = []
    for f in frames[:t + 1]:
        for r in f["repair_log"]:
            if r["t"] != f["t"]:
                continue
            msgs = ", ".join(f"{k}:{v}" for k, v in r["messages"].items() if v)
            lines.append(f"t={f['t']:<4d} repair {r['level']:<3s} "
                         f"alt={len(r['altered'])}  {msgs or '-'}")
        for m in f["messages"]:
            if m["t"] != f["t"]:
                continue
            arrow = f"a{m['sender']}->a{m['receiver']}"
            detail = f"  {m['detail']}" if m["detail"] else ""
            lines.append(f"t={f['t']:<4d} {arrow:<9s} {m['kind']:<14s}{detail}")
    if not lines:
        return "(no protocol messages yet)"
    return "\n".join(lines[-max_lines:])


def _cumulative(ax, frames: Sequence[dict], t: int) -> None:
    ts = [f["t"] for f in frames[:t + 1]]
    ax.set_facecolor(PANEL)
    ax.plot(ts, [f["counters"]["done_tasks"] for f in frames[:t + 1]],
            color="#3fb950", lw=1.4, label="tasks done")
    ax.plot(ts, [f["counters"]["altered_so_far"] for f in frames[:t + 1]],
            color="#ffd54f", lw=1.2, label="altered")
    ax.plot(ts, [f["counters"]["messages_so_far"] for f in frames[:t + 1]],
            color="#ff8a65", lw=1.0, label="messages")
    ax.tick_params(colors=FG, labelsize=6)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.legend(fontsize=6, facecolor=PANEL, labelcolor=FG, framealpha=0.6,
              loc="upper left")


def layout_overview(fig, world: World, frames: Sequence[dict], t: int, *,
                    title: Optional[str] = None, show_future: bool = True,
                    show_waiting: bool = True, show_messages: bool = True,
                    show_message_labels: bool = True,
                    show_taskbar: bool = True,
                    show_dialogue: bool = False) -> dict:
    """Draw the SPEC 11 panel of tick ``t`` onto the *existing* figure ``fig``.

    Returns the axes by name (``grid``/``metrics``/``log``/``line``/``taskbar``).
    The animation uses this directly (``fig.clf()`` then re-layout), so the panel
    code is shared by the GIF, the MP4 and the PNG screenshots.  The bottom
    strip is the per-agent *taskboard* (what every robot is doing right now);
    the log panel shows the event log or, with ``show_dialogue``, the protocol
    dialogue.
    """
    fig.set_facecolor(BG)
    frame = frame_at(frames, t)
    gs = fig.add_gridspec(4, 3, width_ratios=(2.4, 1.0, 1.0),
                          height_ratios=(1.0, 1.2, 1.1, 0.5), hspace=0.25,
                          wspace=0.15)
    ax_grid = fig.add_subplot(gs[0:3, 0:2])
    ax_m = fig.add_subplot(gs[0, 2])
    ax_log = fig.add_subplot(gs[1, 2])
    ax_line = fig.add_subplot(gs[2, 2])
    ax_task = fig.add_subplot(gs[3, :]) if show_taskbar else None

    grid_ax(ax_grid, world, title or
             f"warehouse  {world.H}x{world.W}   t={frame['t']}")
    draw_tasks(ax_grid, frame)
    draw_blocked(ax_grid, frame)
    draw_agents(ax_grid, frame, show_future=show_future,
                show_waiting=show_waiting, show_messages=show_messages,
                show_message_labels=show_message_labels)

    ax_m.set_facecolor(PANEL)
    ax_m.axis("off")
    ax_m.text(0.02, 0.98, _metrics_text(frame), color=FG, fontsize=9,
              va="top", family="monospace", transform=ax_m.transAxes)
    handles = [plt.Line2D([], [], marker="o", ls="none", mfc="#39424a",
                          mec=s["ec"], mew=s["lw"], label=name)
               for name, s in CAT_STYLE.items()]
    ax_m.legend(handles=handles, fontsize=6, loc="lower left", ncol=2,
                facecolor=PANEL, labelcolor=FG, framealpha=0.6)

    ax_log.set_facecolor(PANEL)
    ax_log.axis("off")
    ax_log.set_title("protocol dialogue" if show_dialogue else "event log",
                     color=FG, fontsize=8, loc="left")
    ax_log.text(0.02, 0.95,
                dialogue_lines(frames, frame["t"]) if show_dialogue
                else _log_text(frames, frame["t"]),
                color=FG, fontsize=7, va="top", family="monospace",
                transform=ax_log.transAxes)

    _cumulative(ax_line, frames, frame["t"])
    ax_line.set_title("cumulative", color=FG, fontsize=8, loc="left")
    axes = {"grid": ax_grid, "metrics": ax_m, "log": ax_log, "line": ax_line}
    if ax_task is not None:
        draw_taskbar(ax_task, frame)
        axes["taskbar"] = ax_task
    return axes


def plot_overview(world: World, frames: Sequence[dict], t: int, *,
                  figsize: Tuple[float, float] = (16, 9),
                  title: Optional[str] = None, **kw) -> "plt.Figure":
    """The base panel: grid + metrics + log + cumulative plot + taskboard.

    ``**kw`` is forwarded to :func:`layout_overview` (``show_taskbar``,
    ``show_dialogue``, ``show_waiting``, ``show_future``, ``show_messages``,
    ``show_message_labels``).
    """
    fig = plt.figure(figsize=figsize)
    layout_overview(fig, world, frames, t, title=title, **kw)
    return fig


def interactive(frames: Sequence[dict], world: World, *,
                figsize: Tuple[float, float] = (16, 9), fps: int = 8,
                title: Optional[str] = None):
    """Keyboard-driven playback of the panel (needs a GUI backend).

    Keys: space play/pause, ``left``/``right`` step, ``p`` future paths,
    ``g`` waiting arrows, ``m`` message arrows, ``c`` message bubbles,
    ``d`` taskboard, ``l`` event log <-> protocol dialogue, ``s`` save a PNG.
    """
    from matplotlib.animation import FuncAnimation

    st = {"t": 0, "play": True, "future": True, "waiting": True,
          "messages": True, "msgtext": True, "taskbar": True,
          "dialogue": False}
    fig = plt.figure(figsize=figsize)

    def draw(_t=None):
        layout_overview(fig, world, frames, st["t"], title=title,
                        show_future=st["future"], show_waiting=st["waiting"],
                        show_messages=st["messages"],
                        show_message_labels=st["msgtext"],
                        show_taskbar=st["taskbar"],
                        show_dialogue=st["dialogue"])
        return []

    def gen():
        while True:
            if st["play"]:
                st["t"] = (st["t"] + 1) % len(frames)
            yield st["t"]

    def on_key(ev):
        k = ev.key
        if k == " ":
            st["play"] = not st["play"]
        elif k == "right":
            st["play"] = False
            st["t"] = min(len(frames) - 1, st["t"] + 1)
        elif k == "left":
            st["play"] = False
            st["t"] = max(0, st["t"] - 1)
        elif k == "p":
            st["future"] = not st["future"]
        elif k == "g":
            st["waiting"] = not st["waiting"]
        elif k == "m":
            st["messages"] = not st["messages"]
        elif k == "c":
            st["msgtext"] = not st["msgtext"]
        elif k == "d":
            st["taskbar"] = not st["taskbar"]
        elif k == "l":
            st["dialogue"] = not st["dialogue"]
        elif k == "s":
            fig.savefig(f"frame_{st['t']:04d}.png", dpi=100, facecolor=BG)
        fig.canvas.draw_idle()
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    anim = FuncAnimation(fig, draw, frames=gen, interval=1000 / max(1, fps),
                         cache_frame_data=False)
    plt.show()
    return anim


def render_grid_frame(world: World, frame: dict, *,
                      figsize: Tuple[float, float] = (8, 8),
                      title: Optional[str] = None,
                      show_message_labels: bool = True) -> "plt.Figure":
    """A single grid panel (used by the freeze-frames of a repair pass)."""
    fig = plt.figure(figsize=figsize, facecolor=BG)
    ax = fig.add_subplot(111)
    grid_ax(ax, world, title or f"t={frame['t']}")
    draw_tasks(ax, frame)
    draw_blocked(ax, frame)
    draw_agents(ax, frame, show_message_labels=show_message_labels)
    fig.tight_layout()
    return fig



# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _animation(frames: Sequence[dict], world: World, figsize, title, fps):
    """A ``FuncAnimation`` that re-lays out the panel on a persistent fig."""
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=figsize)

    def draw(t):
        fig.clf()
        layout_overview(fig, world, frames, t, title=title)
        return []

    return fig, FuncAnimation(fig, draw, frames=list(range(len(frames))),
                              interval=1000 / max(1, fps))


def render_gif(frames: Sequence[dict], world: World, path: str, *,
               fps: int = 8, figsize: Tuple[float, float] = (16, 9),
               title: Optional[str] = None, dpi: int = 100,
               max_frames: Optional[int] = None) -> str:
    """Write the animation as a GIF (PillowWriter); returns ``path``.

    ``max_frames`` subsamples very long schedules (every k-th tick) so the GIF
    stays small.
    """
    _ensure_dir(path)
    use = frames
    if max_frames and len(frames) > max_frames:
        step = len(frames) // max_frames + 1
        use = list(frames)[::step]
    fig, anim = _animation(use, world, figsize, title, fps)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def render_mp4(frames: Sequence[dict], world: World, path: str, *,
               fps: int = 8, figsize: Tuple[float, float] = (16, 9),
               title: Optional[str] = None, dpi: int = 100) -> str:
    """Write the animation as an MP4 (needs ``ffmpeg``); returns ``path``."""
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("ffmpeg is not available: cannot write MP4")
    _ensure_dir(path)
    fig, anim = _animation(frames, world, figsize, title, fps)
    anim.save(path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def screenshots(frames: Sequence[dict], world: World, outdir: str, *,
                ticks: Optional[Sequence[int]] = None, prefix: str = "frame",
                dpi: int = 100) -> List[str]:
    """PNG snapshots (``t=0``, before/at/after a disruption, final)."""
    os.makedirs(outdir, exist_ok=True)
    if ticks is None:
        starts = sorted({ev["t0"] for f in frames for ev in f["events"]})
        ticks = ([0] + [max(0, t - 1) for t in starts] + starts
                 + [len(frames) - 1])
    out: List[str] = []
    for t in sorted({int(x) for x in ticks if 0 <= x < len(frames)}):
        fig = plot_overview(world, frames, t)
        path = os.path.join(outdir, f"{prefix}_{t:04d}.png")
        fig.savefig(path, dpi=dpi, facecolor=BG)
        plt.close(fig)
        out.append(path)
    return out




def gantt(plans: Dict[int, Optional[List[PlanStep]]], *,
          t_max: Optional[int] = None,
          categories: Optional[Dict[int, str]] = None,
          t_freeze: Optional[int] = None,
          title: str = "Gantt (plan execution)",
          figsize: Tuple[float, float] = (16, 9)) -> "plt.Figure":
    """One row per agent; bars are run-length-encoded steps by action."""
    return gantt_panels([(title, plans)], t_max=t_max, categories=categories,
                        t_freeze=t_freeze, figsize=figsize)


def gantt_panels(panels: Sequence[Tuple[str, Dict[int, List[PlanStep]]]], *,
                 t_max: Optional[int] = None,
                 categories: Optional[Dict[int, str]] = None,
                 t_freeze: Optional[int] = None,
                 figsize: Tuple[float, float] = (16, 9)) -> "plt.Figure":
    """Gantt of one or more plan maps, stacked top to bottom (before/after)."""
    ids = sorted({a for _, plans in panels for a in plans})
    if t_max is None:
        t_max = max([s[-1].t for _, plans in panels for s in plans.values()
                     if s] or [1])
    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs = fig.add_gridspec(len(panels), 1, hspace=0.35)
    for k, (name, plans) in enumerate(panels):
        ax = fig.add_subplot(gs[k, 0])
        ax.set_facecolor(PANEL)
        for row, aid in enumerate(ids):
            ax.plot([0, t_max], [row, row], color=GRID_C, lw=0.6, zorder=0)
            for key, t0, t1 in _runs(plans.get(aid)):
                _gantt_bar(ax, row, t0, t1, key)
            cat = (categories or {}).get(aid, UNCHANGED)
            if cat != UNCHANGED:
                ax.plot([-1.5], [row], marker="s", ms=5, clip_on=False,
                        color=CAT_STYLE.get(cat, CAT_STYLE[UNCHANGED])["ec"])
        if t_freeze is not None:
            ax.axvline(t_freeze, color="#ff5c4d", lw=1.2, ls="--", alpha=0.8)
            ax.text(t_freeze, len(ids) - 0.2, " freeze", color="#ff5c4d",
                    fontsize=6, rotation=90, va="top")
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels([f"a{a}" for a in ids], color=FG, fontsize=7)
        ax.set_xlim(0, t_max)
        ax.set_ylim(-0.8, len(ids) - 0.2)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(GRID_C)
        ax.set_title(name, color=FG, fontsize=9, loc="left")
    fig.axes[-1].set_xlabel("tick", color=FG, fontsize=8)
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec="none", label=a)
               for a, c in ACTION_C.items()]
    fig.axes[0].legend(handles=handles, fontsize=7, facecolor=PANEL,
                       labelcolor=FG, framealpha=0.6, ncol=4,
                       loc="upper right")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.08)
    return fig


def _runs(steps: Optional[Sequence[PlanStep]]) -> List[Tuple[tuple, int, int]]:
    """Run-length encode ``steps`` into ``((action, item), t0, t1)`` runs."""
    out: List[Tuple[tuple, int, int]] = []
    for s in steps or ():
        key = (s.action, s.item)
        if out and out[-1][0] == key and out[-1][2] == s.t - 1:
            out[-1] = (key, out[-1][1], s.t)
        else:
            out.append((key, s.t, s.t))
    return out


def _gantt_bar(ax, row: int, t0: int, t1: int, key) -> None:
    col = ACTION_C.get(key[0], "#8b949e")
    ax.add_patch(Rectangle((t0 - 0.5, row - 0.32), max(1, t1 - t0 + 1), 0.64,
                           fc=col, ec="none", alpha=0.9, zorder=3))


def before_after_panels(world: World, before: Sequence[dict],
                        after: Sequence[dict], path: str, *, t: int = 0,
                        figsize: Tuple[float, float] = (16, 9),
                        dpi: int = 100) -> str:
    """Side-by-side grid panels (pre- vs post-repair) written to ``path``."""
    fig = plt.figure(figsize=figsize, facecolor=BG)
    for k, (name, frames) in enumerate((("before repair", before),
                                        ("after repair", after))):
        ax = fig.add_subplot(1, 2, k + 1)
        frame = frame_at(frames, t)
        grid_ax(ax, world, f"{name}   t={frame['t']}")
        draw_tasks(ax, frame)
        draw_agents(ax, frame, show_waiting=False)
    fig.tight_layout()
    _ensure_dir(path)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return path
