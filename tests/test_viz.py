"""Base visualisation and plan-space views (``SPEC.md`` sections 11, 14 M4/M6).

The visualisation must work headless (``Agg``), so the tests need no display:
they check the *trace schema* of :func:`viz.build_frames` (the SPEC 11 frame
fields), the counters it reports, and that every export really writes a file
(GIF, PNG screenshots, Gantt, POP graph, repair cards, freeze-frames).
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.colors import to_rgba

import viz
import viz_plan
from config import Config
from disruptions import (AccidentEvent, EmergencyEvent, postpone_blockages,
                         random_blockages)
from negotiation import (ACCEPT, COMMIT, PROPOSE_ORDER, PROPOSE_REROUTE,
                         Negotiation)
from peg import peg_solve
from pocl import pocl
from scenarios import choke_showcase
from sim import simulate_scenario
from world import TaskStatus, assign_tasks, build_world


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _scenario(seed: int = 3, n: int = 3, k: int = 1, hw: int = 14):
    cfg = Config(H=hw, W=hw, n_agents=n, tasks_per_agent=k, seed=seed)
    rng = np.random.default_rng(cfg.seed)
    world, tasks, parking = build_world(cfg, rng)
    per_agent = assign_tasks(tasks, parking, rng)
    plans, failed = peg_solve(world, per_agent, parking, max_nodes=20_000)
    assert not failed, f"seed {seed}: FAILED_INIT {sorted(failed)}"
    return cfg, world, tasks, parking, per_agent, plans


def _repaired(seed: int = 3, **kw):
    """A scenario with blockages, an accident and an emergency, repaired."""
    cfg, world, tasks, parking, per_agent, plans = _scenario(seed=seed, **kw)
    rng = np.random.default_rng(cfg.seed + 5)
    block = postpone_blockages(plans, random_blockages(world, 2, rng, t_max=10,
                                                       avoid=parking))
    acc = [AccidentEvent(0, 4, 4)]
    emg = [EmergencyEvent(1, 5, 3)]
    sim, records, final = simulate_scenario(
        world, {a: list(s) for a, s in plans.items()}, per_agent, parking,
        mode="negotiate", blockages=list(block), accidents=acc,
        emergencies=emg, total_tasks=cfg.n_tasks, comm_radius=cfg.comm_radius,
        record_traces=True)
    events = list(block) + acc + emg
    return cfg, world, tasks, parking, plans, sim, records, final, events


# ----------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------
def test_frames_follow_the_spec_11_schema():
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    frames = viz.build_frames(world, final, tasks, parking, prev=plans,
                              t_freeze=4, events=events, records=records,
                              traces=sim.traces)

    assert len(frames) == sim.makespan + 1
    assert sorted(frames[0]) == sorted(
        ["t", "agents", "blocked", "tasks", "events", "messages",
         "waiting_for", "categories", "repair_log", "counters"])
    ag = frames[0]["agents"][0]
    assert sorted(ag) == sorted(["id", "pos", "state", "carrying_task_id",
                                 "plan_future", "next", "task",
                                 "altered_flash", "broken"])
    counters = frames[-1]["counters"]
    assert counters["done_tasks"] == cfg.n_tasks
    assert counters["total_tasks"] == cfg.n_tasks
    assert counters["disruptions_so_far"] == len(events)
    assert frames[0]["counters"]["disruptions_so_far"] == 0


def test_frames_show_blocked_cells_and_altered_categories():
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    disrupted = sorted({a for r in records for a in r.altered_plan_ids})
    frames = viz.build_frames(world, final, tasks, parking, prev=plans,
                              t_freeze=4, disrupted=disrupted, events=events,
                              records=records, traces=sim.traces)
    windows = [f for f in frames if f["blocked"]]
    assert windows, "no frame shows a blocked cell"
    assert any(b["until"] > f["t"] for f in windows for b in f["blocked"])
    altered = {a for f in frames for a, c in f["categories"].items()
               if c != "unchanged"}
    assert altered <= set(disrupted)
    announced = sorted({ev["t0"] for f in frames for ev in f["events"]})
    assert announced == sorted({viz._event_tick(e) for e in events})


def test_frames_use_the_negotiation_log_for_arrows():
    log = Negotiation(t=6)
    log.propose_order(0, 1, "before")
    log.accept(1, 0, "delay=1")
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    frames = viz.build_frames(world, final, tasks, parking, prev=plans,
                              t_freeze=4, events=events, records=records,
                              log=log, traces=sim.traces)
    assert frames[6]["messages"], "the arrows of tick 6 are missing"
    kinds = {m["kind"] for f in frames for m in f["messages"]}
    assert kinds == {PROPOSE_ORDER, ACCEPT}
    assert frames[6]["counters"]["messages_so_far"] == 2
    assert len(frames[8]["messages"]) == 2      # arrows persist a few ticks
    assert frames[-1]["counters"]["messages_so_far"] == 2


def test_carrying_item_tracks_pick_and_drop():
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    frames = viz.build_frames(world, final, tasks, parking,
                              traces=sim.traces)
    carried = [ag["carrying_task_id"] for f in frames for ag in f["agents"]]
    assert any(c is not None for c in carried)
    assert all(c is None or c in {t.id for t in tasks} for c in carried)



# ----------------------------------------------------------------------
# exports
# ----------------------------------------------------------------------
def test_gif_screenshots_and_gantt_are_written(tmp_path):
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    frames = viz.build_frames(world, final, tasks, parking, prev=plans,
                              t_freeze=4, events=events, records=records,
                              traces=sim.traces)
    gif = str(tmp_path / "anim.gif")
    viz.render_gif(frames, world, gif, fps=6, max_frames=25)
    assert os.path.getsize(gif) > 0
    with open(gif, "rb") as fh:
        assert fh.read(4) == b"GIF8"

    shots = viz.screenshots(frames, world, str(tmp_path / "shots"))
    assert len(shots) >= 3
    assert all(os.path.getsize(p) > 0 for p in shots)

    fig = viz.gantt(final, categories=frames[-1]["categories"], t_freeze=4)
    out = str(tmp_path / "gantt.png")
    fig.savefig(out)
    assert os.path.getsize(out) > 0

    ba = viz.before_after_panels(world, frames, frames,
                                 str(tmp_path / "ba.png"), t=4)
    assert os.path.getsize(ba) > 0


def test_mp4_export_when_ffmpeg_is_present(tmp_path):
    from matplotlib.animation import FFMpegWriter
    if not FFMpegWriter.isAvailable():
        return                                    # ffmpeg is optional
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    frames = viz.build_frames(world, final, tasks, parking,
                              traces=sim.traces, horizon=12)
    out = str(tmp_path / "anim.mp4")
    viz.render_mp4(frames, world, out, fps=6)
    assert os.path.getsize(out) > 0



# ----------------------------------------------------------------------
# plan-space views
# ----------------------------------------------------------------------
def test_pop_figure_and_panel(tmp_path):
    cfg, world, tasks, parking, per_agent, plans = _scenario()
    pop = pocl(0, list(per_agent[0]), parking[0], world, max_nodes=20_000)
    assert pop is not None
    fig = viz_plan.pop_figure(pop, title="agent 0")
    out = str(tmp_path / "pop.png")
    fig.savefig(out)
    assert os.path.getsize(out) > 0
    panel = viz_plan.pop_panel(world, pop, agent=0)
    out2 = str(tmp_path / "pop_panel.png")
    panel.savefig(out2)
    assert os.path.getsize(out2) > 0


def test_ladder_figure_and_repair_cards(tmp_path):
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    assert records, "the scenario must trigger at least one repair"
    fig = viz_plan.ladder_figure([r.level_resolved for r in records])
    out = str(tmp_path / "ladder.png")
    fig.savefig(out)
    assert os.path.getsize(out) > 0

    cards = viz_plan.repair_cards(records, world=world, tasks=tasks,
                                  parking=parking, prev=plans, new=final,
                                  outdir=str(tmp_path / "cards"))
    assert cards
    assert all(os.path.getsize(p) > 0 for p in cards)

    # the "after repair" panel of a card explains the grid symbols too
    import matplotlib.pyplot as plt
    fig = viz_plan.repair_card(records[0], world=world, tasks=tasks,
                               parking=parking, prev=plans, new=final)
    assert [tx.get_text() for tx in fig.axes[-1].get_legend().get_texts()] == \
        [lbl for _, lbl in viz.GRID_SYMBOLS]
    plt.close(fig)

    frames = viz.build_frames(world, final, tasks, parking, prev=plans,
                              t_freeze=records[0].t, events=events,
                              records=records, traces=sim.traces)
    phases = viz_plan.phase_frames(records[0], world=world, frames=frames,
                                   t=records[0].t,
                                   outdir=str(tmp_path / "phases"))
    assert len(phases) == len(viz_plan.PHASES) == 5
    assert all(os.path.getsize(p) > 0 for p in phases)


def test_gantt_before_after_and_categories(tmp_path):
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    cats = viz_plan.categories_of(plans, final, records[0].t)
    fig = viz_plan.gantt_before_after(plans, final, t_freeze=records[0].t,
                                      categories=cats)
    out = str(tmp_path / "gantt2.png")
    fig.savefig(out)
    assert os.path.getsize(out) > 0


# ----------------------------------------------------------------------
# taskboard + protocol dialogue + sequence diagram
# ----------------------------------------------------------------------
def test_taskboard_fields_and_panel(tmp_path):
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    frames = viz.build_frames(world, final, tasks, parking, prev=plans,
                              t_freeze=4, events=events, records=records,
                              traces=sim.traces)
    for ag in frames[0]["agents"]:
        act = ag["task"]
        assert sorted(act) == sorted(["task_id", "phase", "target", "done",
                                      "total"])
        assert act["phase"] in viz.PHASE_ABBR
    phases = {ag["task"]["phase"] for f in frames for ag in f["agents"]}
    assert phases & {"TO_PICKUP", "TO_DELIVER"}       # work is under way
    assert "DONE" in phases                           # and it finishes

    import matplotlib.pyplot as plt
    fig = plt.figure()
    axes = viz.layout_overview(fig, world, frames, len(frames) // 2)
    assert "taskbar" in axes
    out = str(tmp_path / "panel.png")
    fig.savefig(out)
    plt.close(fig)
    assert os.path.getsize(out) > 0

    fig2 = plt.figure()
    assert "taskbar" not in viz.layout_overview(fig2, world, frames, 3,
                                                show_taskbar=False)
    plt.close(fig2)


def test_dialogue_panel_shows_speech_acts():
    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    log = Negotiation(t=6)
    log.propose_order(0, 1, "before")
    log.accept(1, 0, "delay=1")
    frames = viz.build_frames(world, final, tasks, parking, prev=plans,
                              t_freeze=4, events=events, records=records,
                              log=log, traces=sim.traces)
    text = viz.dialogue_lines(frames, 6)
    assert PROPOSE_ORDER in text and ACCEPT in text
    assert "a0->a1" in text and "a1->a0" in text
    assert "delay=1" in text                          # the detail is kept
    fig = viz.plot_overview(world, frames, 6, show_dialogue=True)
    assert fig.axes, "the dialogue-panel figure is empty"


def test_sequence_figure(tmp_path):
    log = Negotiation(t=6)
    log.propose_order(0, 1, "before")
    log.accept(1, 0, "delay=1")
    log.commit(0, [1], "the initiator goes first")
    fig = viz_plan.sequence_figure(log)
    out = str(tmp_path / "seq.png")
    fig.savefig(out)
    assert os.path.getsize(out) > 0

    empty = viz_plan.sequence_figure(Negotiation(t=0))
    out2 = str(tmp_path / "seq_empty.png")
    empty.savefig(out2)
    assert os.path.getsize(out2) > 0          # a silent pass still renders


# ----------------------------------------------------------------------
# message bubbles
# ----------------------------------------------------------------------
def _choked(seed: int = 0, n: int = 4):
    """The corridor showcase, repaired: its single door forces negotiation."""
    sc = choke_showcase(n, seed=seed)
    log = Negotiation(t=0)
    sim, records, final = simulate_scenario(
        sc["world"], {a: list(s) for a, s in sc["plans"].items()},
        sc["per_agent"], sc["parking"], mode="negotiate",
        accidents=list(sc["accidents"]), total_tasks=sc["total_tasks"],
        message_log=log, record_traces=True)
    tasks = [tk for lst in sc["per_agent"] for tk in lst]
    frames = viz.build_frames(
        sc["world"], final, tasks, sc["parking"], prev=sc["plans"],
        t_freeze=records[0].t if records else 0,
        disrupted=sorted({a for r in records for a in r.altered_plan_ids}),
        events=list(sc["accidents"]), records=records, log=log,
        traces=sim.traces)
    return sc, log, records, frames


def test_message_labels_cover_both_endpoints():
    frame = {"messages": [
        {"t": 2, "kind": PROPOSE_REROUTE, "sender": 1, "receiver": 0,
         "detail": "grant (3,3)"},
        {"t": 3, "kind": ACCEPT, "sender": 0, "receiver": 1, "detail": "ok"}]}
    labels = viz.agent_message_labels(frame)
    assert set(labels) == {0, 1}, "the sender AND the receiver get a bubble"
    assert [x["role"] for x in labels[1]] == ["out", "in"]
    assert [x["role"] for x in labels[0]] == ["in", "out"]
    assert labels[0][0]["text"] == viz.MSG_ABBR[PROPOSE_REROUTE]
    assert labels[0][0]["glyph"] == viz.MSG_GLYPH_IN
    assert labels[1][0]["glyph"] == viz.MSG_GLYPH_OUT


def test_message_labels_are_empty_for_a_silent_repair():
    assert viz.agent_message_labels({"messages": []}) == {}
    assert viz.agent_message_labels({}) == {}


def test_message_labels_are_capped_per_agent():
    kinds = (PROPOSE_ORDER, ACCEPT, COMMIT, PROPOSE_REROUTE)
    frame = {"messages": [{"t": i, "kind": k, "sender": 0, "receiver": 1,
                           "detail": ""} for i, k in enumerate(kinds)]}
    labels = viz.agent_message_labels(frame)
    assert len(labels[0]) == viz.MSG_LABELS_MAX
    assert labels[0][-1]["text"].startswith("+")     # the surplus folds into +N
    assert labels[0][0]["kind"] == PROPOSE_ORDER     # order is preserved


def test_message_bubbles_are_drawn_and_can_be_switched_off():
    import matplotlib.pyplot as plt

    sc, log, records, frames = _choked()
    assert log.messages, "the choke showcase must speak"
    assert PROPOSE_REROUTE in {m.kind for m in log.messages}
    t = min(m.t for m in log.messages)
    assert viz.agent_message_labels(frames[t]), "silent tick at t={t}"

    fig, ax = plt.subplots()
    viz.draw_agents(ax, frames[t])
    bubbles = [tx for tx in ax.texts if tx.get_bbox_patch() is not None]
    assert len(bubbles) >= 2, "sender and receiver each show a bubble"
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    viz.draw_agents(ax2, frames[t], show_message_labels=False)
    assert not [tx for tx in ax2.texts if tx.get_bbox_patch() is not None]

    # the panel forwards the flag, and the arrows survive either way
    assert viz.plot_overview(sc["world"], frames, t,
                             show_message_labels=False).axes
    plt.close("all")

# ----------------------------------------------------------------------
# the grid symbol vocabulary (the legend that explains the GIF/PNG)
# ----------------------------------------------------------------------
def test_grid_symbol_handles_cover_the_vocabulary():
    kinds = [kind for kind, _ in viz.GRID_SYMBOLS]
    assert kinds == ["shelf", "agent", "tote", "breakdown", "pickup",
                     "deliver_pending", "deliver_picked", "deliver_done",
                     "blockage", "waiting", "message"]
    handles = viz.grid_symbol_handles()
    assert len(handles) == len(viz.GRID_SYMBOLS)
    assert [h.get_label() for h in handles] == [lbl for _, lbl in
                                                viz.GRID_SYMBOLS]
    by_kind = dict(zip(kinds, handles))
    # the pickup glyph is the hollow diamond, the delivery glyph the star
    assert by_kind["pickup"].get_marker() == "D"
    assert by_kind["deliver_done"].get_marker() == "*"
    assert by_kind["breakdown"].get_marker() == "X"
    # and their colours come from the same tables the drawing code uses
    assert to_rgba(by_kind["pickup"].get_markeredgecolor()) == \
        to_rgba(viz.PICKUP_C)
    for kind, status in (("deliver_pending", TaskStatus.PENDING),
                         ("deliver_picked", TaskStatus.PICKED),
                         ("deliver_done", TaskStatus.DONE)):
        assert to_rgba(by_kind[kind].get_markerfacecolor()) == \
            to_rgba(viz.TASK_STATUS_C[status.value])


def test_draw_tasks_uses_the_shared_status_table():
    import matplotlib.pyplot as plt

    frame = {"tasks": [
        {"task_id": i, "status": st.value, "pickup": (i, 1),
         "delivery": (i, 2)}
        for i, st in enumerate((TaskStatus.PENDING, TaskStatus.PICKED,
                                TaskStatus.DONE))]}
    fig, ax = plt.subplots()
    viz.draw_tasks(ax, frame)
    stars = [ln for ln in ax.get_lines() if ln.get_marker() == "*"]
    diamonds = [ln for ln in ax.get_lines() if ln.get_marker() == "D"]
    assert len(stars) == len(diamonds) == 3
    assert [to_rgba(ln.get_markerfacecolor()) for ln in stars] == [
        to_rgba(viz.TASK_STATUS_C[st.value])
        for st in (TaskStatus.PENDING, TaskStatus.PICKED, TaskStatus.DONE)]
    assert all(to_rgba(ln.get_markerfacecolor()) == to_rgba("none")
               for ln in diamonds), "the pickup diamond stays hollow"
    plt.close(fig)


def test_panel_carries_the_symbol_legend(tmp_path):
    import matplotlib.pyplot as plt

    (cfg, world, tasks, parking, plans, sim, records, final,
     events) = _repaired()
    frames = viz.build_frames(world, final, tasks, parking, prev=plans,
                              t_freeze=4, events=events, records=records,
                              traces=sim.traces)
    t = len(frames) - 1

    fig = plt.figure()
    axes = viz.layout_overview(fig, world, frames, t)
    leg = axes["grid"].get_legend()
    assert leg is not None, "the grid must explain its own symbols"
    assert [tx.get_text() for tx in leg.get_texts()] == \
        [lbl for _, lbl in viz.GRID_SYMBOLS]
    star_cols = {to_rgba(ln.get_markerfacecolor()) for ln in
                 axes["grid"].get_lines() if ln.get_marker() == "*"}
    assert star_cols <= {to_rgba(c) for c in viz.TASK_STATUS_C.values()}
    out = str(tmp_path / "panel.png")
    fig.savefig(out)
    plt.close(fig)
    assert os.path.getsize(out) > 0

    fig2 = plt.figure()
    axes2 = viz.layout_overview(fig2, world, frames, t,
                                show_symbol_legend=False)
    assert axes2["grid"].get_legend() is None
    plt.close(fig2)

    # the single-grid export and the before/after view explain them too
    fig3 = viz.render_grid_frame(world, frames[t])
    assert fig3.axes[0].get_legend() is not None
    plt.close(fig3)
    fig4 = viz.render_grid_frame(world, frames[t], show_symbol_legend=False)
    assert fig4.axes[0].get_legend() is None
    plt.close(fig4)
    ba = viz.before_after_panels(world, frames, frames,
                                 str(tmp_path / "ba.png"), t=t)
    assert os.path.getsize(ba) > 0
    plt.close("all")

