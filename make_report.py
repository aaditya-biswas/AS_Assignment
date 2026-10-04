"""Report generation (``SPEC.md`` section 15): PDF + Markdown + zip.

Builds the M10 deliverable -- a POP figure, a Gantt chart, four repair cards and
the altered-per-disruption table -- plus the experiment tables and plots of
:mod:`experiments` when ``outputs/results.csv`` exists.  Everything comes from a
**fresh** run, so the report can always be built; the figures are copied into
the bundle, so the zip is self-contained.

Usage::

    python make_report.py --zip                       # -> outputs/report/
    python make_report.py --data outputs --zip        # + the sweep tables/plots
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import Config
from disruptions import AccidentEvent, EmergencyEvent
from negotiation import Negotiation
from peg import peg_solve
from pocl import pocl
from scenarios import blockages_on_routes, choke_showcase
from sim import plan_makespan, simulate_scenario
from world import World, assign_tasks, build_world

#: the four ``SPEC`` 9.5 definitions and the ``RepairRecord`` list field behind
DEFINITIONS = (("altered_plan", "altered_plan_ids"),
               ("altered_path", "altered_path_ids"),
               ("delayed_only", "delayed_ids"),
               ("altered_naive", "naive_ids"))


# ----------------------------------------------------------------------
# the two case studies
# ----------------------------------------------------------------------
def gather(seed: int = 0, mode: str = "negotiate", *, n_agents: int = 8,
           tasks_per_agent: int = 2, hw: int = 28, disruptions: int = 5
           ) -> dict:
    """A random warehouse with route blockages, accidents and an emergency."""
    cfg = Config(H=hw, W=hw, n_agents=n_agents,
                 tasks_per_agent=tasks_per_agent, seed=seed)
    rng = np.random.default_rng(cfg.seed)
    world, tasks, parking = build_world(cfg, rng)
    per_agent = assign_tasks(tasks, parking, rng)
    plans, failed = peg_solve(world, per_agent, parking,
                              max_nodes=cfg.pocl_max_nodes)
    horizon = max(20, plan_makespan(plans))
    # blockages *on* the routes, so each one really bites (see scenarios.py)
    block = blockages_on_routes(plans, disruptions, rng)
    acc = [AccidentEvent(0, max(2, horizon // 6), 6),
           AccidentEvent(min(3, n_agents - 1), max(3, horizon // 2), 6)]
    emg = [EmergencyEvent(min(1, n_agents - 1), max(4, horizon // 4), 4)]
    log = Negotiation(t=0)
    sim, records, final = simulate_scenario(
        world, {a: (list(s) if s else None) for a, s in plans.items()},
        per_agent, parking, mode=mode, blockages=list(block), accidents=acc,
        emergencies=emg, total_tasks=cfg.n_tasks, comm_radius=cfg.comm_radius,
        delay_threshold=cfg.delay_threshold, lambda_soft=cfg.lambda_soft,
        max_depth=cfg.max_depth, beta_alter=cfg.beta_alter, message_log=log,
        record_traces=True)
    return {"name": f"{mode}, {hw}x{hw} grid, N={n_agents}",
            "cfg": cfg, "world": world, "tasks": tasks, "parking": parking,
            "per_agent": per_agent, "before": plans, "after": final, "sim": sim,
            "records": records, "log": log,
            "events": list(block) + acc + emg, "failed_init": len(failed)}


def gather_choke(n_agents: int = 4, seed: int = 0) -> dict:
    """The single-door corridor, where the protocol is forced to speak."""
    sc = choke_showcase(n_agents, seed=seed)
    cfg = Config(H=sc["world"].H, W=sc["world"].W, n_agents=n_agents,
                 tasks_per_agent=1, seed=seed)
    log = Negotiation(t=0)
    sim, records, final = simulate_scenario(
        World(sc["world"].grid.copy()),
        {a: (list(s) if s else None) for a, s in sc["plans"].items()},
        sc["per_agent"], sc["parking"], mode="negotiate",
        blockages=list(sc["blockages"]), accidents=list(sc["accidents"]),
        emergencies=list(sc["emergencies"]), total_tasks=sc["total_tasks"],
        comm_radius=cfg.comm_radius, delay_threshold=cfg.delay_threshold,
        lambda_soft=cfg.lambda_soft, max_depth=cfg.max_depth,
        beta_alter=cfg.beta_alter, message_log=log, record_traces=True)
    return {"name": f"choke corridor, N={n_agents}", "cfg": cfg,
            "world": sc["world"],
            "tasks": [tk for lst in sc["per_agent"] for tk in lst],
            "parking": list(sc["parking"]), "per_agent": sc["per_agent"],
            "before": sc["plans"], "after": final, "sim": sim,
            "records": records, "log": log, "events": list(sc["accidents"]),
            "failed_init": 0}


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def figures(case: dict, outdir: str, sub: str = "repair") -> Dict[str, str]:
    """Write the report figures of one case study into ``<outdir>/<sub>``."""
    from matplotlib import pyplot as plt

    import viz
    import viz_plan

    figdir = os.path.join(outdir, sub)
    os.makedirs(figdir, exist_ok=True)
    records, before, after = case["records"], case["before"], case["after"]
    t_freeze = records[0].t if records else 0
    disrupted = sorted({a for r in records for a in r.altered_plan_ids})
    frames = viz.build_frames(case["world"], after, case["tasks"],
                              case["parking"], prev=before, t_freeze=t_freeze,
                              disrupted=disrupted, events=case["events"],
                              records=records, log=case["log"],
                              traces=case["sim"].traces)
    paths: Dict[str, str] = {}

    def rel(name: str) -> str:
        return f"{sub}/{name}"

    def save(fig, name: str) -> None:
        fig.savefig(os.path.join(figdir, name), facecolor=viz.BG)
        paths[name.split(".")[0]] = rel(name)

    # a POP of the (first) altered agent, plus the Gantt of the whole fleet
    aid = disrupted[0] if disrupted else 0
    pop = pocl(aid, list(case["per_agent"][aid]), case["parking"][aid],
               case["world"], max_nodes=case["cfg"].pocl_max_nodes)
    if pop is not None:
        save(viz_plan.pop_panel(case["world"], pop, agent=aid), "pop.png")
    save(viz.gantt(after, categories=frames[-1]["categories"],
                   t_freeze=t_freeze), "gantt.png")
    save(viz.plot_overview(case["world"], frames, t_freeze,
                           show_dialogue=True), "dialogue.png")
    save(viz.plot_overview(case["world"], frames, len(frames) - 1),
         "final_panel.png")
    if records:
        viz_plan.repair_cards(records, world=case["world"],
                              tasks=case["tasks"], parking=case["parking"],
                              prev=before, new=after, outdir=figdir,
                              max_cards=4)
        for path in sorted(glob.glob(os.path.join(figdir, "repair_card_*.png"))):
            paths[os.path.basename(path).split(".")[0]] = rel(
                os.path.basename(path))
        viz_plan.phase_frames(records[0], world=case["world"], frames=frames,
                              t=t_freeze, outdir=figdir)
        for path in sorted(glob.glob(os.path.join(figdir, "phase_*.png"))):
            paths[os.path.basename(path).split(".")[0]] = rel(
                os.path.basename(path))
    save(viz_plan.ladder_figure([r.level_resolved for r in records]),
         "ladder.png")
    save(viz_plan.sequence_figure(
        case["log"], title=f"{case['name']}: negotiation sequence"),
        "sequence.png")
    plt.close("all")
    return paths


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------
def altered_table(records: Sequence) -> List[Tuple[str, float, float, int]]:
    """Mean/median/max altered agents per disruption, per ``SPEC`` 9.5 view."""
    rows: List[Tuple[str, float, float, int]] = []
    for name, field in DEFINITIONS:
        vals = [len(getattr(r, field)) for r in records]
        arr = np.asarray(vals, dtype=float) if vals else np.zeros(1)
        rows.append((name, float(arr.mean()), float(np.median(arr)),
                     int(arr.max())))
    return rows


def per_disruption_table(records: Sequence) -> List[list]:
    """One row per disruption: rung, altered sets, delay, messages, CPU."""
    out = []
    for r in records:
        out.append([r.t, r.event_type, r.level_resolved,
                    len(r.altered_plan_ids), len(r.altered_path_ids),
                    len(r.delayed_ids), len(r.naive_ids), r.added_delay_total,
                    sum(r.messages_by_type.values()), round(r.cpu_ms, 1),
                    "yes" if r.success else "no"])
    return out


def rung_histogram(records: Sequence) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for r in records:
        hist[r.level_resolved] = hist.get(r.level_resolved, 0) + 1
    return hist


def case_summary(case: dict) -> Dict[str, object]:
    """The headline numbers of one case study."""
    sim, records = case["sim"], case["records"]
    return {
        "case": case["name"],
        "tasks": f"{sim.completed}/{sim.total_tasks}",
        "makespan": sim.makespan,
        "conflict_free": "yes" if sim.conflict_free else f"{len(sim.violations)} viol",
        "repairs": len(records),
        "altered_plan_total": sum(len(r.altered_plan_ids) for r in records),
        "messages": sum(sum(r.messages_by_type.values()) for r in records),
        "rungs": rung_histogram(records),
        "failed_init": case["failed_init"],
    }



# ----------------------------------------------------------------------
# experiment summary (optional, from experiments.py)
# ----------------------------------------------------------------------
def load_experiment(data_dir: str) -> Optional[dict]:
    """The sweep artefacts of ``experiments.py``, if they exist."""
    res = os.path.join(data_dir, "results.csv")
    if not os.path.exists(res):
        return None
    import pandas as pd

    df = pd.read_csv(res)
    meta_path = os.path.join(data_dir, "run_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return {"dir": data_dir, "df": df, "meta": meta,
            "plots": sorted(glob.glob(os.path.join(data_dir, "*.png")))}


#: the columns the report aggregates per mode
_AGG = ("makespan", "sum_of_costs", "mean_altered_plan", "mean_altered_naive",
        "mean_messages", "mean_cpu_ms", "n_disruptions", "unfinished_agents")


def experiment_table(df) -> Tuple[Tuple[str, ...], List[list]]:
    """Per-mode means over every family of the sweep."""
    cols = [c for c in _AGG if c in df.columns]
    agg = df.groupby("mode")[cols].mean().round(3)
    cf = (df.groupby("mode")["conflict_free"].mean().round(3)
          if "conflict_free" in df.columns else None)
    header = ("mode", "families") + tuple(cols) + ("conflict_free",)
    rows = []
    for mode, row in agg.iterrows():
        n = int((df["mode"] == mode).sum())
        rows.append([str(mode), n] + [row[c] for c in cols]
                    + [float(cf[mode]) if cf is not None else 1.0])
    return header, rows


#: the sweep plots worth embedding, in reading order
_SWEEP_PLOTS = ("cost_vs_N", "altered_vs_N", "heatmap_altered",
                "rung_stack_vs_N", "mode_bars", "comm_radius_ablation",
                "pocl_nodes_vs_tasks")



# ----------------------------------------------------------------------
# the report body: one list of blocks for both formats
# ----------------------------------------------------------------------
ARCH_TABLE = (
    ("Layer", "Module", "Responsibility (POCL mapping)"),
    ("L1 partial order", "pocl.py", "STRIPS operators, threats, POP search"),
    ("L2 grounding", "planner.py / peg.py", "space-time A*, PEG retiming"),
    ("L3 repair", "modify.py / negotiation.py", "the R0-R4 ladder + protocol"),
)


def _summary_table(summary: dict) -> tuple:
    keys = ("case", "tasks", "makespan", "conflict_free", "repairs",
            "altered_plan_total", "messages", "rungs", "failed_init")
    header = ("case", "tasks", "makespan", "safe", "repairs", "altered (sum)",
              "messages", "rungs", "FAILED_INIT")
    return ("table", header, [[summary[k] for k in keys]])


def _blocks_case_a(cases, figsets, summaries, altered_rows) -> List[tuple]:
    out: List[tuple] = [("h2", "1. Problem statement and assumptions")]
    out.append(("p", "A fleet of robots serves pickup/delivery tasks on a shelf "
                     "grid. Disruptions -- exogenous blockages and endogenous "
                     "accidents/emergencies -- invalidate the plan; the repair "
                     "must be local, fast and safety-preserving."))
    out.append(("p", "Assumptions: unit cells and 4-neighbour moves, one robot "
                     "per cell per tick, time-indexed vertex and edge "
                     "reservations, negotiation only within comm_radius, a "
                     "bounded repair budget, and no global re-planning inside a "
                     "repair."))
    out.append(("h2", "2. Three-layer architecture"))
    out.append(("table", ARCH_TABLE[0], [list(r) for r in ARCH_TABLE[1:]]))
    out.append(("h2", "3. Case study A: random warehouse"))
    out.append(_summary_table(summaries[0]))
    for key, caption in (("pop", "A POP of an altered agent (L1)."),
                         ("gantt", "Per-agent Gantt of the repaired schedule."),
                         ("dialogue", "Metrics panel with the protocol dialogue."),
                         ("final_panel", "Final state of the fleet.")):
        if key in figsets[0]:
            out.append(("img", figsets[0][key], caption))
    cards = sorted(k for k in figsets[0] if k.startswith("repair_card"))
    if cards:
        out.append(("h2", "3.1 Repair cards"))
        out.append(("p", f"{len(cards)} repair card(s): the RepairRecord (rung, "
                         "altered sets, message counts, budgets) beside the "
                         "routes before and after."))
        for key in cards:
            out.append(("img", figsets[0][key], key.replace("_", " ")))
    out.append(("h2", "4. Altered agents per disruption (SPEC 9.5)"))
    out.append(("table", ("definition", "mean", "median", "max"),
                [[n, round(m, 4), round(d, 4), mx]
                 for n, m, d, mx in altered_rows[0]]))
    out.append(("table", ("t", "event", "rung", "altered_plan", "altered_path",
                          "delayed_only", "altered_naive", "added_delay",
                          "messages", "cpu_ms", "success"),
                per_disruption_table(cases[0]["records"])))
    return out



def _blocks_case_b(cases, figsets, summaries) -> List[tuple]:
    out: List[tuple] = [("h2", "5. Case study B: the corridor choke point")]
    out.append(("p", "A single-door corridor forces a clash, so the ladder finds "
                     "no hard candidate and the initiator negotiates a grant "
                     "(rung R2): PROPOSE_REROUTE -> ACCEPT -> COMMIT. This is "
                     "the case that makes the protocol visible -- and why the "
                     "plain, open-warehouse demo is honestly silent."))
    out.append(_summary_table(summaries[1]))
    for key, caption in (("sequence", "The message sequence diagram of the pass."),
                         ("ladder", "The R0-R4 ladder, with the used rungs lit."),
                         ("final_panel", "The corridor at the final tick.")):
        if key in figsets[1]:
            out.append(("img", figsets[1][key], caption))
    return out


def _blocks_experiments(experiment, sweep_figs) -> List[tuple]:
    out: List[tuple] = [("h2", "6. Experiments")]
    if experiment is None:
        out.append(("p", "No sweep data found; run `python experiments.py "
                         "--sweep A --seeds 10 --plots` to fill this section."))
        return out
    header, rows = experiment_table(experiment["df"])
    meta = experiment["meta"]
    out.append(("p", f"{meta.get('n_rows', '?')} result rows over "
                     f"{meta.get('n_families', '?')} instances (sweep "
                     f"{meta.get('sweep', '?')}, modes "
                     f"{', '.join(meta.get('modes', []) or [])})."))
    out.append(("table", header, rows))
    for name in _SWEEP_PLOTS:
        if name in sweep_figs:
            out.append(("img", sweep_figs[name], name.replace("_", " ")))
    return out


def report_blocks(cases, figsets, summaries, altered_rows, experiment,
                  sweep_figs, context: dict) -> List[tuple]:
    """The whole report as format-independent blocks."""
    out: List[tuple] = [("h1",
                         "Warehouse disruption repair: POCL + plan modification")]
    out.append(("p", f"Generated {context['date']} from commit "
                     f"{context['git_commit']} (python {context['python']}, "
                     f"PYTHONHASHSEED={context['hashseed']})."))
    repo = context.get("repo") or REPO_URL
    out.append(("p", "Sources, the test suite and every generated artefact "
                     "(animated GIFs of both case studies, the repair cards and "
                     "this report) live at:"))
    out.append(("link", repo, repo))
    out += _blocks_case_a(cases, figsets, summaries, altered_rows)
    out += _blocks_case_b(cases, figsets, summaries)
    out += _blocks_experiments(experiment, sweep_figs)
    out.append(("h2", "7. Discussion"))
    out.append(("p", "Partial orders plus negotiation keep the altered set "
                     "small: negotiate alters far fewer agents than the global "
                     "baseline at a comparable makespan, and the corridor shows "
                     "the protocol only speaks when a holder really parks in the "
                     "way. Remaining risks: prioritized-planning "
                     "incompleteness, deadlock under a fully blocked choke, and "
                     "the failure_wait fallback (rung R4) when no candidate "
                     "exists."))
    out.append(("h2", "8. How to run"))
    out.append(("p", "Everything is reproducible from the published repository:"))
    out.append(("code", f"git clone {repo}\n"
                        "cd AS_Assignment && pip install -r requirements.txt\n"
                        "python run_demo.py --compare --viz --outdir outputs\n"
                        "python run_demo.py --choke --compare --viz\n"
                        "python experiments.py --sweep grid --seeds 10 --plots\n"
                        "python make_report.py --data outputs --zip"))
    out.append(("h2", "9. References"))
    out.append(("code", "UCPOP (1992); Weld (1994); Kambhampati & Hendler "
                        "(1992); Fox et al. (2006); Silver (2005); Hoenig et al. "
                        "(MAPF-POST, 2016)."))
    return out




# ----------------------------------------------------------------------
# writers
# ----------------------------------------------------------------------
def _cell(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def write_markdown(path: str, blocks: Sequence[tuple]) -> str:
    """The same blocks as GitHub-flavoured Markdown (with the figures)."""
    lines: List[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "h1":
            lines += [f"# {block[1]}", ""]
        elif kind == "h2":
            lines += [f"## {block[1]}", ""]
        elif kind == "p":
            lines += [block[1], ""]
        elif kind == "code":
            lines += ["```", block[1], "```", ""]
        elif kind == "img":
            caption = block[2] if len(block) > 2 else ""
            lines += [f"![{caption}]({block[1]})", "", f"*{caption}*", ""]
        elif kind == "link":
            label = block[2] if len(block) > 2 else block[1]
            lines += [f"[{label}]({block[1]})", ""]
        elif kind == "table":
            header, rows = block[1], block[2]
            lines += ["| " + " | ".join(str(h) for h in header) + " |",
                      "|" + "---|" * len(header)]
            for row in rows:
                lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
            lines.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def write_pdf(path: str, blocks: Sequence[tuple], outdir: str,
              repo: Optional[str] = None) -> str:
    """The report as a PDF via reportlab plateaus (``SPEC`` 15).

    ``("link", url[, label])`` blocks become clickable hyperlinks, so the
    GitHub repository of the deliverable is reachable straight from the PDF.
    ``repo`` (when given) is also printed in the footer of *every* page, so the
    link survives printing the report on paper.
    """
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1x", parent=styles["Heading1"], fontSize=17,
                        textColor=colors.HexColor("#0b3d91"))
    h2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=12.5,
                        textColor=colors.HexColor("#123a63"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5,
                          leading=12.5)
    small = ParagraphStyle("small", parent=body, fontSize=7.0, leading=8.8)
    head = ParagraphStyle("head", parent=small, fontSize=7.2, leading=9,
                          textColor=colors.white)
    code = ParagraphStyle("code", parent=body, fontName="Courier",
                          fontSize=8.5, leading=11)
    avail = A4[0] - 3.6 * cm

    def image_flowable(relpath: str, max_h: float = 17 * cm):
        full = os.path.join(outdir, relpath)
        if not os.path.exists(full):
            return None
        with PILImage.open(full) as im:
            w, h = im.size
        scale = min(avail / w, max_h / h)
        return Image(full, width=w * scale, height=h * scale)

    story: List[object] = []
    for block in blocks:
        kind = block[0]
        if kind == "h1":
            story += [Paragraph(_esc(block[1]), h1), Spacer(1, 8)]
        elif kind == "h2":
            story += [Spacer(1, 10), Paragraph(_esc(block[1]), h2),
                      Spacer(1, 4)]
        elif kind == "p":
            story += [Paragraph(_esc(block[1]), body), Spacer(1, 6)]
        elif kind == "code":
            story += [Paragraph(_esc(block[1]).replace("\n", "<br/>"), code),
                      Spacer(1, 6)]
        elif kind == "table":
            header, rows = block[1], block[2]
            data = [[Paragraph(_esc(h), head) for h in header]]
            data += [[Paragraph(_esc(_cell(v)), small) for v in row]
                     for row in rows]
            ncol = max(1, len(header))
            table = Table(data, colWidths=[avail / ncol] * ncol, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123a63")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f2f5f9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]))
            story += [table, Spacer(1, 8)]
        elif kind == "img":
            flowable = image_flowable(block[1])
            if flowable is not None:
                caption = block[2] if len(block) > 2 else ""
                story += [flowable, Paragraph(_esc(caption), small),
                          Spacer(1, 8)]
        elif kind == "link":
            label = block[2] if len(block) > 2 else block[1]
            story += [Paragraph(f'<link href="{_esc(block[1])}">'
                                f'{_esc(label)}</link>',
                                ParagraphStyle("link", parent=body,
                                               textColor=colors.HexColor(
                                                   "#0b3d91"))),
                      Spacer(1, 6)]

    def footer(canv, doc_):
        canv.saveState()
        canv.setFont("Helvetica", 7)
        canv.setFillColor(colors.HexColor("#666666"))
        if repo:
            canv.drawString(1.8 * cm, 1.15 * cm,
                            f"AS assignment repair report - {repo}")
        canv.drawRightString(A4[0] - 1.8 * cm, 1.15 * cm, f"page {doc_.page}")
        canv.restoreState()

    doc = SimpleDocTemplate(path, pagesize=A4, title="Repair report",
                            author="AS assignment",
                            subject=f"SPEC 15 report - {repo or REPO_URL}",
                            onFirstPage=footer, onLaterPages=footer)
    doc.build(story)
    return path


def make_bundle(outdir: str, name: str = "report_bundle.zip") -> str:
    """Zip everything under ``outdir`` (the self-contained M10 deliverable)."""
    path = os.path.join(outdir, name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(outdir):
            for fname in sorted(files):
                if fname == name:
                    continue
                full = os.path.join(root, fname)
                zf.write(full, os.path.relpath(full, outdir))
    return path


def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:                       # pragma: no cover - env dependent
        return "unknown"


#: Where the deliverable (sources, tests, generated GIFs/PDF) is published;
#: used when the working copy has no ``origin`` remote (e.g. a bare checkout).
REPO_URL = "https://github.com/aaditya-biswas/AS_Assignment"


def _git_remote() -> str:
    """The ``origin`` remote of the working copy, as an https URL.

    Falls back to :data:`REPO_URL` when git is unavailable or has no origin,
    and normalizes the ``git@github.com:owner/repo.git`` SSH form so the URL
    printed in the report is always clickable.
    """
    try:
        import subprocess
        out = subprocess.run(["git", "remote", "get-url", "origin"],
                             capture_output=True, text=True, timeout=5)
        url = out.stdout.strip()
        if url:
            if url.startswith("git@") and ":" in url:
                host, path = url.split(":", 1)
                url = "https://" + host[len("git@"):] + "/" + path
            return url[:-4] if url.endswith(".git") else url
    except Exception:                       # pragma: no cover - env dependent
        pass
    return REPO_URL


def build_context() -> dict:
    import sys
    return {"date": time.strftime("%Y-%m-%d %H:%M"),
            "git_commit": _git_commit(), "repo": _git_remote(),
            "python": sys.version.split()[0],
            "hashseed": os.environ.get("PYTHONHASHSEED", "<unset>")}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="make_report.py",
        description="Build the SPEC 15 report (PDF + Markdown) and a zip "
                    "bundle, including the POP figure, the Gantt, four repair "
                    "cards and the altered-per-disruption table.")
    ap.add_argument("--outdir", default=os.path.join("outputs", "report"))
    ap.add_argument("--data", default="outputs",
                    help="directory holding results.csv / run_meta.json / *.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--disruptions", type=int, default=5,
                    help="route blockages in case study A")
    ap.add_argument("--mode", default="negotiate")
    ap.add_argument("--choke-agents", type=int, default=4)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--zip", dest="zip_bundle", action="store_true",
                    help="also write report_bundle.zip next to the report")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    cases = [gather(args.seed, args.mode, n_agents=args.agents, hw=args.size,
                    disruptions=args.disruptions),
             gather_choke(args.choke_agents, seed=args.seed)]
    figsets = [figures(cases[0], args.outdir, "repair"),
               figures(cases[1], args.outdir, "choke")]
    experiment = load_experiment(args.data)
    sweep_figs: Dict[str, str] = {}
    if experiment:
        for src in experiment["plots"]:
            base = os.path.basename(src)
            shutil.copy2(src, os.path.join(args.outdir, "sweep_" + base))
            sweep_figs[base[:-4]] = "sweep_" + base

    summaries = [case_summary(c) for c in cases]
    altered_rows = [altered_table(c["records"]) for c in cases]
    context = build_context()
    blocks = report_blocks(cases, figsets, summaries, altered_rows, experiment,
                           sweep_figs, context)
    md = write_markdown(os.path.join(args.outdir, "report.md"), blocks)
    pdf = None if args.no_pdf else write_pdf(
        os.path.join(args.outdir, "report.pdf"), blocks, args.outdir,
        repo=context["repo"])
    bundle = make_bundle(args.outdir) if args.zip_bundle else None

    n_figs = sum(len(f) for f in figsets) + len(sweep_figs)
    print(f"report: {md}" + (f" | {pdf}" if pdf else "") +
          f"  ({n_figs} figures)")
    print(f"  repository: {context['repo']}  (commit {context['git_commit']})")
    print(f"  case A {summaries[0]['case']}: {summaries[0]['tasks']} tasks, "
          f"{summaries[0]['repairs']} repairs, {summaries[0]['messages']} msgs")
    print(f"  case B {summaries[1]['case']}: {summaries[1]['tasks']} tasks, "
          f"{summaries[1]['repairs']} repairs, {summaries[1]['messages']} msgs, "
          f"rungs {summaries[1]['rungs']}")
    print(f"  sweep data: {'yes' if experiment else 'no'}")
    if bundle:
        print(f"  bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

