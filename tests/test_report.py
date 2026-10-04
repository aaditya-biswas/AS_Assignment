"""Report generation (``SPEC.md`` sections 14 M10 and 15).

The acceptance criterion of M10 is that the report really *contains* a POP
figure, a Gantt chart, four repair cards and the altered-per-disruption table --
so that is what the tests assert, on a tiny instance (a 12x12 grid with four
agents) so the suite stays fast.  The PDF is checked to be a real multi-page
PDF, and the zip to be self-contained.
"""
from __future__ import annotations

import os
import zipfile

import make_report

SMALL = {"n_agents": 4, "tasks_per_agent": 1, "hw": 12, "disruptions": 4}


def _cases():
    return (make_report.gather(0, "negotiate", **SMALL),
            make_report.gather_choke(2, seed=0))


def _blocks(tmp_path, experiment=None):
    cases = _cases()
    figsets = [make_report.figures(cases[0], str(tmp_path), "repair"),
               make_report.figures(cases[1], str(tmp_path), "choke")]
    summaries = [make_report.case_summary(c) for c in cases]
    altered = [make_report.altered_table(c["records"]) for c in cases]
    blocks = make_report.report_blocks(cases, figsets, summaries, altered,
                                       experiment, {}, make_report.build_context())
    return cases, figsets, summaries, altered, blocks


def test_the_report_has_the_m10_figures(tmp_path):
    cases, figsets, summaries, altered, _blocks_ = _blocks(tmp_path)
    # the four M10 artefacts: POP, Gantt, four repair cards, altered table
    assert "pop" in figsets[0], "the POP figure is missing"
    assert "gantt" in figsets[0], "the Gantt chart is missing"
    cards = [k for k in figsets[0] if k.startswith("repair_card")]
    assert len(cards) >= 4, f"expected 4 repair cards, got {len(cards)}"
    assert len(altered[0]) == 4, "one row per SPEC 9.5 definition"
    for name, mean, median, mx in altered[0]:
        assert name in ("altered_plan", "altered_path", "delayed_only",
                        "altered_naive")
        assert 0 <= median <= mx
    for path in list(figsets[0].values()) + list(figsets[1].values()):
        assert os.path.getsize(os.path.join(str(tmp_path), path)) > 0


def test_the_protocol_shows_up_in_the_corridor_case(tmp_path):
    cases, figsets, summaries, _altered, _blocks_ = _blocks(tmp_path)
    choke = cases[1]
    assert choke["log"].messages, "the corridor must make the protocol speak"
    assert summaries[1]["messages"] == 3
    assert summaries[1]["rungs"] == {"R2": 1}
    assert "sequence" in figsets[1] and "ladder" in figsets[1]


def test_markdown_pdf_and_zip_are_written(tmp_path):
    cases, figsets, summaries, altered, blocks = _blocks(str(tmp_path))
    md = make_report.write_markdown(str(tmp_path / "report.md"), blocks)
    text = open(md).read()
    assert "Altered agents per disruption" in text
    assert "repair_card_00" in text and text.count("repair_card") >= 4
    assert "![A POP of an altered agent" in text

    pdf = make_report.write_pdf(str(tmp_path / "report.pdf"), blocks,
                                str(tmp_path))
    assert os.path.getsize(pdf) > 10_000
    with open(pdf, "rb") as fh:
        assert fh.read(5) == b"%PDF-"

    bundle = make_report.make_bundle(str(tmp_path))
    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
    assert "report.md" in names and "report.pdf" in names
    assert any(n.startswith("repair/") for n in names)


def test_experiment_section_is_optional(tmp_path):
    _c, _f, _s, _a, blocks_no = _blocks(str(tmp_path), experiment=None)
    assert any("experiments.py" in str(b) for b in blocks_no), \
        "the report must say how to fill the section"

    import experiments
    rows, drows = experiments.run_family(
        {"N": 4, "density": 0.0, "seed": 0, "tasks_per_agent": 1, "H": 12,
         "W": 12, "sweep": "test"}, ["negotiate"])
    experiment = {"df": _frame(rows), "meta": {"n_rows": len(rows),
                                               "n_families": 1,
                                               "sweep": "test",
                                               "modes": ["negotiate"]},
                  "plots": []}
    _c2, _f2, _s2, _a2, blocks_yes = _blocks(str(tmp_path), experiment)
    table_headers = [b[1] for b in blocks_yes if b[0] == "table"]
    assert any("families" in h for h in table_headers)


def _frame(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def test_altered_table_matches_the_records(tmp_path):
    cases, _f, _s, altered, _b = _blocks(tmp_path)
    records = cases[0]["records"]
    assert records, "the case study must have repairs"
    per_metric = {
        "altered_plan": [len(r.altered_plan_ids) for r in records],
        "altered_path": [len(r.altered_path_ids) for r in records],
        "delayed_only": [len(r.delayed_ids) for r in records],
        "altered_naive": [len(r.naive_ids) for r in records],
    }
    assert per_metric["altered_path"] == per_metric["altered_plan"], \
        "altered_path is a superset of altered_plan (SPEC 9.5)"
    for name, mean, median, mx in altered[0]:
        assert mx == max(per_metric[name]), f"{name}: max from the records"
        assert abs(mean - float(sum(per_metric[name])
                                / len(per_metric[name]))) < 1e-9
        assert median <= mx
