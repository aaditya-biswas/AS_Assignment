"""Experiment sweeps (``SPEC.md`` section 12).

The tests keep the sweep *tiny* (12x12 grids, one task per agent) but check the
things a full run relies on: the SPEC 12 column schema, that the same seed
gives byte-identical rows, that ``--jobs > 1`` does not change the result, and
that the CSVs/meta/plots are written where the report expects them.
"""
from __future__ import annotations

import csv
import json
import os

import experiments

FAMILY = {"N": 4, "density": 0.0, "seed": 0, "tasks_per_agent": 1,
          "H": 12, "W": 12, "sweep": "test"}


def _rows(family=None, modes=("negotiate", "self_only")):
    fam = dict(FAMILY if family is None else family)
    return experiments.run_family(fam, list(modes))


def test_sweep_jobs_shapes():
    assert len(experiments.sweep_jobs("A", 2)) == 5 * 2
    assert len(experiments.sweep_jobs("B", 2)) == 5 * 2           # one N, 5 d
    assert len(experiments.sweep_jobs("grid", 1)) == 5 * 5
    quick = experiments.sweep_jobs("grid", 3, quick=True)
    assert len(quick) == 2 * 2 * 3
    assert {j["density"] for j in experiments.sweep_jobs("B", 1)} == \
        {0.0, 0.02, 0.05, 0.10, 0.15}


def _strip_timing(rows, drows):
    """Drop the wall-clock columns (they are not reproducible)."""
    keep = [{k: v for k, v in r.items() if k != "mean_cpu_ms"}
            for r in rows]
    dkeep = [{k: v for k, v in r.items() if k != "cpu_ms"} for r in drows]
    return keep, dkeep


def test_result_rows_carry_every_spec_12_column():
    rows, drows = _rows()
    assert len(rows) == 2, "one row per mode"
    assert {r["mode"] for r in rows} == {"negotiate", "self_only"}
    for row in rows:
        assert set(experiments.RESULT_COLUMNS) <= set(row), "column schema"
        assert row["N"] == 4 and row["seed"] == 0
        assert row["n_disruptions"] == sum(1 for d in drows
                                           if d["mode"] == row["mode"])
        hit = {n: row[f"mean_{n}"] for n in
               ("altered_plan", "altered_path", "delayed_only",
                "altered_naive")}
        assert all(isinstance(v, float) for v in hit.values())
        assert json.loads(row["rung_histogram"])           # a real histogram
        assert json.loads(row["messages_by_type"])
    for drow in drows:
        assert set(experiments.DISRUPTION_COLUMNS) <= set(drow)
        assert drow["event_type"] in ("accident", "emergency", "blockage",
                                      "breakdown", "conflict")
        assert drow["mode"] in ("negotiate", "self_only")


def test_the_same_seed_gives_identical_rows():
    rows_a, drows_a = _strip_timing(*_rows())
    rows_b, drows_b = _strip_timing(*_rows())
    assert rows_a == rows_b, "the sweep must be reproducible"
    assert drows_a == drows_b


def test_parallel_sweep_matches_the_serial_one():
    families = [dict(FAMILY), dict(FAMILY, seed=1)]
    serial, dserial, _ = experiments.run_sweep(families, ["negotiate"],
                                               jobs=1, verbose=False)
    par, dpar, _ = experiments.run_sweep(families, ["negotiate"], jobs=2,
                                         verbose=False)
    serial, dserial = _strip_timing(serial, dserial)
    par, dpar = _strip_timing(par, dpar)
    assert serial == par and dserial == dpar, "pool order must not matter"


def test_write_outputs_writes_the_spec_12_artefacts(tmp_path):
    rows, drows = _rows()
    meta = experiments.run_meta(experiments.build_parser().parse_args(
        ["--quick", "--outdir", str(tmp_path)]), [FAMILY], ["negotiate"],
        1.25, len(rows))
    paths = experiments.write_outputs(str(tmp_path), rows, drows, meta)
    for path in paths.values():
        assert os.path.getsize(path) > 0
    with open(paths["results"]) as fh:
        header = next(csv.reader(fh))
    assert header == list(experiments.RESULT_COLUMNS), "fixed column order"
    written = json.load(open(paths["meta"]))
    assert written["n_rows"] == len(rows)
    assert written["PYTHONHASHSEED"] == os.environ.get("PYTHONHASHSEED",
                                                       "<unset>")
    assert "packages" in written and "git_commit" in written


def test_make_plots_writes_figures(tmp_path):
    rows, drows = _rows()
    written = experiments.make_plots(rows, drows, str(tmp_path))
    assert written, "at least the trend/heatmap figures"
    for path in written:
        assert os.path.getsize(path) > 0
    assert os.path.exists(os.path.join(str(tmp_path), "mode_bars.png"))


def test_empty_rows_write_a_valid_csv(tmp_path):
    path = os.path.join(str(tmp_path), "empty.csv")
    experiments._write_csv([], experiments.RESULT_COLUMNS, path)
    with open(path) as fh:
        header = next(csv.reader(fh))
    assert header == list(experiments.RESULT_COLUMNS)
