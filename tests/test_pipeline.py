"""End-to-end smoke test (light settings) and regression against the committed sample results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from volsurf.data import PreparedSnapshot
from volsurf.pipeline import MODELS, Calibration, StudyConfig, fit_table, run_study
from volsurf.plots import plot_all
from volsurf.report import BLOCKS, inject, summary, write_outputs

from .conftest import ROOT

EXPECTED = ROOT / "results" / "sample" / "summary.json"


def test_regression_against_committed_sample_results(
    sample: PreparedSnapshot, sample_cal: Calibration
) -> None:
    exp = json.loads(EXPECTED.read_text(encoding="utf-8"))
    fit = fit_table(sample, sample_cal)
    for m in MODELS:
        np.testing.assert_allclose(fit[f"rmse_vp[{m}]"], exp["rmse_vp"][m], rtol=2e-3, atol=2e-3)
    ss = sample_cal.ssvi
    assert ss.params.rho == pytest.approx(exp["ssvi"]["rho"], abs=1e-3)
    assert ss.params.eta == pytest.approx(exp["ssvi"]["eta"], rel=1e-3)
    assert ss.params.gamma == pytest.approx(exp["ssvi"]["gamma"], abs=1e-3)
    np.testing.assert_allclose(ss.theta, exp["ssvi"]["theta"], rtol=1e-6)
    np.testing.assert_allclose([s.forward for s in sample.slices], exp["snapshot"]["forwards"], rtol=1e-10)
    assert [len(s.k) for s in sample.slices] == exp["snapshot"]["quotes"]
    for got, ref in zip(sample_cal.svi_con, exp["svi_constrained_params"], strict=True):
        np.testing.assert_allclose(
            got.params.w(np.linspace(-0.5, 0.2, 50)),
            _w(ref, np.linspace(-0.5, 0.2, 50)),
            rtol=2e-3,
            atol=2e-5,
        )


def _w(x: list[float], k: np.ndarray) -> np.ndarray:
    a, b, rho, m, s = x
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def test_full_study_smoke(sample: PreparedSnapshot, tmp_path: Path) -> None:
    cfg = StudyConfig(n_paths=20_000, steps_per_day=1, n_timing_runs=1, n_noise_runs=1)
    res = run_study(sample, cfg)
    write_outputs(res, tmp_path / "res")
    figs = plot_all(res, tmp_path / "fig")
    assert all(f.exists() and f.stat().st_size > 10_000 for f in figs)
    s = summary(res)
    assert s["snapshot"]["synthetic"] is True and s["ssvi"]["guaranteed_arbitrage_free"] is True
    t = res.barrier.table
    assert (t.dip_lv > 0).all() and (np.diff(t.dip_lv) > 0).all()  # higher barrier, dearer knock-in
    assert (t.knock_prob_lv > 0).all()
    readme = tmp_path / "README.md"
    readme.write_text(
        "".join(f"<!-- BEGIN:{k} -->\nold\n<!-- END:{k} -->\n" for k in BLOCKS), encoding="utf-8"
    )
    assert inject(readme, tmp_path / "res" / "tables.md") == len(BLOCKS)
    assert "old" not in readme.read_text(encoding="utf-8")
