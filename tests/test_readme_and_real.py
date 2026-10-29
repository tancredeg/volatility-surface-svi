"""Consistency of the published numbers.

* The README tables are exactly the generated ``results/real/tables.md`` blocks
  (no hand-edited number survives a rerun).
* When the raw SPY chain is available locally (it is not redistributed), a
  recalibration reproduces ``results/real/summary.json``.
"""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pytest

from volsurf.data import load_snapshot
from volsurf.pipeline import MODELS, calibrate_all, fit_table

from .conftest import ROOT

REAL_META = ROOT / "data" / "snapshots" / "spy_20260603.json"
REAL_RESULTS = ROOT / "results" / "real"
BLOCK = re.compile(r"<!-- BEGIN:([\w-]+) -->\n(.*?)\n<!-- END:\1 -->", re.S)


def test_readme_blocks_match_generated_tables() -> None:
    generated = dict(BLOCK.findall((REAL_RESULTS / "tables.md").read_text(encoding="utf-8")))
    readme = dict(BLOCK.findall((ROOT / "README.md").read_text(encoding="utf-8")))
    assert set(readme) == set(generated)
    for name, body in generated.items():
        assert readme[name] == body, f"README block {name!r} differs from results/real/tables.md"


def test_real_snapshot_metadata_is_complete() -> None:
    meta = json.loads(REAL_META.read_text(encoding="utf-8"))
    assert meta["synthetic"] is False and len(meta["chain_sha256"]) == 64
    assert meta["rate"]["rate_cc"] == pytest.approx(-np.log(1 - 0.0363 * 91 / 360) * 365 / 91, abs=1e-6)
    assert [d["ex_date"] for d in meta["dividends"]] == ["2026-06-18", "2026-09-18"]


def test_recalibration_reproduces_published_real_results() -> None:
    meta = json.loads(REAL_META.read_text(encoding="utf-8"))
    if not (ROOT / meta["chain_file"]).exists():
        pytest.skip("raw SPY chain not available (not redistributed)")
    warnings.filterwarnings("ignore", message="Values in x were outside bounds")
    snap = load_snapshot(REAL_META)
    exp = json.loads((REAL_RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert [len(s.k) for s in snap.slices] == exp["snapshot"]["quotes"]
    np.testing.assert_allclose([s.forward for s in snap.slices], exp["snapshot"]["forwards"], rtol=1e-12)
    cal = calibrate_all(snap)
    fit = fit_table(snap, cal)
    for m in MODELS:
        np.testing.assert_allclose(fit[f"rmse_vp[{m}]"], exp["rmse_vp"][m], rtol=1e-3, atol=1e-3)
    assert cal.ssvi.params.rho == pytest.approx(exp["ssvi"]["rho"], abs=1e-4)
    assert cal.ssvi.params.gamma == pytest.approx(exp["ssvi"]["gamma"], abs=1e-4)
    np.testing.assert_allclose(cal.ssvi.theta, exp["ssvi"]["theta"], rtol=1e-9)
