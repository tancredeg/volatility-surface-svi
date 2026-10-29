"""Arbitrage audit: Durrleman g agrees with an independent convexity check on call prices."""

from __future__ import annotations

import numpy as np

from volsurf.arbitrage import butterfly_report, calendar_report, density_negative_points
from volsurf.data import PreparedSnapshot
from volsurf.pipeline import Calibration, arbitrage_table, ssvi_surface_check
from volsurf.svi import K_CHECK, SVIParams

VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
CLEAN = SVIParams(a=0.004, b=0.06, rho=-0.6, m=0.02, sigma=0.15)


def test_price_convexity_check_detects_vogt() -> None:
    assert density_negative_points(VOGT, 100.0, 1.0, 1.0, K_CHECK) > 0
    assert density_negative_points(CLEAN, 100.0, 0.25, 0.99, K_CHECK) == 0


def test_butterfly_report_locates_vogt_violation_outside_quotes() -> None:
    rep = butterfly_report(VOGT, K_CHECK, (-0.5, 0.5))
    assert rep["g_min"] < 0 and 0.7 < rep["k_at_g_min"] < 1.1
    assert rep["neg_points_supported"] == 0 and rep["neg_points_extrapolated"] > 0
    rep_in = butterfly_report(VOGT, K_CHECK, (-1.5, 1.5))
    assert rep_in["neg_points_supported"] > 0 and rep_in["neg_points_extrapolated"] == 0


def test_calendar_report_detects_and_locates_crossing() -> None:
    # Longer slice with a flatter right wing: it crosses below the shorter one for large k only.
    short = SVIParams(a=0.002, b=0.05, rho=-0.3, m=0.0, sigma=0.1)
    long = SVIParams(a=0.006, b=0.04, rho=-0.95, m=0.0, sigma=0.1)
    k = np.linspace(-1.0, 1.0, 2001)
    crossing = k[long.w(k) < short.w(k)]
    assert crossing.size > 0 and crossing.min() > 0.0  # sanity of the constructed example
    rep = calendar_report(short, long, k, (-0.4, 0.1), (-0.4, 0.1))
    assert rep["viol_points_supported"] == 0
    assert rep["viol_points_extrapolated"] == crossing.size
    assert rep["k_viol_min"] == crossing.min() and rep["max_crossing_w"] > 0
    wide = calendar_report(short, long, k, (-1.0, 1.0), (-1.0, 1.0))
    assert wide["viol_points_supported"] == crossing.size
    none = calendar_report(short, short, k, (-1.0, 1.0), (-1.0, 1.0))
    assert none["viol_points_supported"] + none["viol_points_extrapolated"] == 0


def test_audit_on_sample(sample: PreparedSnapshot, sample_cal: Calibration) -> None:
    bfly, cal = arbitrage_table(sample, sample_cal)
    for model in ("SVI constrained", "SSVI"):
        b = bfly[bfly.model == model]
        assert (b.g_min >= -1e-7).all() and (b.density_neg_points == 0).all()
        c = cal[cal.model == model]
        assert (c.viol_points_supported + c.viol_points_extrapolated == 0).all()
    # the two butterfly checks agree slice by slice
    for _, row in bfly.iterrows():
        g_bad = row.neg_points_supported + row.neg_points_extrapolated > 0
        assert g_bad == (row.density_neg_points > 0)


def test_ssvi_surface_is_arbitrage_free_in_continuous_time(
    sample: PreparedSnapshot, sample_cal: Calibration
) -> None:
    chk = ssvi_surface_check(sample_cal.ssvi, float(sample.maturities.max()), n_t=60)
    assert chk["guaranteed"] == 1.0
    assert chk["g_min"] > 0 and chk["dw_dt_min"] >= 0 and chk["local_var_min"] > 0
    assert np.isfinite(list(chk.values())).all()
