"""Raw SVI: Durrleman g, known butterfly counterexample, parameter recovery, constraints."""

from __future__ import annotations

import numpy as np
import pytest

from volsurf.data import PreparedSnapshot
from volsurf.pipeline import Calibration
from volsurf.svi import K_CHECK, SVIParams, _quasi_explicit_inner, calibrate_svi, calibrate_svi_surface

TRUE = SVIParams(a=0.004, b=0.06, rho=-0.6, m=0.02, sigma=0.15)
T = 0.25
K = np.linspace(-0.5, 0.15, 120)


def test_derivatives_match_finite_differences() -> None:
    k, h = np.linspace(-1, 1, 41), 1e-5
    np.testing.assert_allclose(TRUE.dw(k), (TRUE.w(k + h) - TRUE.w(k - h)) / (2 * h), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(
        TRUE.d2w(k), (TRUE.w(k + h) - 2 * TRUE.w(k) + TRUE.w(k - h)) / h**2, rtol=1e-3, atol=1e-6
    )


def test_vogt_counterexample_has_butterfly_arbitrage() -> None:
    # Gatheral & Jacquier (2014), Example 3.1: (a, b, m, rho, sigma) = (-0.0410, 0.1331, 0.3586, 0.3060, 0.4153)
    vogt = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    g = vogt.g(K_CHECK)
    assert g.min() < -0.03
    assert 0.7 < K_CHECK[np.argmin(g)] < 1.1
    assert vogt.max_wing_slope < 2.0  # Lee's bound holds: it is necessary, not sufficient


def test_quasi_explicit_inner_is_exact_at_true_m_sigma() -> None:
    w = TRUE.w(K)
    _, x = _quasi_explicit_inner(TRUE.m, TRUE.sigma, K, w, np.ones_like(K))
    np.testing.assert_allclose(x, TRUE.as_array(), atol=1e-10)


@pytest.mark.parametrize("constrained", [False, True])
def test_recovery_without_noise(constrained: bool) -> None:
    iv = np.sqrt(TRUE.w(K) / T)
    fit = calibrate_svi(K, iv, T, constrained=constrained)
    assert fit.success
    assert fit.rmse < 1e-7
    np.testing.assert_allclose(fit.params.w(K), TRUE.w(K), atol=1e-8)


def test_recovery_with_noise_is_at_noise_level() -> None:
    rng = np.random.default_rng(1)
    noise = 0.002  # 0.2 vol point
    iv = np.sqrt(TRUE.w(K) / T) + rng.normal(0, noise, K.shape)
    fit = calibrate_svi(K, iv, T, constrained=True)
    assert fit.rmse <= 1.5 * noise
    true_iv = np.sqrt(TRUE.w(K) / T)
    assert np.sqrt(np.mean((fit.params.implied_vol(K, T) - true_iv) ** 2)) < 0.5 * noise


def test_constrained_surface_is_free_of_static_arbitrage(sample_cal: Calibration) -> None:
    prev = None
    for f in sample_cal.svi_con:
        assert f.success
        assert f.params.g(K_CHECK).min() >= -1e-7
        assert f.params.max_wing_slope <= 2.0
        assert f.params.min_total_variance >= 0.0
        if prev is not None:
            assert (f.params.w(K_CHECK) - prev.w(K_CHECK)).min() >= -1e-7
        prev = f.params


def test_constrained_costs_little_fit(sample_cal: Calibration) -> None:
    for raw, con in zip(sample_cal.svi_raw, sample_cal.svi_con, strict=True):
        assert con.rmse >= raw.rmse - 1e-6  # the constrained problem is a restriction
        assert con.rmse - raw.rmse < 0.0015  # costs less than 0.15 vol point


def test_noise_stability_of_constrained_fit() -> None:
    rng = np.random.default_rng(7)
    atm, wing = [], []
    for _ in range(8):
        iv = np.sqrt(TRUE.w(K) / T) + rng.normal(0, 0.001, K.shape)
        p = calibrate_svi(K, iv, T, constrained=True).params
        atm.append(float(np.sqrt(p.w(0.0) / T)))
        wing.append(float(np.sqrt(p.w(-0.5) / T)))
    assert np.std(atm) < 5e-4 and np.std(wing) < 2e-3


def test_constrained_fit_is_robust_to_quote_noise(sample: PreparedSnapshot) -> None:
    """Regression test for a bug fixed in v1.0: SLSQP alone sometimes ended in a feasible
    point ~3 vol points worse than the optimum. With the penalty start, imposing no-arbitrage
    must cost little on every noisy recalibration, and the ATM level must stay put."""
    rng = np.random.default_rng(11)
    atm: list[list[float]] = []
    for _ in range(6):
        data = [(s.maturity, s.k, s.iv_mid + rng.normal(0.0, 0.001, s.k.shape)) for s in sample.slices]
        raw = calibrate_svi_surface(data, constrained=False)
        con = calibrate_svi_surface(data, constrained=True)
        for r, c in zip(raw, con, strict=True):
            assert c.success
            assert c.rmse - r.rmse < 0.0015  # < 0.15 vol point
        atm.append([float(np.sqrt(c.params.w(0.0) / c.maturity)) for c in con])
    assert np.std(np.array(atm), axis=0).max() < 0.001  # < 0.1 vol point


def test_surface_calibration_orders_slices() -> None:
    slices = [
        (t, K, np.sqrt(SVIParams(TRUE.a * t / T, TRUE.b * t / T, TRUE.rho, TRUE.m, TRUE.sigma).w(K) / t))
        for t in (0.1, 0.25, 0.5)
    ]
    fits = calibrate_svi_surface(slices, constrained=True)
    assert [f.maturity for f in fits] == [0.1, 0.25, 0.5]
    assert all(f.rmse < 1e-4 for f in fits)
