"""SSVI: SVI equivalence, Theorem 4.1/4.2 guarantees, theta curve, parameter recovery."""

from __future__ import annotations

import numpy as np
import pytest

from volsurf.ssvi import SSVIParams, ThetaCurve, calibrate_ssvi, eta_max, theorem42_fstar

K = np.linspace(-1.5, 1.5, 601)


def test_slice_equals_raw_svi_slice() -> None:
    p = SSVIParams(rho=-0.55, eta=1.4, gamma=0.45)
    for th in (1e-4, 0.003, 0.05, 1.0):
        np.testing.assert_allclose(p.to_svi(th).w(K), p.w(K, th), rtol=1e-12, atol=1e-15)


def test_dw_dtheta_matches_finite_difference() -> None:
    p = SSVIParams(rho=-0.5, eta=1.2, gamma=0.4)
    th, h = 0.01, 1e-7
    fd = (p.w(K, th + h) - p.w(K, th - h)) / (2 * h)
    np.testing.assert_allclose(p.dw_dtheta(K, th), fd, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("gamma", [0.05, 0.25, 0.45, 0.5])
def test_fstar_is_the_maximum(gamma: float) -> None:
    th = np.logspace(-9, 5, 400_001)
    f = th ** (1 - 2 * gamma) * (1 + th) ** (2 * gamma - 2)
    assert theorem42_fstar(gamma) == pytest.approx(f.max(), rel=1e-6)


@pytest.mark.parametrize("rho", [-0.9, -0.5, 0.0, 0.6])
@pytest.mark.parametrize("gamma", [0.1, 0.3, 0.5])
def test_boundary_parameters_are_butterfly_free_at_every_theta(rho: float, gamma: float) -> None:
    p = SSVIParams(rho=rho, eta=eta_max(rho, gamma), gamma=gamma)
    assert p.guaranteed_arbitrage_free
    thetas = np.logspace(-8, 1, 60)
    m1, m2 = p.butterfly_margins(thetas)
    assert m1 > 0 and m2 >= -1e-9
    assert min(float(p.to_svi(float(t)).g(K).min()) for t in thetas) >= -1e-9
    assert p.calendar_margin(thetas) >= 0


def test_gamma_above_half_has_no_small_theta_guarantee() -> None:
    p = SSVIParams(rho=-0.5, eta=1.3, gamma=0.6)
    assert not p.guaranteed_arbitrage_free
    _, m2 = p.butterfly_margins(np.logspace(-10, -6, 50))
    assert m2 < 0
    with pytest.raises(ValueError):
        calibrate_ssvi([(0.1, K[:5], np.full(5, 0.2))], gamma_max=0.6)


def test_total_variance_increases_with_theta() -> None:
    p = SSVIParams(rho=-0.7, eta=1.0, gamma=0.5)
    th = np.linspace(1e-4, 0.5, 200)
    w = np.array([p.w(K, t) for t in th])
    assert np.all(np.diff(w, axis=0) >= -1e-15)


def test_theta_curve_is_monotone_and_starts_at_zero() -> None:
    c = ThetaCurve([0.08, 0.16, 0.24, 0.49], [0.0014, 0.0033, 0.0055, 0.0135])
    t = np.linspace(0, 0.8, 801)
    assert float(c(0.0)) == 0.0
    assert np.all(np.diff(c(t)) >= 0) and np.all(c.derivative(t) >= 0)
    assert float(c(0.24)) == pytest.approx(0.0055)
    with pytest.raises(ValueError):
        ThetaCurve([0.1, 0.2], [0.003, 0.002])


def test_recovers_known_surface() -> None:
    true = SSVIParams(rho=-0.55, eta=1.3, gamma=0.42)
    mats = [0.08, 0.16, 0.25, 0.5]
    thetas = [0.0015, 0.0033, 0.0056, 0.0135]
    k = np.linspace(-0.4, 0.15, 80)
    slices = [(t, k, np.sqrt(true.w(k, th) / t)) for t, th in zip(mats, thetas, strict=True)]
    fit = calibrate_ssvi(slices, mode="anchored")
    assert fit.params.rho == pytest.approx(true.rho, abs=1e-4)
    assert fit.params.eta == pytest.approx(true.eta, rel=1e-3)
    assert fit.params.gamma == pytest.approx(true.gamma, abs=1e-3)
    assert fit.rmse_total < 1e-5
    joint = calibrate_ssvi(slices, mode="joint")
    np.testing.assert_allclose(joint.theta, thetas, rtol=1e-3)


def test_non_monotone_atm_variance_is_flagged() -> None:
    k = np.linspace(-0.2, 0.1, 40)
    p = SSVIParams(rho=-0.5, eta=1.0, gamma=0.4)
    slices = [(0.1, k, np.sqrt(p.w(k, 0.004) / 0.1)), (0.2, k, np.sqrt(p.w(k, 0.0035) / 0.2))]
    fit = calibrate_ssvi(slices)
    assert fit.theta_adjusted and np.all(np.diff(fit.theta) >= 0)
    with pytest.raises(ValueError):
        calibrate_ssvi(slices, mode="other")
