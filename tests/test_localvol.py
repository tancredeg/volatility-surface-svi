"""Dupire local volatility: flat-surface identity and an independent price-space check."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from volsurf.localvol import build_ssvi_local_vol_grid, local_variance, ssvi_local_variance
from volsurf.pipeline import Calibration
from volsurf.ssvi import SSVIParams, ThetaCurve


def test_flat_surface_gives_the_flat_vol() -> None:
    y = np.linspace(-1, 1, 21)
    sigma, t = 0.23, 0.7
    v = local_variance(y, np.full_like(y, sigma**2 * t), 0.0, 0.0, np.full_like(y, sigma**2))
    np.testing.assert_allclose(v, sigma**2)


def _normalised_call(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Undiscounted call price divided by the forward, as a function of x = K/F (Black with total variance w)."""
    sw = np.sqrt(w)
    d1 = -np.log(x) / sw + 0.5 * sw
    return ndtr(d1) - x * ndtr(d1 - sw)


def test_matches_dupire_formula_in_price_space() -> None:
    # Independent check: sigma_loc^2 = 2 dc/dT / (x^2 d2c/dx2) on forward-normalised call prices.
    p = SSVIParams(rho=-0.55, eta=1.4, gamma=0.45)
    theta = ThetaCurve([0.08, 0.16, 0.25, 0.5], [0.0015, 0.0033, 0.0056, 0.0135])
    t, ht, hx = 0.2, 1e-5, 1e-3
    for y in (-0.3, -0.1, 0.0, 0.08):
        x = np.exp(y)

        def c(tt: float, xx: np.ndarray) -> np.ndarray:
            return _normalised_call(p.w(np.log(xx), float(theta(tt))), xx)

        xs = np.array([x - hx, x, x + hx])
        dc_dt = (c(t + ht, xs[1:2]) - c(t - ht, xs[1:2]))[0] / (2 * ht)
        d2c = (c(t, xs[2:]) - 2 * c(t, xs[1:2]) + c(t, xs[:1]))[0] / hx**2
        v_price = 2 * dc_dt / (x**2 * d2c)
        v_analytic = float(ssvi_local_variance(p, theta, t, np.array([y]))[0])
        assert v_analytic == pytest.approx(v_price, rel=2e-3)


def test_local_variance_is_positive_on_calibrated_surface(sample_cal: Calibration) -> None:
    fit = sample_cal.ssvi
    y = np.linspace(-1.5, 1.0, 251)
    for t in np.geomspace(1e-3, 0.5, 40):
        assert ssvi_local_variance(fit.params, fit.theta_curve(), float(t), y).min() > 0
    with pytest.raises(ValueError):
        ssvi_local_variance(fit.params, fit.theta_curve(), 0.0, y)


def test_grid_interpolation_and_clipping(sample_cal: Calibration) -> None:
    fit = sample_cal.ssvi
    times = np.linspace(0, 0.25, 11)
    grid = build_ssvi_local_vol_grid(
        fit.params, fit.theta_curve(), times, np.linspace(-1, 1, 201), vol_cap=0.5
    )
    assert grid.vol.shape == (10, 201) and grid.n_clipped > 0 and grid.vol.max() <= 0.5
    mid = 0.5 * (times[3] + times[4])
    exact = np.sqrt(ssvi_local_variance(fit.params, fit.theta_curve(), mid, np.array([0.0])))
    assert grid(3, np.array([0.0]))[0] == pytest.approx(float(exact[0]), rel=1e-12)
