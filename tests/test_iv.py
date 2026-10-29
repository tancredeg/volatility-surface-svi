"""Black-76 identities and implied-vol inversion."""

from __future__ import annotations

import numpy as np
import pytest

from volsurf.iv import black76_price, black76_vega, forward_delta, implied_vol


def test_put_call_parity_holds_exactly() -> None:
    F, D, T = 761.75, 0.991, 0.244
    K = np.linspace(400.0, 1000.0, 61)
    sig = np.linspace(0.05, 0.9, 61)
    c = black76_price(F, K, T, D, sig, True)
    p = black76_price(F, K, T, D, sig, False)
    np.testing.assert_allclose(c - p, D * (F - K), atol=1e-9)


@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip(is_call: bool) -> None:
    rng = np.random.default_rng(0)
    n = 2000
    F, D = 100.0, 0.98
    T = rng.uniform(0.02, 2.0, n)
    sig = rng.uniform(0.05, 1.5, n)
    K = F * np.exp(rng.uniform(-0.8, 0.5, n) * sig * np.sqrt(T) * 2)
    price = black76_price(F, K, T, D, sig, is_call)
    iv = implied_vol(price, F, K, T, D, is_call)
    vega = black76_vega(F, K, T, D, sig)
    ok = vega > 1e-6
    assert ok.mean() > 0.95
    np.testing.assert_allclose(iv[ok], sig[ok], atol=1e-8)


def test_implied_vol_is_nan_outside_no_arbitrage_bounds() -> None:
    F, K, T, D = 100.0, 90.0, 0.5, 0.99
    intrinsic = D * (F - K)
    assert np.isnan(implied_vol(intrinsic - 0.01, F, K, T, D, True))
    assert np.isnan(implied_vol(D * F + 0.01, F, K, T, D, True))
    assert np.isnan(implied_vol(-1.0, F, K, T, D, False))


def test_scalar_input_returns_scalar_like() -> None:
    p = float(black76_price(100.0, 100.0, 1.0, 1.0, 0.2, True))
    assert float(implied_vol(p, 100.0, 100.0, 1.0, 1.0, True)) == pytest.approx(0.2, abs=1e-10)


def test_vega_matches_finite_difference() -> None:
    F, K, T, D, s, h = 100.0, 110.0, 0.7, 0.97, 0.25, 1e-6
    fd = (black76_price(F, K, T, D, s + h, True) - black76_price(F, K, T, D, s - h, True)) / (2 * h)
    assert float(black76_vega(F, K, T, D, s)) == pytest.approx(float(fd), rel=1e-6)


def test_forward_delta_ranges_and_parity() -> None:
    K = np.linspace(50.0, 200.0, 31)
    dc = forward_delta(100.0, K, 0.5, 0.3, True)
    dp = forward_delta(100.0, K, 0.5, 0.3, False)
    assert np.all((dc > 0) & (dc < 1)) and np.all((dp < 0) & (dp > -1))
    np.testing.assert_allclose(dc - dp, 1.0)
