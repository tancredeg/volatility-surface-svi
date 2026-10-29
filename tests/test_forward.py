"""Discounting, parity forwards and the dividend-aware forward curve."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsurf.forward import CashDividend, ForwardCurve, implied_spot, parity_forward, rate_from_tbill_discount
from volsurf.iv import black76_price


def test_tbill_conversion_matches_definition() -> None:
    r = rate_from_tbill_discount(0.0363, 91)
    assert r == pytest.approx(0.036974, abs=1e-6)
    assert np.exp(-r * 91 / 365) == pytest.approx(1 - 0.0363 * 91 / 360, rel=1e-12)
    with pytest.raises(ValueError):
        rate_from_tbill_discount(4.0, 91)


def _chain(F: float, D: float, T: float, strikes: np.ndarray, spread: float = 0.02) -> pd.DataFrame:
    rows = []
    for is_call, name in ((True, "call"), (False, "put")):
        vol = 0.2 - 0.3 * np.log(strikes / F)
        mid = black76_price(F, strikes, T, D, vol, is_call)
        rows.append(
            pd.DataFrame({"option_type": name, "strike": strikes, "bid": mid - spread, "ask": mid + spread})
        )
    return pd.concat(rows, ignore_index=True)


def test_parity_forward_recovers_known_forward() -> None:
    F, D, T = 761.7558, 0.9911, 0.2445
    est = parity_forward(_chain(F, D, T, np.arange(700.0, 830.0, 1.0)), D)
    assert est.forward == pytest.approx(F, abs=1e-9)
    assert est.n_pairs >= 40 and est.dispersion < 1e-8


def test_parity_forward_needs_pairs() -> None:
    chain = _chain(100.0, 1.0, 0.5, np.array([90.0, 100.0]))
    with pytest.raises(ValueError):
        parity_forward(chain, 1.0)


def test_forward_curve_dividend_jump_and_spot_consistency() -> None:
    r = 0.037
    divs = (CashDividend(0.04, 1.9), CashDividend(0.29, 1.89))
    S0 = implied_spot(761.7558, 0.2445, r, divs)
    curve = ForwardCurve(S0, r, divs)
    assert float(curve(0.2445)) == pytest.approx(761.7558, rel=1e-12)
    jump = float(curve(0.04)) - float(curve.left(0.04))
    assert jump == pytest.approx(-1.9, rel=1e-12)
    assert float(curve(0.0)) == pytest.approx(S0)
    # between dividends the forward grows at the rate
    assert float(curve(0.2)) / float(curve(0.1)) == pytest.approx(np.exp(r * 0.1), rel=1e-12)
