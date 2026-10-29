"""Barrier options: closed forms (cross-checked with QuantLib when installed) and Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from volsurf.barrier import (
    bs_call,
    bs_down_and_in_put,
    bs_down_and_out_call,
    bs_put,
    flat_vol,
    mc_barrier,
    simulation_times,
)
from volsurf.data import PreparedSnapshot
from volsurf.forward import ForwardCurve
from volsurf.iv import black76_price
from volsurf.localvol import build_ssvi_local_vol_grid
from volsurf.pipeline import Calibration

S, r, q, sig, T = 100.0, 0.04, 0.015, 0.22, 0.5


def test_closed_form_limits() -> None:
    assert bs_down_and_out_call(S, 100, 1e-8, T, r, q, sig) == pytest.approx(bs_call(S, 100, T, r, q, sig))
    assert bs_down_and_in_put(S, 100, 1e-8, T, r, q, sig) == pytest.approx(0.0, abs=1e-12)
    assert bs_down_and_in_put(S, 100, S, T, r, q, sig) == pytest.approx(bs_put(S, 100, T, r, q, sig))
    assert bs_down_and_out_call(S, 100, S, T, r, q, sig) == 0.0
    # continuity of the two branches at K = H
    lo = bs_down_and_out_call(S, 95 - 1e-9, 95, T, r, q, sig)
    hi = bs_down_and_out_call(S, 95 + 1e-9, 95, T, r, q, sig)
    assert lo == pytest.approx(hi, abs=1e-7)


def test_closed_forms_match_quantlib() -> None:
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(3, 6, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    mat = today + 182
    t = dc.yearFraction(today, mat)

    def curve(rate: float) -> object:
        return ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))

    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        curve(q),
        curve(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sig, dc)),
    )
    for K in (80.0, 95.0, 100.0, 110.0):
        for H in (70.0, 90.0, 97.0, 99.5):
            for btype, otype, mine in (
                (ql.Barrier.DownOut, ql.Option.Call, bs_down_and_out_call),
                (ql.Barrier.DownIn, ql.Option.Put, bs_down_and_in_put),
            ):
                opt = ql.BarrierOption(
                    btype, H, 0.0, ql.PlainVanillaPayoff(otype, K), ql.EuropeanExercise(mat)
                )
                opt.setPricingEngine(ql.AnalyticBarrierEngine(proc))
                assert mine(S, K, H, t, r, q, sig) == pytest.approx(opt.NPV(), abs=1e-10)


def test_monte_carlo_matches_closed_form_under_flat_vol() -> None:
    curve = ForwardCurve(S, r - q)  # continuous carry r - q, no cash dividends
    res = mc_barrier(
        strike=100.0,
        barriers=[85.0, 95.0],
        maturity=T,
        discount=np.exp(-r * T),
        forward=curve,
        vol=flat_vol(sig),
        times=simulation_times(T, 365 * 2),
        n_paths=120_000,
        seed=11,
    )
    for j, H in enumerate((85.0, 95.0)):
        doc, dip = res.doc_call[j], res.dip_put[j]
        assert abs(doc.value - bs_down_and_out_call(S, 100, H, T, r, q, sig)) < 4 * doc.stderr
        assert abs(dip.value - bs_down_and_in_put(S, 100, H, T, r, q, sig)) < 4 * dip.stderr
    assert abs(res.call.value - bs_call(S, 100, T, r, q, sig)) < 4 * res.call.stderr
    assert abs(res.put.value - bs_put(S, 100, T, r, q, sig)) < 4 * res.put.stderr


def test_bridge_correction_removes_discrete_monitoring_bias() -> None:
    curve = ForwardCurve(S, r - q)
    common = {
        "strike": 100.0,
        "barriers": [92.0],
        "maturity": T,
        "discount": np.exp(-r * T),
        "forward": curve,
        "vol": flat_vol(sig),
        "times": simulation_times(T, 52),
        "n_paths": 60_000,
        "seed": 5,
    }
    with_bridge = mc_barrier(**common).doc_call[0]  # type: ignore[arg-type]
    no_bridge = mc_barrier(bridge=False, **common).doc_call[0]  # type: ignore[arg-type]
    exact = bs_down_and_out_call(S, 100, 92.0, T, r, q, sig)
    assert abs(with_bridge.value - exact) < 4 * with_bridge.stderr
    assert no_bridge.value - exact > 5 * no_bridge.stderr  # weekly monitoring misses crossings


def test_martingale_with_cash_dividend() -> None:
    curve = ForwardCurve(100.0, 0.03, ())
    from volsurf.forward import CashDividend

    curve = ForwardCurve(100.0, 0.03, (CashDividend(0.1, 1.5),))
    times = simulation_times(0.25, 365, [0.1])
    assert np.any(np.isclose(times, 0.1))
    res = mc_barrier(
        strike=1e-6,
        barriers=[50.0],
        maturity=0.25,
        discount=1.0,
        forward=curve,
        vol=flat_vol(0.3),
        times=times,
        n_paths=100_000,
        seed=2,
    )
    assert abs(res.call.value - float(curve(0.25))) < 4 * res.call.stderr
    with pytest.raises(ValueError):
        mc_barrier(
            strike=1.0,
            barriers=[50.0],
            maturity=0.3,
            discount=1.0,
            forward=curve,
            vol=flat_vol(0.3),
            times=times,
            n_paths=10,
        )


def test_local_vol_monte_carlo_reprices_the_surface(
    sample: PreparedSnapshot, sample_cal: Calibration
) -> None:
    fit = sample_cal.ssvi
    sl = sample.slices[2]
    T_ = sl.maturity
    curve = ForwardCurve(sl.forward * sl.discount, sample.meta.rate)  # forward-consistent, no dividends
    assert float(curve(T_)) == pytest.approx(sl.forward, rel=1e-12)
    times = simulation_times(T_, 365 * 2)
    grid = build_ssvi_local_vol_grid(fit.params, fit.theta_curve(), times, np.linspace(-1.5, 1.0, 501))
    res = mc_barrier(
        strike=sl.forward,
        barriers=[0.5 * sl.forward],
        maturity=T_,
        discount=sl.discount,
        forward=curve,
        vol=grid,
        times=times,
        n_paths=100_000,
        seed=3,
    )
    for x in (0.85, 0.95, 1.0, 1.05):
        K = x * sl.forward
        vol = float(fit.implied_vol(np.array([np.log(x)]), T_)[0])
        target = float(black76_price(sl.forward, K, T_, sl.discount, vol, x >= 1.0))
        est = res.vanilla(K, x >= 1.0)
        assert abs(est.value - target) < 4 * est.stderr + 0.002 * target
