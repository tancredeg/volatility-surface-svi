"""Barrier options: closed forms under flat Black-Scholes vol, Monte Carlo under local vol.

Products (continuous monitoring, no rebate, spot S above the barrier H):
    * down-and-out call (DOC): call that dies if S touches H;
    * down-and-in put (DIP): put that comes alive if S touches H. A reverse
      convertible is a bond minus a DIP, so the DIP price is what the issuer
      pays the investor for the barrier risk.

Closed forms (Reiner & Rubinstein 1991; Haug 2007, section 4.17.1) with carry
b = r - q, mu = (b - sigma^2/2)/sigma^2, s = sigma sqrt(T), and for phi = +1
(call) / -1 (put), eta = +1 (down barrier):

    A = phi S e^{-qT} N(phi x1) - phi K e^{-rT} N(phi (x1 - s)),       x1 = ln(S/K)/s + (1+mu) s
    B = phi S e^{-qT} N(phi x2) - phi K e^{-rT} N(phi (x2 - s)),       x2 = ln(S/H)/s + (1+mu) s
    C = phi S e^{-qT} (H/S)^{2(mu+1)} N(eta y1) - phi K e^{-rT} (H/S)^{2 mu} N(eta (y1 - s)),
                                                                      y1 = ln(H^2/(S K))/s + (1+mu) s
    D = same as C with y2 = ln(H/S)/s + (1+mu) s

    DOC:  K >= H: A - C          K < H: B - D
    DIP:  K >  H: B - C + D      K <= H: A

Monte Carlo: Y_t = ln(S_t / F(0, t)) follows dY = -v/2 dt + sqrt(v) dW with
v = sigma_loc^2(t, Y), simulated with a log-Euler step (S_t / F(0, t) is an exact
martingale step by step). Continuous monitoring uses the Brownian-bridge
crossing probability between grid dates,

    P(cross | Y_i, Y_{i+1}) = exp(-2 (Y_i - b_i)(Y_{i+1} - b_{i+1}) / (v dt)),  b = ln(H / F(0, t)),

as a conditional survival weight (no extra random numbers, lower variance).
Antithetic normals, fixed seed.

References:
    Reiner, E. & Rubinstein, M. (1991). Breaking down the barriers. Risk, 4(8), 28-35.
    Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas, 2nd ed., McGraw-Hill.
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering, Springer, sec. 6.4.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from .forward import ForwardCurve
from .iv import FloatArray


def _rr_terms(
    S: float, K: float, H: float, T: float, r: float, q: float, sigma: float, phi: float
) -> tuple[float, float, float, float]:
    s = sigma * np.sqrt(T)
    mu = (r - q - 0.5 * sigma**2) / sigma**2
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    x1 = np.log(S / K) / s + (1.0 + mu) * s
    x2 = np.log(S / H) / s + (1.0 + mu) * s
    y1 = np.log(H * H / (S * K)) / s + (1.0 + mu) * s
    y2 = np.log(H / S) / s + (1.0 + mu) * s
    h1, h2 = (H / S) ** (2.0 * (mu + 1.0)), (H / S) ** (2.0 * mu)
    A = phi * S * dq * ndtr(phi * x1) - phi * K * dr * ndtr(phi * (x1 - s))
    B = phi * S * dq * ndtr(phi * x2) - phi * K * dr * ndtr(phi * (x2 - s))
    C = phi * S * dq * h1 * ndtr(y1) - phi * K * dr * h2 * ndtr(y1 - s)
    D = phi * S * dq * h1 * ndtr(y2) - phi * K * dr * h2 * ndtr(y2 - s)
    return float(A), float(B), float(C), float(D)


def bs_down_and_out_call(
    spot: float, strike: float, barrier: float, maturity: float, rate: float, div_yield: float, sigma: float
) -> float:
    """Continuously monitored down-and-out call, flat vol, continuous dividend yield, no rebate."""
    if spot <= barrier:
        return 0.0
    A, B, C, D = _rr_terms(spot, strike, barrier, maturity, rate, div_yield, sigma, 1.0)
    return A - C if strike >= barrier else B - D


def bs_down_and_in_put(
    spot: float, strike: float, barrier: float, maturity: float, rate: float, div_yield: float, sigma: float
) -> float:
    """Continuously monitored down-and-in put, flat vol, continuous dividend yield, no rebate."""
    if spot <= barrier:
        return bs_put(spot, strike, maturity, rate, div_yield, sigma)
    A, B, C, D = _rr_terms(spot, strike, barrier, maturity, rate, div_yield, sigma, -1.0)
    return B - C + D if strike > barrier else A


def bs_call(
    spot: float, strike: float, maturity: float, rate: float, div_yield: float, sigma: float
) -> float:
    """Black-Scholes-Merton call with continuous dividend yield."""
    s = sigma * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - div_yield) * maturity) / s + 0.5 * s
    return float(
        spot * np.exp(-div_yield * maturity) * ndtr(d1) - strike * np.exp(-rate * maturity) * ndtr(d1 - s)
    )


def bs_put(spot: float, strike: float, maturity: float, rate: float, div_yield: float, sigma: float) -> float:
    """Black-Scholes-Merton put with continuous dividend yield."""
    s = sigma * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - div_yield) * maturity) / s + 0.5 * s
    return float(
        strike * np.exp(-rate * maturity) * ndtr(s - d1) - spot * np.exp(-div_yield * maturity) * ndtr(-d1)
    )


def simulation_times(maturity: float, steps_per_year: int, extra: ArrayLike = ()) -> FloatArray:
    """Uniform grid on [0, T] plus extra dates inside (0, T) (e.g. ex-dividend dates)."""
    n = max(int(np.ceil(maturity * steps_per_year)), 1)
    base = np.linspace(0.0, maturity, n + 1)
    ex = np.asarray(extra, dtype=np.float64)
    ex = ex[(ex > 0.0) & (ex < maturity)]
    return np.unique(np.concatenate([base, ex]))


VolFn = Callable[[int, FloatArray], FloatArray]


def flat_vol(sigma: float) -> VolFn:
    """Constant-volatility function with the MC engine's signature."""

    def f(step: int, y: FloatArray) -> FloatArray:
        return np.full_like(y, sigma)

    return f


@dataclass(frozen=True)
class Estimate:
    value: float
    stderr: float


def _pair_estimate(x: FloatArray, discount: float) -> Estimate:
    half = len(x) // 2
    pair = 0.5 * (x[:half] + x[half:])  # antithetic pair averages are i.i.d.
    return Estimate(float(discount * pair.mean()), float(discount * pair.std(ddof=1) / np.sqrt(half)))


@dataclass(frozen=True)
class BarrierMC:
    """Monte-Carlo output on common paths."""

    barriers: FloatArray
    doc_call: list[Estimate]
    dip_put: list[Estimate]
    knock_prob: FloatArray
    call: Estimate
    put: Estimate
    terminal_spot: FloatArray  # S_T on every path (antithetic halves concatenated)
    discount: float
    n_steps: int

    def vanilla(self, strike: float, is_call: bool) -> Estimate:
        """European price at any strike from the simulated terminal distribution."""
        s_T = self.terminal_spot
        pay = np.maximum(s_T - strike, 0.0) if is_call else np.maximum(strike - s_T, 0.0)
        return _pair_estimate(pay, self.discount)


def mc_barrier(
    *,
    strike: float,
    barriers: Sequence[float],
    maturity: float,
    discount: float,
    forward: ForwardCurve,
    vol: VolFn,
    times: ArrayLike,
    n_paths: int,
    seed: int = 12345,
    bridge: bool = True,
) -> BarrierMC:
    """Down-and-out calls and down-and-in puts for several barriers on common paths.

    Args:
        strike: K. barriers: H values (each below the spot). maturity: T (= times[-1]).
        discount: D(0, T).
        forward: Forward curve F(0, t), right-continuous at ex-dividend dates.
        vol: sigma(step, y) for the step [times[i], times[i+1]].
        times: Simulation dates, from 0 to T.
        n_paths: Number of paths (rounded up to an even number, antithetic pairs).
        seed: RNG seed. bridge: Apply the Brownian-bridge crossing correction.
    """
    tt = np.asarray(times, dtype=np.float64)
    if abs(tt[-1] - maturity) > 1e-12 or tt[0] != 0.0:
        raise ValueError("times must start at 0 and end at the maturity.")
    half = (n_paths + 1) // 2
    rng = np.random.default_rng(seed)
    y = np.zeros(2 * half)
    bars = np.asarray(barriers, dtype=np.float64)
    log_h = np.log(bars)[:, None]
    survive = np.ones((len(bars), 2 * half))
    for i in range(len(tt) - 1):
        dt = tt[i + 1] - tt[i]
        xi = rng.standard_normal(half)
        xi = np.concatenate([xi, -xi])
        sig = vol(i, y)
        var = sig * sig * dt
        y_new = y - 0.5 * var + np.sqrt(var) * xi
        b_start = log_h - np.log(float(forward(tt[i])))
        b_end_left = log_h - np.log(float(forward.left(tt[i + 1])))
        b_end = log_h - np.log(float(forward(tt[i + 1])))
        if bridge:
            gap0 = y - b_start
            gap1 = y_new - b_end_left
            both_above = (gap0 > 0) & (gap1 > 0)
            p_cross = np.where(both_above, np.exp(-2.0 * gap0 * gap1 / np.maximum(var, 1e-300)), 1.0)
            survive *= 1.0 - p_cross
        survive *= (y_new > b_end_left) & (y_new > b_end)
        y = y_new
    s_T = float(forward(maturity)) * np.exp(y)
    call_pay = np.maximum(s_T - strike, 0.0)
    put_pay = np.maximum(strike - s_T, 0.0)
    return BarrierMC(
        barriers=bars,
        doc_call=[_pair_estimate(call_pay * w, discount) for w in survive],
        dip_put=[_pair_estimate(put_pay * (1.0 - w), discount) for w in survive],
        knock_prob=np.asarray(1.0 - survive.mean(axis=1), dtype=np.float64),
        call=_pair_estimate(call_pay, discount),
        put=_pair_estimate(put_pay, discount),
        terminal_spot=s_T,
        discount=discount,
        n_steps=len(tt) - 1,
    )
