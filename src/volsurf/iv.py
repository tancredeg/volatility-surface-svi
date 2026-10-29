"""Black-76 pricing on the forward and vectorised implied-volatility inversion.

Working on the forward removes the spot, the rate and the dividend yield from
the implied-volatility problem: only the forward ``F``, the discount factor
``D`` and the time to expiry ``T`` are needed.

Reference: Black, F. (1976). The pricing of commodity contracts.
Journal of Financial Economics, 3(1-2), 167-179.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]

SIGMA_MIN = 1e-4
SIGMA_MAX = 5.0


def _as_float(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def black76_price(
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
) -> FloatArray:
    """Discounted Black-76 price of a European call or put.

    C = D [F N(d1) - K N(d2)],   P = D [K N(-d2) - F N(-d1)],
    d1 = (ln(F/K) + s^2/2) / s,  d2 = d1 - s,  s = sigma sqrt(T).

    Args:
        forward: Forward price F for the option expiry.
        strike: Strike K.
        maturity: Time to expiry T in years.
        discount: Discount factor D = P(0, T).
        sigma: Black volatility (annualised, decimal). Must be > 0.
        is_call: True for calls, False for puts.

    Returns:
        Array of prices, broadcast over the inputs.
    """
    f, k, t, d, sig = (_as_float(x) for x in (forward, strike, maturity, discount, sigma))
    call = np.asarray(is_call, dtype=bool)
    s = sig * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * s * s) / s
    d2 = d1 - s
    call_px = d * (f * ndtr(d1) - k * ndtr(d2))
    put_px = d * (k * ndtr(-d2) - f * ndtr(-d1))
    return np.asarray(np.where(call, call_px, put_px), dtype=np.float64)


def black76_vega(
    forward: ArrayLike, strike: ArrayLike, maturity: ArrayLike, discount: ArrayLike, sigma: ArrayLike
) -> FloatArray:
    """Black-76 vega dPrice/dsigma = D F phi(d1) sqrt(T), identical for calls and puts."""
    f, k, t, d, sig = (_as_float(x) for x in (forward, strike, maturity, discount, sigma))
    s = sig * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * s * s) / s
    return np.asarray(d * f * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi) * np.sqrt(t))


def forward_delta(
    forward: ArrayLike, strike: ArrayLike, maturity: ArrayLike, sigma: ArrayLike, is_call: ArrayLike
) -> FloatArray:
    """Undiscounted forward delta: N(d1) for calls, N(d1) - 1 for puts."""
    f, k, t, sig = (_as_float(x) for x in (forward, strike, maturity, sigma))
    s = sig * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * s * s) / s
    nd1 = ndtr(d1)
    return np.asarray(np.where(np.asarray(is_call, dtype=bool), nd1, nd1 - 1.0), dtype=np.float64)


def no_arbitrage_bounds(
    forward: ArrayLike, strike: ArrayLike, discount: ArrayLike, is_call: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Model-free bounds on a European price: intrinsic < price < D F (call) or D K (put)."""
    f, k, d = (_as_float(x) for x in (forward, strike, discount))
    call = np.asarray(is_call, dtype=bool)
    lower = np.where(call, d * np.maximum(f - k, 0.0), d * np.maximum(k - f, 0.0))
    upper = np.where(call, d * f, d * k)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def implied_vol(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount: ArrayLike,
    is_call: ArrayLike,
    n_iter: int = 100,
) -> FloatArray:
    """Black-76 implied volatility by vectorised bisection on [SIGMA_MIN, SIGMA_MAX].

    The Black price is strictly increasing in sigma, so bisection always
    converges inside the bracket; 100 halvings give a bracket width below
    1e-29. Prices outside the model-free bounds, or outside the prices at the
    bracket ends, return NaN (no silent clipping).

    Args:
        price: Observed option price (discounted).
        forward, strike, maturity, discount, is_call: As in ``black76_price``.
        n_iter: Number of bisection steps.

    Returns:
        Implied volatilities, NaN where no solution exists in the bracket.
    """
    arrays = np.broadcast_arrays(*(_as_float(x) for x in (price, forward, strike, maturity, discount)))
    shape = arrays[0].shape
    p, f, k, t, d = (np.ravel(a) for a in arrays)
    call = np.ravel(np.broadcast_to(np.asarray(is_call, dtype=bool), shape))
    lower, upper = no_arbitrage_bounds(f, k, d, call)
    p_lo = black76_price(f, k, t, d, SIGMA_MIN, call)
    p_hi = black76_price(f, k, t, d, SIGMA_MAX, call)
    ok = (p > lower) & (p < upper) & (p >= p_lo) & (p <= p_hi)
    lo: NDArray[np.float64] = np.full(p.shape, SIGMA_MIN)
    hi: NDArray[np.float64] = np.full(p.shape, SIGMA_MAX)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        above = black76_price(f, k, t, d, mid, call) > p
        hi = np.asarray(np.where(above, mid, hi), dtype=np.float64)
        lo = np.asarray(np.where(above, lo, mid), dtype=np.float64)
    out = 0.5 * (lo + hi)
    out[~ok] = np.nan
    return out.reshape(shape)
