"""Discounting, put-call-parity forwards and a dividend-aware forward curve.

Conventions
-----------
* Rates are continuously compounded, ACT/365.
* The forward of each listed expiry is read from the option quotes themselves
  (put-call parity), so no spot price or dividend yield enters the
  implied-volatility computation.
* The forward curve between expiries (needed only by the barrier pricer) is
  F(0, t) = (S0 - sum_{t_j <= t} div_j e^{-r t_j}) e^{r t}, with S0 implied
  from the parity forward of a reference expiry.

References: Hull, J. Options, Futures, and Other Derivatives, ch. 5 and 17.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def rate_from_tbill_discount(discount_rate: float, days: int = 91) -> float:
    """Continuously compounded ACT/365 rate from a T-bill discount yield.

    The bill price is P = 1 - d n / 360; the continuous rate solving
    P = exp(-r n / 365) is r = -ln(P) 365 / n.

    Args:
        discount_rate: Discount-basis yield d (decimal), e.g. FRED DTB3 / 100.
        days: Days to maturity n of the bill.

    Returns:
        Continuously compounded rate r.
    """
    price = 1.0 - discount_rate * days / 360.0
    if price <= 0.0:
        raise ValueError(f"Invalid discount rate {discount_rate}: bill price {price} <= 0.")
    return float(-np.log(price) * 365.0 / days)


def discount_factor(rate: float, maturity: ArrayLike) -> FloatArray:
    """D(T) = exp(-r T)."""
    return np.asarray(np.exp(-rate * np.asarray(maturity, dtype=np.float64)), dtype=np.float64)


@dataclass(frozen=True)
class ForwardEstimate:
    """Forward implied from put-call parity on near-the-money strikes."""

    forward: float
    n_pairs: int
    dispersion: float  # p90 - p10 of the per-strike forwards, in price units
    band: float


def parity_forward(quotes: pd.DataFrame, discount: float, band: float = 0.03) -> ForwardEstimate:
    """Forward from put-call parity C - P = D (F - K) on near-the-money strikes.

    For each strike quoted two-sided in both a call and a put, the parity
    forward is F_K = K + (C_mid - P_mid) / D. The estimate is the median of
    F_K over strikes with |K / K0 - 1| <= band, where K0 minimises |C - P|.

    SPY options are American: parity is approximate. Restricting to
    near-the-money strikes keeps the early-exercise premium small; the
    dispersion of F_K is returned as a quality diagnostic.

    Args:
        quotes: One expiry; columns option_type ('call'/'put'), strike, bid, ask.
        discount: Discount factor D to the expiry.
        band: Relative half-width of the strike window around K0.

    Returns:
        ForwardEstimate.

    Raises:
        ValueError: if fewer than 3 call/put pairs are available in the band.
    """
    q = quotes[(quotes["bid"] > 0) & (quotes["ask"] > quotes["bid"])]
    mid = 0.5 * (q["bid"] + q["ask"])
    calls = mid[q["option_type"] == "call"].groupby(q.loc[q["option_type"] == "call", "strike"]).mean()
    puts = mid[q["option_type"] == "put"].groupby(q.loc[q["option_type"] == "put", "strike"]).mean()
    both = pd.concat([calls.rename("C"), puts.rename("P")], axis=1).dropna()
    if len(both) < 3:
        raise ValueError("Fewer than 3 strikes quoted in both calls and puts: no parity forward.")
    k0 = float((both["C"] - both["P"]).abs().idxmin())
    strikes = both.index.to_numpy(dtype=np.float64)
    near = np.abs(strikes / k0 - 1.0) <= band
    if near.sum() < 3:
        raise ValueError(f"Fewer than 3 call/put pairs within {band:.0%} of K0={k0}.")
    fk = strikes[near] + (both["C"].to_numpy()[near] - both["P"].to_numpy()[near]) / discount
    return ForwardEstimate(
        forward=float(np.median(fk)),
        n_pairs=int(near.sum()),
        dispersion=float(np.quantile(fk, 0.9) - np.quantile(fk, 0.1)),
        band=band,
    )


@dataclass(frozen=True)
class CashDividend:
    """A cash dividend: ex-date as a year fraction from the valuation time."""

    time: float
    amount: float


def implied_spot(forward: float, maturity: float, rate: float, dividends: Sequence[CashDividend]) -> float:
    """Spot consistent with a forward: S0 = F D(T) + sum_{t_j <= T} div_j e^{-r t_j}."""
    pv_div = sum(d.amount * np.exp(-rate * d.time) for d in dividends if 0.0 < d.time <= maturity)
    return float(forward * np.exp(-rate * maturity) + pv_div)


@dataclass(frozen=True)
class ForwardCurve:
    """F(0, t) = (S0 - sum_{0 < t_j <= t} div_j e^{-r t_j}) e^{r t}.

    Dividends are cash amounts at their ex-dates; the curve is
    right-continuous (post-dividend at t_j). ``left`` gives the left limit.
    """

    spot: float
    rate: float
    dividends: tuple[CashDividend, ...] = ()

    def _pv_div(self, t: FloatArray, inclusive: bool) -> FloatArray:
        out = np.zeros_like(t)
        for d in self.dividends:
            if d.time <= 0.0:
                continue
            hit = (t >= d.time) if inclusive else (t > d.time)
            out = out + np.where(hit, d.amount * np.exp(-self.rate * d.time), 0.0)
        return out

    def __call__(self, t: ArrayLike) -> FloatArray:
        tt = np.asarray(t, dtype=np.float64)
        return np.asarray((self.spot - self._pv_div(tt, True)) * np.exp(self.rate * tt))

    def left(self, t: ArrayLike) -> FloatArray:
        """Left limit F(0, t-), i.e. the pre-dividend forward at an ex-date."""
        tt = np.asarray(t, dtype=np.float64)
        return np.asarray((self.spot - self._pv_div(tt, False)) * np.exp(self.rate * tt))
