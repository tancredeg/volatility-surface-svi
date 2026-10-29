"""Dupire local volatility from a total-implied-variance surface.

In log-moneyness y = ln(K / F(0, T)) and total implied variance w(y, T),
Gatheral (2006, eq. 1.10) writes the local variance as

    v_L(y, T) = (dw/dT) / [ 1 - (y/w) w' + (1/4)(-1/4 - 1/w + y^2/w^2) w'^2 + (1/2) w'' ],

where primes are y-derivatives. The denominator equals Durrleman's g(y), so

    v_L = (dw/dT) / g.

On an SSVI surface w(y, T) = W(y, theta(T)), dw/dT = (dW/dtheta) theta'(T). With
theta' >= 0 (calendar) and g > 0 (butterfly), v_L >= 0 everywhere: an
arbitrage-free surface is exactly what makes Dupire well defined.

The local vol is used at y = ln(S_t / F(0, t)), i.e. with the spot measured in
units of its own forward.

References:
    Dupire, B. (1994). Pricing with a smile. Risk, 7(1), 18-20.
    Gatheral, J. (2006). The Volatility Surface. Wiley, ch. 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .iv import FloatArray
from .ssvi import SSVIParams, ThetaCurve
from .svi import durrleman_g


def local_variance(
    y: ArrayLike, w: ArrayLike, dw_dy: ArrayLike, d2w_dy2: ArrayLike, dw_dt: ArrayLike
) -> FloatArray:
    """v_L = (dw/dT) / g(y) (Gatheral 2006, eq. 1.10)."""
    return np.asarray(np.asarray(dw_dt, dtype=np.float64) / durrleman_g(y, w, dw_dy, d2w_dy2))


def ssvi_local_variance(params: SSVIParams, theta: ThetaCurve, t: float, y: ArrayLike) -> FloatArray:
    """Local variance of an SSVI surface at time t > 0 on a grid of log-moneyness y."""
    if t <= 0.0:
        raise ValueError("Local variance is evaluated at t > 0.")
    yy = np.asarray(y, dtype=np.float64)
    th = float(theta(t))
    slice_ = params.to_svi(th)
    dw_dt = params.dw_dtheta(yy, th) * float(theta.derivative(t))
    return local_variance(yy, slice_.w(yy), slice_.dw(yy), slice_.d2w(yy), dw_dt)


@dataclass(frozen=True)
class LocalVolGrid:
    """Local vol tabulated at the mid-point of each Monte-Carlo step on a grid of y.

    ``vol[i]`` applies to the step [times[i], times[i+1]]; values are linearly
    interpolated in y and held flat beyond the grid ends.
    """

    times: FloatArray
    y: FloatArray
    vol: FloatArray
    n_clipped: int

    def __call__(self, step: int, y: FloatArray) -> FloatArray:
        return np.asarray(np.interp(y, self.y, self.vol[step]))


def build_ssvi_local_vol_grid(
    params: SSVIParams,
    theta: ThetaCurve,
    times: ArrayLike,
    y_grid: ArrayLike,
    vol_floor: float = 0.01,
    vol_cap: float = 3.0,
) -> LocalVolGrid:
    """Tabulate sqrt(v_L) at step mid-points; values outside [floor, cap] are clipped and counted."""
    tt = np.asarray(times, dtype=np.float64)
    yy = np.asarray(y_grid, dtype=np.float64)
    mids = 0.5 * (tt[:-1] + tt[1:])
    rows = []
    n_clipped = 0
    for t in mids:
        v = ssvi_local_variance(params, theta, float(t), yy)
        vol = np.sqrt(np.maximum(v, 0.0))
        bad = (vol < vol_floor) | (vol > vol_cap) | ~np.isfinite(vol)
        n_clipped += int(bad.sum())
        rows.append(np.clip(np.nan_to_num(vol, nan=vol_floor), vol_floor, vol_cap))
    return LocalVolGrid(times=tt, y=yy, vol=np.vstack(rows), n_clipped=n_clipped)
