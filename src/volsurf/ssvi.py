"""Surface SVI (SSVI) with power-law phi, calibrated free of static arbitrage.

SSVI (Gatheral & Jacquier 2014, eq. 4.1), with theta_t the ATM total variance:

    w(k, theta) = theta/2 [ 1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2) ],
    phi(theta)  = eta / ( theta^gamma (1 + theta)^(1 - gamma) )      (Remark 4.4).

No-arbitrage, as enforced here:

* Calendar (Theorem 4.1): theta_t non-decreasing in t, and
  0 <= d(theta phi)/dtheta <= (1 + sqrt(1 - rho^2)) phi / rho^2.
  For the power law, d(theta phi)/dtheta = phi (1 - gamma)/(1 + theta), which
  lies in [0, phi) for gamma in [0, 1): the second condition always holds.
* Butterfly (Theorem 4.2, sufficient): theta phi (1 + |rho|) < 4 and
  theta phi^2 (1 + |rho|) <= 4 for all theta > 0. For the power law,
  sup_theta theta phi = eta (as theta -> infinity) and, for 0 < gamma <= 1/2,
  theta phi^2 = eta^2 f(theta) with f(theta) = theta^(1-2 gamma) (1+theta)^(2 gamma-2),
  maximised at theta* = 1 - 2 gamma:  f* = (1-2 gamma)^(1-2 gamma) (2-2 gamma)^(2 gamma-2).
  Both conditions therefore hold for every theta > 0 iff
      eta (1 + |rho|) <= 4   and   eta^2 f*(gamma) (1 + |rho|) <= 4.
  These are enforced exactly. The simpler eta (1 + |rho|) <= 2 of Remark 4.4
  implies them but is more restrictive. With gamma > 1/2, theta phi^2 -> infinity
  as theta -> 0 and no guarantee holds near t = 0, which matters for Dupire.

Each SSVI slice is a raw SVI slice with a = theta(1 - rho^2)/2,
b = theta phi/2, m = -rho/phi, sigma = sqrt(1 - rho^2)/phi.

References:
    Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
        Quantitative Finance, 14(1), 59-71 (arXiv:1204.0646).
    Gatheral, J. Arbitrage-free SVI volatility surfaces, Imperial College
        distinguished lecture slides (power law with gamma in (0, 1/2]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from .iv import FloatArray
from .svi import SVIParams


@dataclass(frozen=True)
class SSVIParams:
    """Global SSVI shape parameters with power-law phi."""

    rho: float
    eta: float
    gamma: float

    def phi(self, theta: ArrayLike) -> FloatArray:
        th = np.asarray(theta, dtype=np.float64)
        return np.asarray(self.eta / (th**self.gamma * (1.0 + th) ** (1.0 - self.gamma)))

    def dphi_dtheta(self, theta: ArrayLike) -> FloatArray:
        """phi'(theta) = phi(theta) [ -gamma/theta - (1 - gamma)/(1 + theta) ]."""
        th = np.asarray(theta, dtype=np.float64)
        return np.asarray(self.phi(th) * (-self.gamma / th - (1.0 - self.gamma) / (1.0 + th)))

    def w(self, k: ArrayLike, theta: ArrayLike) -> FloatArray:
        """Total implied variance w(k, theta)."""
        kk = np.asarray(k, dtype=np.float64)
        th = np.asarray(theta, dtype=np.float64)
        x = self.phi(th) * kk
        r = self.rho
        return np.asarray(0.5 * th * (1.0 + r * x + np.sqrt((x + r) ** 2 + 1.0 - r * r)))

    def dw_dtheta(self, k: ArrayLike, theta: ArrayLike) -> FloatArray:
        """dw/dtheta at fixed k (phi depends on theta)."""
        kk = np.asarray(k, dtype=np.float64)
        th = np.asarray(theta, dtype=np.float64)
        phi = self.phi(th)
        x = phi * kk
        r = self.rho
        root = np.sqrt((x + r) ** 2 + 1.0 - r * r)
        dx = kk * self.dphi_dtheta(th)
        return np.asarray(0.5 * (1.0 + r * x + root) + 0.5 * th * (r + (x + r) / root) * dx)

    def to_svi(self, theta: float) -> SVIParams:
        """Equivalent raw-SVI slice (Gatheral & Jacquier 2014, Section 4)."""
        phi = float(self.phi(theta))
        r = self.rho
        return SVIParams(
            a=0.5 * theta * (1.0 - r * r),
            b=0.5 * theta * phi,
            rho=r,
            m=-r / phi,
            sigma=float(np.sqrt(1.0 - r * r) / phi),
        )

    def butterfly_margins(self, theta: ArrayLike) -> tuple[float, float]:
        """min over theta of 4 - theta phi (1+|rho|) and 4 - theta phi^2 (1+|rho|) (Thm 4.2)."""
        th = np.asarray(theta, dtype=np.float64)
        phi = self.phi(th)
        c = 1.0 + abs(self.rho)
        return float((4.0 - th * phi * c).min()), float((4.0 - th * phi**2 * c).min())

    def calendar_margin(self, theta: ArrayLike) -> float:
        """min over theta of the upper Thm 4.1 bound minus d(theta phi)/dtheta (>= 0 required)."""
        th = np.asarray(theta, dtype=np.float64)
        phi = self.phi(th)
        d_theta_phi = phi + th * self.dphi_dtheta(th)
        r2 = self.rho**2
        upper = np.inf if r2 == 0 else (1.0 + np.sqrt(1.0 - r2)) / r2
        lower_ok = float(d_theta_phi.min())
        return min(lower_ok, float((upper * phi - d_theta_phi).min()))

    @property
    def guaranteed_arbitrage_free(self) -> bool:
        """Theorems 4.1 and 4.2 hold for every theta > 0 (no static arbitrage, all maturities)."""
        if not 0.0 < self.gamma <= 0.5 + 1e-12:
            return False
        return bool(self.eta <= eta_max(self.rho, min(self.gamma, 0.5)) * (1.0 + 1e-9))


def theorem42_fstar(gamma: float) -> float:
    """f* = max_theta theta^(1-2g) (1+theta)^(2g-2) = (1-2g)^(1-2g) (2-2g)^(2g-2), 0 < g <= 1/2."""
    if not 0.0 < gamma <= 0.5:
        raise ValueError("f* is finite only for 0 < gamma <= 1/2.")
    e = 1.0 - 2.0 * gamma
    base = e**e if e > 0 else 1.0
    return float(base * (2.0 - 2.0 * gamma) ** (2.0 * gamma - 2.0))


def eta_max(rho: float, gamma: float) -> float:
    """Largest eta satisfying Theorem 4.2 for all theta > 0 (power-law phi, gamma <= 1/2)."""
    c = 1.0 + abs(rho)
    return float(min(4.0 / c * (1.0 - 1e-9), 2.0 / np.sqrt(c * theorem42_fstar(gamma))))


class ThetaCurve:
    """ATM total variance theta(t): monotone PCHIP through (0, 0) and the calibrated
    nodes, extended linearly beyond the last node with the terminal slope.

    PCHIP preserves monotonicity, so theta'(t) >= 0 and calendar no-arbitrage
    (Theorem 4.1, condition 1) holds for all t, not only at the nodes.
    """

    def __init__(self, maturities: ArrayLike, theta: ArrayLike) -> None:
        t = np.concatenate([[0.0], np.asarray(maturities, dtype=np.float64)])
        th = np.concatenate([[0.0], np.asarray(theta, dtype=np.float64)])
        if np.any(np.diff(t) <= 0) or np.any(np.diff(th) < 0):
            raise ValueError("ThetaCurve needs increasing maturities and non-decreasing theta.")
        self._t_last = float(t[-1])
        self._interp = PchipInterpolator(t, th, extrapolate=False)
        self._d = self._interp.derivative()
        self._th_last = float(th[-1])
        self._slope_last = float(self._d(self._t_last))

    def __call__(self, t: ArrayLike) -> FloatArray:
        tt = np.asarray(t, dtype=np.float64)
        inside = np.asarray(self._interp(np.minimum(tt, self._t_last)))
        return np.asarray(
            np.where(tt > self._t_last, self._th_last + self._slope_last * (tt - self._t_last), inside)
        )

    def derivative(self, t: ArrayLike) -> FloatArray:
        tt = np.asarray(t, dtype=np.float64)
        inside = np.asarray(self._d(np.minimum(tt, self._t_last)))
        return np.asarray(np.where(tt > self._t_last, self._slope_last, inside))


def atm_total_variance(k: ArrayLike, iv: ArrayLike, maturity: float, n_points: int = 8) -> float:
    """Market ATM total variance: quadratic fit of w(k) on the n quotes nearest k = 0, at k = 0."""
    kk = np.asarray(k, dtype=np.float64)
    ww = np.asarray(iv, dtype=np.float64) ** 2 * maturity
    idx = np.argsort(np.abs(kk))[: max(n_points, 3)]
    coef = np.polyfit(kk[idx], ww[idx], 2)
    return float(coef[-1])


@dataclass(frozen=True)
class SSVIFit:
    params: SSVIParams
    maturities: FloatArray
    theta: FloatArray
    rmse: FloatArray  # per slice, decimal vol
    rmse_total: float
    mode: str
    theta_adjusted: bool  # market ATM variance had to be made monotone
    success: bool

    def theta_curve(self) -> ThetaCurve:
        return ThetaCurve(self.maturities, self.theta)

    def implied_vol(self, k: ArrayLike, t: float) -> FloatArray:
        th = float(self.theta_curve()(t))
        return np.asarray(np.sqrt(self.params.w(k, th) / t))


def _eta_from(rho: float, s: float, gamma: float) -> float:
    return s * eta_max(rho, gamma)


def calibrate_ssvi(
    slices: list[tuple[float, FloatArray, FloatArray]],
    *,
    mode: str = "anchored",
    gamma_max: float = 0.5,
    n_starts: int = 8,
    seed: int = 0,
) -> SSVIFit:
    """Calibrate SSVI on several maturities by least squares on implied vol.

    Reparametrisation eta = s eta_max(rho, gamma), s in (0, 1], turns the exact
    Theorem 4.2 conditions (for all theta > 0) into a box bound; with gamma in
    (0, 1/2] the fitted surface is free of static arbitrage at every maturity
    by construction (see module docstring).

    Modes:
        ``anchored``: theta_t fixed at the market ATM total variance of each
            slice (made monotone if needed), then (rho, eta, gamma) fitted.
            The ATM level of every slice is matched by construction.
        ``joint``: theta_t free (non-negative increments) jointly with
            (rho, eta, gamma).

    Args:
        slices: List of (T, k, iv).
        mode: ``anchored`` or ``joint``.
        gamma_max: Upper bound on gamma, at most 1/2 (required by the guarantee).
        n_starts: Multi-start count. seed: RNG seed.
    """
    if not 0.0 < gamma_max <= 0.5:
        raise ValueError("gamma_max must lie in (0, 1/2] for the no-arbitrage guarantee.")
    data = sorted(slices, key=lambda s: s[0])
    mats = np.array([s[0] for s in data])
    ks = [np.asarray(s[1], dtype=np.float64) for s in data]
    ivs = [np.asarray(s[2], dtype=np.float64) for s in data]
    theta_mkt = np.array([atm_total_variance(k, v, T) for (T, k, v) in zip(mats, ks, ivs, strict=True)])
    theta_mono = np.maximum.accumulate(theta_mkt)
    adjusted = bool(np.any(theta_mono > theta_mkt + 1e-15))
    n = len(mats)

    def unpack(x: FloatArray) -> tuple[SSVIParams, FloatArray]:
        rho, s, gamma = float(x[0]), float(x[1]), float(x[2])
        theta = np.cumsum(x[3:]) if mode == "joint" else theta_mono
        return SSVIParams(rho, _eta_from(rho, s, gamma), gamma), theta

    def resid(x: FloatArray) -> FloatArray:
        p, theta = unpack(x)
        out = [
            (np.sqrt(p.w(k, th) / T) - v) * 100.0 for T, k, v, th in zip(mats, ks, ivs, theta, strict=True)
        ]
        return np.concatenate(out)

    lo = [-0.999, 1e-3, 1e-3]
    hi = [0.999, 1.0, gamma_max]
    if mode == "joint":
        incr = np.diff(np.concatenate([[0.0], theta_mono]))
        lo += [1e-10] * n
        hi += [1.0] * n
    elif mode != "anchored":
        raise ValueError(f"Unknown mode {mode!r}.")
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for i in range(n_starts):
        x0 = (
            [-0.6, 0.6, min(0.4, gamma_max)]
            if i == 0
            else [rng.uniform(-0.9, 0.0), rng.uniform(0.1, 1.0), rng.uniform(0.05, gamma_max)]
        )
        if mode == "joint":
            x0 += list(np.maximum(incr, 1e-9))
        res = least_squares(
            resid,
            np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9),
            bounds=(lo, hi),
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=5000,
        )
        cost = float(np.sum(res.fun**2))
        if cost < best_cost:
            best, best_cost = res, cost
    assert best is not None
    p, theta = unpack(np.asarray(best.x))
    per = []
    for T, k, v, th in zip(mats, ks, ivs, theta, strict=True):
        per.append(float(np.sqrt(np.mean((np.sqrt(p.w(k, th) / T) - v) ** 2))))
    all_err = np.concatenate(
        [np.sqrt(p.w(k, th) / T) - v for T, k, v, th in zip(mats, ks, ivs, theta, strict=True)]
    )
    return SSVIFit(
        params=p,
        maturities=mats,
        theta=np.asarray(theta, dtype=np.float64),
        rmse=np.array(per),
        rmse_total=float(np.sqrt(np.mean(all_err**2))),
        mode=mode,
        theta_adjusted=adjusted,
        success=bool(best.success),
    )
