"""Raw SVI slices: total variance, Durrleman's butterfly function and calibration.

Raw SVI (Gatheral 2004), in log-moneyness k = ln(K/F) and total implied
variance w = sigma_BS^2 T:

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ].

A slice is free of butterfly arbitrage iff Durrleman's function

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2

is non-negative for all k (Gatheral & Jacquier 2014, eq. 2.1 and Lemma 2.2);
the risk-neutral density is p(k) = g(k)/sqrt(2 pi w) exp(-d_-(k)^2/2).
Roger Lee's moment formula bounds the wing slopes of w by 2:
b (1 + |rho|) <= 2.

References:
    Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
        parametrization with application to the valuation of volatility derivatives.
    Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
        Quantitative Finance, 14(1), 59-71.
    Lee, R. (2004). The moment formula for implied volatility at extreme strikes.
        Mathematical Finance, 14(3), 469-480.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares, lsq_linear, minimize

from .iv import FloatArray

PARAM_NAMES = ("a", "b", "rho", "m", "sigma")
LOWER = np.array([-0.5, 1e-4, -0.999, -1.0, 1e-4])
UPPER = np.array([0.5, 2.0, 0.999, 1.0, 2.0])
K_CONSTRAINT: FloatArray = np.linspace(-1.5, 1.5, 301, dtype=np.float64)
G_MARGIN = 1e-4  # g >= G_MARGIN on the constraint grid keeps g >= 0 between grid points
K_CHECK: FloatArray = np.linspace(-1.5, 1.5, 3001, dtype=np.float64)
W_FLOOR = 1e-10
SCALE = np.array([1e-2, 1e-1, 1.0, 1e-1, 1e-1])
FEAS_TOL = 1e-7


def durrleman_g(k: ArrayLike, w: ArrayLike, dw: ArrayLike, d2w: ArrayLike) -> FloatArray:
    """Durrleman's g(k) from total variance and its first two k-derivatives."""
    kk, ww, w1, w2 = (np.asarray(x, dtype=np.float64) for x in (k, w, dw, d2w))
    return np.asarray((1.0 - kk * w1 / (2.0 * ww)) ** 2 - 0.25 * w1**2 * (1.0 / ww + 0.25) + 0.5 * w2)


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters (a, b, rho, m, sigma)."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> FloatArray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma], dtype=np.float64)

    @classmethod
    def from_array(cls, x: ArrayLike) -> SVIParams:
        a, b, rho, m, sigma = (float(v) for v in np.asarray(x, dtype=np.float64))
        return cls(a, b, rho, m, sigma)

    def w(self, k: ArrayLike) -> FloatArray:
        """Total implied variance w(k)."""
        dk = np.asarray(k, dtype=np.float64) - self.m
        return np.asarray(self.a + self.b * (self.rho * dk + np.sqrt(dk * dk + self.sigma**2)))

    def dw(self, k: ArrayLike) -> FloatArray:
        """w'(k) = b [rho + (k - m)/sqrt((k - m)^2 + sigma^2)]."""
        dk = np.asarray(k, dtype=np.float64) - self.m
        return np.asarray(self.b * (self.rho + dk / np.sqrt(dk * dk + self.sigma**2)))

    def d2w(self, k: ArrayLike) -> FloatArray:
        """w''(k) = b sigma^2 / ((k - m)^2 + sigma^2)^(3/2)."""
        dk = np.asarray(k, dtype=np.float64) - self.m
        return np.asarray(self.b * self.sigma**2 / (dk * dk + self.sigma**2) ** 1.5)

    def g(self, k: ArrayLike) -> FloatArray:
        """Durrleman's butterfly function on this slice."""
        return durrleman_g(k, self.w(k), self.dw(k), self.d2w(k))

    def implied_vol(self, k: ArrayLike, maturity: float) -> FloatArray:
        """sigma_BS(k) = sqrt(w(k)/T); NaN where w <= 0 (no silent clipping)."""
        w = self.w(k)
        out = np.full_like(w, np.nan)
        pos = w > 0
        out[pos] = np.sqrt(w[pos] / maturity)
        return out

    @property
    def min_total_variance(self) -> float:
        """min_k w(k) = a + b sigma sqrt(1 - rho^2)."""
        return float(self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2))

    @property
    def max_wing_slope(self) -> float:
        """Largest asymptotic wing slope b (1 + |rho|); Lee's bound is 2."""
        return float(self.b * (1.0 + abs(self.rho)))


@dataclass(frozen=True)
class SVIFit:
    params: SVIParams
    maturity: float
    rmse: float  # implied-vol RMSE, decimal (0.01 = 1 vol point)
    success: bool
    constrained: bool
    at_bound: tuple[str, ...]
    g_min: float  # min of g on K_CHECK
    n_points: int


def _model_vol(x: FloatArray, k: FloatArray, maturity: float) -> FloatArray:
    w = SVIParams.from_array(x).w(k)
    return np.asarray(np.sqrt(np.maximum(w, W_FLOOR) / maturity))


def _quasi_explicit_inner(
    m: float, sigma: float, k: FloatArray, w: FloatArray, weight: FloatArray
) -> tuple[float, FloatArray]:
    """Inner step of the quasi-explicit calibration (Zeliade 2009).

    For fixed (m, sigma) and y = (k - m)/sigma, z = sqrt(y^2 + 1), raw SVI reads
    w = a + u (z + y)/2 + v (z - y)/2 with u = b sigma (1 + rho), v = b sigma (1 - rho).
    It is linear in (a, u, v); the box 0 <= u, v <= 2 sigma encodes b > 0, |rho| <= 1 and
    Lee's bound b (1 +/- rho) <= 2. Solved exactly by bounded linear least squares.
    """
    y = (k - m) / sigma
    z = np.sqrt(y * y + 1.0)
    sw = np.sqrt(weight)
    A = np.column_stack([np.ones_like(y), 0.5 * (z + y), 0.5 * (z - y)]) * sw[:, None]
    rhs = w * sw
    coef, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    if not (0.0 <= coef[1] <= 2.0 * sigma and 0.0 <= coef[2] <= 2.0 * sigma):
        coef = lsq_linear(A, rhs, bounds=([-np.inf, 0.0, 0.0], [np.inf, 2.0 * sigma, 2.0 * sigma])).x
    a, u, v = (float(c) for c in coef)
    u, v = max(u, 1e-10), max(v, 1e-10)
    resid = A @ coef - rhs
    b = (u + v) / (2.0 * sigma)
    rho = (u - v) / (u + v)
    return float(resid @ resid), np.array([a, b, rho, m, sigma])


def _vol_jacobian(x: FloatArray, k: FloatArray, maturity: float) -> FloatArray:
    """d sigma_SVI / d(a, b, rho, m, sigma) with sigma_SVI = sqrt(w/T), times 100 (vol points)."""
    a, b, rho, m, sig = x
    dk = k - m
    root = np.sqrt(dk * dk + sig * sig)
    w = np.maximum(a + b * (rho * dk + root), W_FLOOR)
    dw = np.column_stack([np.ones_like(k), rho * dk + root, b * dk, -b * (rho + dk / root), b * sig / root])
    return np.asarray(dw * (100.0 / (2.0 * np.sqrt(w * maturity)))[:, None])


def _initial_guesses(k: FloatArray, iv: FloatArray, maturity: float, n_starts: int) -> list[FloatArray]:
    """Quasi-explicit starts: grid over (m, sigma) with m inside the data range, then a
    bounded Nelder-Mead refinement of the ``n_starts`` best grid points.

    The inner problem is weighted by 1/(2 sigma_mkt T)^2 so that its residual is the
    first-order expansion of the implied-vol residual (same loss as the final fit).
    """
    w = iv**2 * maturity
    weight = 1.0 / (2.0 * iv * maturity) ** 2
    m_lo, m_hi = float(k.min()), float(k.max())
    m_grid = np.linspace(m_lo, m_hi, 9)
    s_grid = np.array([0.01, 0.03, 0.07, 0.15, 0.3, 0.6])
    scored = sorted(
        ((_quasi_explicit_inner(m, s, k, w, weight)[0], m, s) for m in m_grid for s in s_grid),
        key=lambda t: t[0],
    )

    def outer(v: FloatArray) -> float:
        return _quasi_explicit_inner(float(v[0]), float(np.exp(v[1])), k, w, weight)[0]

    starts = []
    for _, m0, s0 in scored[: max(n_starts, 1)]:
        res = minimize(
            outer,
            np.array([m0, np.log(s0)]),
            method="Nelder-Mead",
            bounds=[(m_lo, m_hi), (np.log(5e-3), np.log(1.5))],
            options={"xatol": 1e-7, "fatol": 1e-14, "maxiter": 1000},
        )
        _, x = _quasi_explicit_inner(float(res.x[0]), float(np.exp(res.x[1])), k, w, weight)
        starts.append(np.clip(x, LOWER + 1e-9, UPPER - 1e-9))
    return starts


def _safe_starts(k: FloatArray, iv: FloatArray, maturity: float, lower: SVIParams | None) -> list[SVIParams]:
    """Near-feasible starting slices for the constrained fit.

    Without a previous slice: a nearly flat smile at the market ATM total
    variance (g close to 1). With a previous slice: that slice scaled by the
    ratio of ATM total variances (w >= w_prev when the ratio is >= 1) and the
    same slice shifted up by the ATM difference. Feasibility is not assumed;
    SLSQP restores it and every result is re-checked.
    """
    idx = np.argsort(np.abs(k))[:8]
    theta = float(np.mean(iv[idx] ** 2) * maturity)
    if lower is None:
        return [SVIParams(a=theta * 0.9, b=1e-3, rho=-0.3, m=0.0, sigma=0.1)]
    w0 = float(lower.w(0.0))
    lam = max(theta / w0, 1.0) if w0 > 0 else 1.0
    scaled = SVIParams(lower.a * lam, lower.b * lam, lower.rho, lower.m, lower.sigma)
    shifted = SVIParams(lower.a + max(theta - w0, 1e-6), lower.b, lower.rho, lower.m, lower.sigma)
    return [scaled, shifted]


def _at_bound(x: FloatArray) -> tuple[str, ...]:
    tol = 1e-6 * (UPPER - LOWER)
    hit = (x - LOWER < tol) | (UPPER - x < tol)
    return tuple(n for n, h in zip(PARAM_NAMES, hit, strict=True) if h)


def svi_constraints(x: FloatArray, lower: SVIParams | None, k_grid: FloatArray) -> FloatArray:
    """Inequality constraints c(x) >= 0: Durrleman g on a grid, Lee's wing bound,
    non-negative minimum variance and, optionally, no calendar crossing with the
    previous (shorter) slice."""
    p = SVIParams.from_array(x)
    parts = [
        p.g(k_grid) - G_MARGIN,
        np.array([2.0 - p.b * (1.0 + p.rho), 2.0 - p.b * (1.0 - p.rho), p.min_total_variance]),
    ]
    if lower is not None:
        parts.append(p.w(k_grid) - lower.w(k_grid))
    return np.concatenate(parts)


PENALTY_WEIGHTS = (1e1, 1e2, 1e3, 1e4, 1e5)


def _penalty_start(
    x0: FloatArray, resid: Callable[[FloatArray], FloatArray], lower: SVIParams | None
) -> FloatArray:
    """Quadratic-penalty continuation from the unconstrained optimum.

    Minimises ||resid(x)||^2 + lambda ||min(c(x), 0)||^2 for increasing lambda,
    each stage started from the previous one, with ``least_squares`` (TRF).
    The end point violates the constraints by O(1/lambda) and is used to start
    SLSQP, which then restores exact feasibility locally.
    """
    x = np.clip(np.asarray(x0, dtype=np.float64), LOWER + 1e-9, UPPER - 1e-9)
    for lam in PENALTY_WEIGHTS:
        scale = float(np.sqrt(lam))

        def aug(z: FloatArray, s: float = scale) -> FloatArray:
            return np.concatenate([resid(z), s * np.minimum(svi_constraints(z, lower, K_CONSTRAINT), 0.0)])

        x = np.asarray(
            least_squares(
                aug,
                x,
                bounds=(LOWER, UPPER),
                method="trf",
                x_scale=SCALE,
                ftol=1e-12,
                xtol=1e-12,
                max_nfev=300,
            ).x
        )
    return np.clip(x, LOWER + 1e-9, UPPER - 1e-9)


def is_feasible(p: SVIParams, lower: SVIParams | None = None, tol: float = FEAS_TOL) -> bool:
    """Exact post-check on the dense grid K_CHECK (finer than the constraint grid)."""
    ok = p.g(K_CHECK).min() >= -tol and p.max_wing_slope <= 2.0 + tol and p.min_total_variance >= -tol
    if lower is not None:
        ok = ok and float((p.w(K_CHECK) - lower.w(K_CHECK)).min()) >= -tol
    return bool(ok)


def calibrate_svi(
    k: ArrayLike,
    iv: ArrayLike,
    maturity: float,
    *,
    constrained: bool = False,
    lower: SVIParams | None = None,
    extra_starts: Sequence[SVIParams] = (),
    n_starts: int = 4,
) -> SVIFit:
    """Fit raw SVI by least squares on implied volatility.

    Loss: sum_i (sigma_SVI(k_i) - sigma_mkt(k_i))^2, identical for every model
    in this package so that RMSEs are comparable.

    * ``constrained=False``: box bounds only (scipy ``least_squares``, TRF).
    * ``constrained=True``: SLSQP with g(k) >= 0 on a grid of 151 points over
      [-1.5, 1.5], Lee's bound b(1 +/- rho) <= 2, min_k w >= 0 and, if
      ``lower`` is given, w(k) >= w_prev(k) on the same grid (no calendar
      arbitrage with the previous slice). g is constrained on 301 points over
      [-1.5, 1.5] with a margin of 1e-4; feasibility is re-checked on 3001
      points. SLSQP is started from the raw fit, from near-feasible slices built
      from the previous slice (scaled and shifted up to the market ATM level) or
      a near-flat slice, and from ``extra_starts``; the best feasible result wins.

    Args:
        k: Log-moneyness ln(K/F) of the quotes.
        iv: Market implied vols (decimal).
        maturity: T in years.
        constrained: Enforce static no-arbitrage (see above).
        lower: Calibrated previous (shorter) slice, for the calendar constraint.
        extra_starts: Additional starting points (e.g. the SSVI slice, which is
            arbitrage-free and therefore a feasible start for the constrained fit).
        n_starts: Number of quasi-explicit starts (the procedure is deterministic).

    Returns:
        SVIFit with parameters, RMSE (decimal vol), feasibility diagnostics.
    """
    kk = np.asarray(k, dtype=np.float64)
    vv = np.asarray(iv, dtype=np.float64)
    starts = _initial_guesses(kk, vv, maturity, n_starts)

    def resid(x: FloatArray) -> FloatArray:
        return np.asarray((_model_vol(x, kk, maturity) - vv) * 100.0)

    best_x: FloatArray | None = None
    best_cost = np.inf
    for x0 in starts:
        res = least_squares(
            resid,
            x0,
            jac=lambda x: _vol_jacobian(x, kk, maturity),
            bounds=(LOWER, UPPER),
            method="trf",
            x_scale=SCALE,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=500,
        )
        cost = float(np.sum(res.fun**2))
        if cost < best_cost:
            best_cost, best_x = cost, np.asarray(res.x)
    assert best_x is not None
    success = True

    if constrained:
        # SLSQP in scaled variables u = x / SCALE (parameters differ by two orders of magnitude).
        # SLSQP alone is erratic here (it can end in a feasible point 10x worse than the optimum
        # depending on the start), so its main start is the end of a penalty continuation, which
        # is already near-feasible and near-optimal; the other starts are kept as safeguards.
        cons = {"type": "ineq", "fun": lambda u: svi_constraints(u * SCALE, lower, K_CONSTRAINT)}
        feasible: list[tuple[float, FloatArray]] = []
        fallback: tuple[float, FloatArray] | None = None
        candidates = [*extra_starts, *_safe_starts(kk, vv, maturity, lower)]
        for x0 in [
            _penalty_start(best_x, resid, lower),
            *(np.clip(p.as_array(), LOWER + 1e-9, UPPER - 1e-9) for p in candidates),
            best_x,
            starts[0],
        ]:
            res = minimize(
                lambda u: float(np.sum(resid(u * SCALE) ** 2)),
                x0 / SCALE,
                method="SLSQP",
                bounds=list(zip(LOWER / SCALE, UPPER / SCALE, strict=True)),
                constraints=[cons],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            x = np.clip(np.asarray(res.x) * SCALE, LOWER, UPPER)
            cost = float(np.sum(resid(x) ** 2))
            if is_feasible(SVIParams.from_array(x), lower):
                feasible.append((cost, x))
            elif fallback is None or cost < fallback[0]:
                fallback = (cost, x)
        if feasible:
            best_cost, best_x = min(feasible, key=lambda t: t[0])
        else:
            assert fallback is not None
            best_cost, best_x = fallback
            success = False

    params = SVIParams.from_array(best_x)
    rmse = float(np.sqrt(np.mean((_model_vol(best_x, kk, maturity) - vv) ** 2)))
    return SVIFit(
        params=params,
        maturity=maturity,
        rmse=rmse,
        success=success,
        constrained=constrained,
        at_bound=_at_bound(best_x),
        g_min=float(params.g(K_CHECK).min()),
        n_points=len(kk),
    )


def calibrate_svi_surface(
    slices: list[tuple[float, FloatArray, FloatArray]],
    *,
    constrained: bool,
    extra_starts: Sequence[SVIParams | None] | None = None,
    n_starts: int = 4,
) -> list[SVIFit]:
    """Calibrate slices from short to long maturity.

    With ``constrained=True`` each slice is also forced above the previous
    calibrated slice (calendar no-arbitrage), so the set of slices is free of
    static arbitrage on the constraint grid.

    Args:
        slices: List of (T, k, iv), sorted by maturity.
        extra_starts: Optional extra start per slice (same order), e.g. SSVI slices.
    """
    out: list[SVIFit] = []
    lower: SVIParams | None = None
    extras = list(extra_starts) if extra_starts is not None else [None] * len(slices)
    for (T, k, iv), extra in zip(slices, extras, strict=True):
        fit = calibrate_svi(
            k,
            iv,
            T,
            constrained=constrained,
            lower=lower if constrained else None,
            extra_starts=() if extra is None else (extra,),
            n_starts=n_starts,
        )
        out.append(fit)
        lower = fit.params
    return out
