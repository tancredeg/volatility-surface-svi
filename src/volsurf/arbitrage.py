"""Static-arbitrage audit of calibrated slices: butterfly and calendar checks.

Butterfly (one slice): Durrleman's g(k) >= 0 for all k (Gatheral & Jacquier
2014, Lemma 2.2), cross-checked independently by the convexity of Black-76
call prices in strike, d^2 C / dK^2 >= 0 (the risk-neutral density, Breeden &
Litzenberger 1978).

Calendar (two slices T1 < T2): total implied variance must not decrease with
maturity at fixed log-moneyness, w(k, T2) >= w(k, T1) (Gatheral & Jacquier 2014,
Section 2).

Every violation is located either inside the strike range quoted for the
slice(s) (``supported``: it contradicts the data directly) or outside it
(``extrapolated``: it comes from the parametric wings, which a local-vol model
or any wing-sensitive product nevertheless uses).

References:
    Breeden, D. & Litzenberger, R. (1978). Prices of state-contingent claims
        implicit in option prices. Journal of Business, 51(4), 621-651.
    Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
        Quantitative Finance, 14(1), 59-71.
"""

from __future__ import annotations

import numpy as np

from .iv import FloatArray, black76_price
from .svi import SVIParams

G_TOL = 1e-7
W_TOL = 1e-10


def density_negative_points(
    p: SVIParams, forward: float, maturity: float, discount: float, k: FloatArray
) -> int:
    """Number of grid points where the second strike-difference of call prices is negative.

    Uses the non-uniform three-point formula on K = F e^k, so it does not rely
    on any SVI derivative: an independent check of the Durrleman condition.
    """
    K = forward * np.exp(k)
    vol = np.sqrt(np.maximum(p.w(k), 1e-14) / maturity)
    c = black76_price(forward, K, maturity, discount, vol, True)
    h1, h2 = np.diff(K)[:-1], np.diff(K)[1:]
    d2 = 2.0 * (h1 * c[2:] - (h1 + h2) * c[1:-1] + h2 * c[:-2]) / (h1 * h2 * (h1 + h2))
    return int((d2 < -1e-9 * discount).sum())


def butterfly_report(p: SVIParams, k: FloatArray, quoted: tuple[float, float]) -> dict[str, float]:
    """Durrleman g on the grid ``k``: minimum, its location and violations split by quoted range."""
    g = p.g(k)
    inside = (k >= quoted[0]) & (k <= quoted[1])
    neg = g < -G_TOL
    return {
        "g_min": float(g.min()),
        "k_at_g_min": float(k[np.argmin(g)]),
        "neg_points_supported": int((neg & inside).sum()),
        "neg_points_extrapolated": int((neg & ~inside).sum()),
        "max_wing_slope": p.max_wing_slope,
    }


def calendar_report(
    short: SVIParams,
    long: SVIParams,
    k: FloatArray,
    quoted_short: tuple[float, float],
    quoted_long: tuple[float, float],
) -> dict[str, float]:
    """Calendar crossings w_long(k) < w_short(k) on the grid ``k``.

    ``supported`` is the intersection of the two quoted ranges: only there do
    both slices rest on market data.
    """
    diff = long.w(k) - short.w(k)
    lo, hi = max(quoted_short[0], quoted_long[0]), min(quoted_short[1], quoted_long[1])
    inside = (k >= lo) & (k <= hi)
    viol = diff < -W_TOL
    return {
        "viol_points_supported": int((viol & inside).sum()),
        "viol_points_extrapolated": int((viol & ~inside).sum()),
        "k_viol_min": float(k[viol].min()) if viol.any() else float("nan"),
        "k_viol_max": float(k[viol].max()) if viol.any() else float("nan"),
        "max_crossing_w": float(-diff.min()) if viol.any() else 0.0,
    }
