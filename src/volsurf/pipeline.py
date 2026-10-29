"""End-to-end study: calibration, fit quality, arbitrage audit, timing, stability, local vol and barriers.

Every number published in the README is produced by ``run_study`` and written to
``results/<snapshot>/`` by ``scripts/run_results.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .arbitrage import butterfly_report, calendar_report, density_negative_points
from .barrier import (
    BarrierMC,
    bs_down_and_in_put,
    bs_down_and_out_call,
    flat_vol,
    mc_barrier,
    simulation_times,
)
from .data import PreparedSnapshot
from .forward import ForwardCurve, implied_spot
from .iv import FloatArray, black76_price, implied_vol
from .localvol import LocalVolGrid, build_ssvi_local_vol_grid, ssvi_local_variance
from .ssvi import SSVIFit, calibrate_ssvi
from .svi import K_CHECK, SVIFit, SVIParams, calibrate_svi_surface

MODELS = ("SVI raw", "SVI constrained", "SSVI")


@dataclass(frozen=True)
class StudyConfig:
    """Settings of the published study (``fast()`` gives a lighter variant for tests)."""

    n_paths: int = 400_000
    steps_per_day: int = 4
    barrier_levels: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90, 0.95)
    strike_level: float = 1.0  # strike as a fraction of the initial spot (term-sheet convention)
    product_target_years: float = 0.25  # the listed expiry closest to 3 months is used
    n_timing_runs: int = 10
    n_noise_runs: int = 50
    noise_vol: float = 0.001  # 0.10 vol point, the order of the SVI residuals
    seed: int = 20260603

    @classmethod
    def fast(cls) -> StudyConfig:
        return cls(n_paths=40_000, steps_per_day=2, n_timing_runs=2, n_noise_runs=4)


@dataclass
class Calibration:
    svi_raw: list[SVIFit]
    svi_con: list[SVIFit]
    ssvi: SSVIFit
    seconds: dict[str, float] = field(default_factory=dict)

    def slice_params(self, model: str) -> list[SVIParams]:
        if model == "SVI raw":
            return [f.params for f in self.svi_raw]
        if model == "SVI constrained":
            return [f.params for f in self.svi_con]
        return [self.ssvi.params.to_svi(float(th)) for th in self.ssvi.theta]


def _data(snap: PreparedSnapshot) -> list[tuple[float, FloatArray, FloatArray]]:
    return [(s.maturity, s.k, s.iv_mid) for s in snap.slices]


def calibrate_all(
    snap: PreparedSnapshot, data: list[tuple[float, FloatArray, FloatArray]] | None = None
) -> Calibration:
    """Raw SVI, constrained SVI and SSVI, calibrated independently on the same quotes."""
    d = data if data is not None else _data(snap)
    t0 = time.perf_counter()
    ssvi = calibrate_ssvi(d, mode="anchored")
    t1 = time.perf_counter()
    raw = calibrate_svi_surface(d, constrained=False)
    t2 = time.perf_counter()
    con = calibrate_svi_surface(d, constrained=True)
    t3 = time.perf_counter()
    return Calibration(raw, con, ssvi, {"SSVI": t1 - t0, "SVI raw": t2 - t1, "SVI constrained": t3 - t2})


def fit_table(snap: PreparedSnapshot, cal: Calibration) -> pd.DataFrame:
    """Per maturity and model: RMSE and max error in vol points, share of quotes priced inside bid-ask."""
    rows = []
    for i, s in enumerate(snap.slices):
        row: dict[str, object] = {
            "expiry": s.expiry,
            "T_days": s.maturity * 365.0,
            "quotes": len(s.k),
            "k_min": s.k_range[0],
            "k_max": s.k_range[1],
            "atm_vol_pct": float(np.sqrt(cal.ssvi.theta[i] / s.maturity) * 100),
            "half_spread_vp": float(np.median(s.iv_ask - s.iv_bid) * 50),
        }
        for model in MODELS:
            p = cal.slice_params(model)[i]
            err = (np.sqrt(np.maximum(p.w(s.k), 1e-12) / s.maturity) - s.iv_mid) * 100
            model_iv = np.sqrt(np.maximum(p.w(s.k), 1e-12) / s.maturity)
            row[f"rmse_vp[{model}]"] = float(np.sqrt(np.mean(err**2)))
            row[f"maxerr_vp[{model}]"] = float(np.abs(err).max())
            row[f"in_spread[{model}]"] = float(np.mean((model_iv >= s.iv_bid) & (model_iv <= s.iv_ask)))
        rows.append(row)
    return pd.DataFrame(rows)


def arbitrage_table(snap: PreparedSnapshot, cal: Calibration) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Butterfly (Durrleman g and an independent price-convexity check) and calendar audits.

    All checks run on k in [-1.5, 1.5] (3001 points). Each violation is
    classified as inside the quoted range (``supported``) or outside it
    (extrapolation), because only the former contradicts the data directly.
    """
    k = K_CHECK
    bfly, cal_rows = [], []
    for model in MODELS:
        params = cal.slice_params(model)
        for s, p in zip(snap.slices, params, strict=True):
            bfly.append(
                {
                    "model": model,
                    "expiry": s.expiry,
                    "T_days": s.maturity * 365.0,
                    **butterfly_report(p, k, s.k_range),
                    "density_neg_points": density_negative_points(p, s.forward, s.maturity, s.discount, k),
                }
            )
        for i in range(len(snap.slices) - 1):
            s1, s2 = snap.slices[i], snap.slices[i + 1]
            cal_rows.append(
                {
                    "model": model,
                    "pair": f"{s1.maturity * 365:.0f}d->{s2.maturity * 365:.0f}d",
                    **calendar_report(params[i], params[i + 1], k, s1.k_range, s2.k_range),
                }
            )
    return pd.DataFrame(bfly), pd.DataFrame(cal_rows)


def ssvi_surface_check(fit: SSVIFit, t_max: float, n_t: int = 400) -> dict[str, float]:
    """Continuous-in-time audit of the SSVI surface on (0, t_max]: min g, min dw/dt, min local variance."""
    theta = fit.theta_curve()
    k = np.linspace(-1.5, 1.5, 601)
    g_min, dwdt_min, lv_min = np.inf, np.inf, np.inf
    for t in np.geomspace(1e-4, t_max, n_t):
        th = float(theta(t))
        sl = fit.params.to_svi(th)
        g_min = min(g_min, float(sl.g(k).min()))
        dwdt_min = min(dwdt_min, float((fit.params.dw_dtheta(k, th) * theta.derivative(t)).min()))
        lv_min = min(lv_min, float(ssvi_local_variance(fit.params, theta, float(t), k).min()))
    return {
        "g_min": g_min,
        "dw_dt_min": dwdt_min,
        "local_var_min": lv_min,
        "guaranteed": float(fit.params.guaranteed_arbitrage_free),
    }


def timing_table(snap: PreparedSnapshot, n_runs: int) -> pd.DataFrame:
    """Median, min and max wall-clock seconds of each full-surface calibration."""
    runs: dict[str, list[float]] = {m: [] for m in MODELS}
    for _ in range(n_runs):
        cal = calibrate_all(snap)
        for m in MODELS:
            runs[m].append(cal.seconds[m])
    n_slices = len(snap.slices)
    return pd.DataFrame(
        [
            {
                "model": m,
                "slices": n_slices,
                "median_s": float(np.median(v)),
                "min_s": float(np.min(v)),
                "max_s": float(np.max(v)),
                "median_ms_per_slice": float(np.median(v) / n_slices * 1000),
                "runs": n_runs,
            }
            for m, v in runs.items()
        ]
    )


def stability_table(snap: PreparedSnapshot, n_runs: int, noise_vol: float, seed: int) -> pd.DataFrame:
    """Recalibrate on quotes perturbed by N(0, noise_vol^2) in implied vol; report the
    dispersion (std, in vol points) of ATM vol, ATM skew and extrapolated wings, per model."""
    rng = np.random.default_rng(seed)
    probes = {"atm": 0.0, "wing_put_-0.5": -0.5, "wing_call_+0.3": 0.3}
    rec: dict[tuple[str, int, str], list[float]] = {}
    bounds_hit: dict[str, int] = dict.fromkeys(MODELS, 0)
    for _ in range(n_runs):
        data = [(s.maturity, s.k, s.iv_mid + rng.normal(0.0, noise_vol, size=s.k.shape)) for s in snap.slices]
        cal = calibrate_all(snap, data)
        bounds_hit["SVI raw"] += sum(bool(f.at_bound) for f in cal.svi_raw)
        bounds_hit["SVI constrained"] += sum(bool(f.at_bound) for f in cal.svi_con)
        for m in MODELS:
            for i, (s, p) in enumerate(zip(snap.slices, cal.slice_params(m), strict=True)):
                for name, kk in probes.items():
                    rec.setdefault((m, i, name), []).append(
                        float(np.sqrt(max(float(p.w(kk)), 1e-12) / s.maturity))
                    )
                skew = float(p.dw(0.0)) / (2.0 * np.sqrt(float(p.w(0.0)) * s.maturity))
                rec.setdefault((m, i, "atm_skew"), []).append(skew)
    rows = []
    for m in MODELS:
        for i, s in enumerate(snap.slices):
            rows.append(
                {
                    "model": m,
                    "expiry": s.expiry,
                    "T_days": s.maturity * 365.0,
                    "std_atm_vp": float(np.std(rec[(m, i, "atm")]) * 100),
                    "std_atm_skew": float(np.std(rec[(m, i, "atm_skew")])),
                    "std_wing_put_vp": float(np.std(rec[(m, i, "wing_put_-0.5")]) * 100),
                    "std_wing_call_vp": float(np.std(rec[(m, i, "wing_call_+0.3")]) * 100),
                    "runs": n_runs,
                    "fits_at_bound": bounds_hit[m],
                }
            )
    return pd.DataFrame(rows)


@dataclass
class BarrierStudy:
    expiry: str
    maturity: float
    spot: float
    spot_dispersion: float
    strike: float
    forward_T: float
    discount: float
    flat_vol: float
    div_yield_equiv: float
    lv_grid: LocalVolGrid
    lv: BarrierMC
    flat: BarrierMC
    table: pd.DataFrame
    repricing: pd.DataFrame


def barrier_study(snap: PreparedSnapshot, fit: SSVIFit, cfg: StudyConfig) -> BarrierStudy:
    """Down-and-out calls and down-and-in puts: Dupire local vol (from SSVI) vs flat BS vol.

    The flat vol is the implied vol at the strike (what a naive desk would plug
    in). Both models are priced with the same Monte-Carlo engine, paths and
    forward curve, so the gap isolates the volatility model. The flat-vol
    prices are also checked against the closed forms with the equivalent
    continuous dividend yield.
    """
    idx = int(np.argmin([abs(s.maturity - cfg.product_target_years) for s in snap.slices]))
    sl = snap.slices[idx]
    T, F, D, r = sl.maturity, sl.forward, sl.discount, snap.meta.rate
    divs = snap.meta.cash_dividends()
    S0 = implied_spot(F, T, r, divs)
    spots = [implied_spot(s.forward, s.maturity, r, divs) for s in snap.slices]
    curve = ForwardCurve(S0, r, divs)
    K = cfg.strike_level * S0
    k_strike = float(np.log(K / F))
    sigma_flat = float(fit.implied_vol(np.array([k_strike]), T)[0])
    q_eq = r - float(np.log(F / S0)) / T
    times = simulation_times(T, 365 * cfg.steps_per_day, [d.time for d in divs])
    grid = build_ssvi_local_vol_grid(fit.params, fit.theta_curve(), times, np.linspace(-1.5, 1.0, 501))
    barriers = [lvl * S0 for lvl in cfg.barrier_levels]
    common = {
        "strike": K,
        "barriers": barriers,
        "maturity": T,
        "discount": D,
        "forward": curve,
        "times": times,
        "n_paths": cfg.n_paths,
        "seed": cfg.seed,
    }
    lv = mc_barrier(vol=grid, **common)  # type: ignore[arg-type]
    fl = mc_barrier(vol=flat_vol(sigma_flat), **common)  # type: ignore[arg-type]
    rows = []
    for j, lvl in enumerate(cfg.barrier_levels):
        H = barriers[j]
        sigma_h = float(fit.implied_vol(np.array([np.log(H / F)]), T)[0])
        rows.append(
            {
                "barrier_pct": lvl * 100,
                "doc_lv": lv.doc_call[j].value,
                "doc_lv_se": lv.doc_call[j].stderr,
                "doc_flat": fl.doc_call[j].value,
                "doc_flat_se": fl.doc_call[j].stderr,
                "doc_flat_closed_form": bs_down_and_out_call(S0, K, H, T, r, q_eq, sigma_flat),
                "dip_lv": lv.dip_put[j].value,
                "dip_lv_se": lv.dip_put[j].stderr,
                "dip_flat": fl.dip_put[j].value,
                "dip_flat_se": fl.dip_put[j].stderr,
                "dip_flat_closed_form": bs_down_and_in_put(S0, K, H, T, r, q_eq, sigma_flat),
                "knock_prob_lv": float(lv.knock_prob[j]),
                "knock_prob_flat": float(fl.knock_prob[j]),
                "barrier_vol_pct": sigma_h * 100,
                "doc_barrier_vol_closed_form": bs_down_and_out_call(S0, K, H, T, r, q_eq, sigma_h),
                "dip_barrier_vol_closed_form": bs_down_and_in_put(S0, K, H, T, r, q_eq, sigma_h),
            }
        )
    table = pd.DataFrame(rows)
    table["doc_gap_pct"] = (table["doc_lv"] / table["doc_flat"] - 1.0) * 100
    table["dip_gap_pct"] = (table["dip_lv"] / table["dip_flat"] - 1.0) * 100
    table["doc_gap_pct_of_spot"] = (table["doc_lv"] - table["doc_flat"]) / S0 * 100
    table["dip_gap_pct_of_spot"] = (table["dip_lv"] - table["dip_flat"]) / S0 * 100
    # A reverse convertible is long a bond and short the DIP: underpricing the DIP by x% of notional
    # overstates the coupon the issuer can pay by x% / T per annum.
    table["coupon_overstatement_pa_pct"] = table["dip_gap_pct_of_spot"] / T
    rep = []
    for lvl in (0.80, 0.90, 1.00, 1.10):
        Kx = lvl * S0
        is_call = lvl >= 1.0
        kx = float(np.log(Kx / F))
        vol_s = float(fit.implied_vol(np.array([kx]), T)[0])
        target = float(black76_price(F, Kx, T, D, vol_s, is_call))
        est = lv.vanilla(Kx, is_call)
        iv_mc = float(implied_vol(est.value, F, Kx, T, D, is_call))
        mkt = _market_vol(sl, kx)
        rep.append(
            {
                "strike_pct": lvl * 100,
                "type": "call" if is_call else "put",
                "ssvi_vol_pct": vol_s * 100,
                "market_vol_pct": mkt * 100,
                "lv_mc_vol_pct": iv_mc * 100,
                "lv_minus_ssvi_vp": (iv_mc - vol_s) * 100,
                "price_ssvi": target,
                "price_lv_mc": est.value,
                "price_se": est.stderr,
                "error_in_se": (est.value - target) / est.stderr,
            }
        )
    return BarrierStudy(
        expiry=sl.expiry,
        maturity=T,
        spot=S0,
        spot_dispersion=float(np.ptp(spots)),
        strike=K,
        forward_T=F,
        discount=D,
        flat_vol=sigma_flat,
        div_yield_equiv=q_eq,
        lv_grid=grid,
        lv=lv,
        flat=fl,
        table=table,
        repricing=pd.DataFrame(rep),
    )


def _market_vol(sl: object, k: float) -> float:
    """Market mid vol linearly interpolated in k (NaN outside the quoted range)."""
    kk = sl.k  # type: ignore[attr-defined]
    iv = sl.iv_mid  # type: ignore[attr-defined]
    if k < kk[0] or k > kk[-1]:
        return float("nan")
    return float(np.interp(k, kk, iv))


@dataclass
class StudyResults:
    snapshot: PreparedSnapshot
    calibration: Calibration
    fit: pd.DataFrame
    butterfly: pd.DataFrame
    calendar: pd.DataFrame
    ssvi_check: dict[str, float]
    timing: pd.DataFrame
    stability: pd.DataFrame
    barrier: BarrierStudy
    config: StudyConfig


def run_study(snap: PreparedSnapshot, cfg: StudyConfig | None = None) -> StudyResults:
    cfg = cfg or StudyConfig()
    cal = calibrate_all(snap)
    bfly, calendar = arbitrage_table(snap, cal)
    return StudyResults(
        snapshot=snap,
        calibration=cal,
        fit=fit_table(snap, cal),
        butterfly=bfly,
        calendar=calendar,
        ssvi_check=ssvi_surface_check(cal.ssvi, float(snap.maturities.max())),
        timing=timing_table(snap, cfg.n_timing_runs),
        stability=stability_table(snap, cfg.n_noise_runs, cfg.noise_vol, cfg.seed),
        barrier=barrier_study(snap, cal.ssvi, cfg),
        config=cfg,
    )
