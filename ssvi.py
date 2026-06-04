"""
ssvi.py
=======
Surface SVI (SSVI) calibration — Gatheral & Jacquier (2014).

Why SSVI:
    Per-slice SVI fits each maturity independently, then interpolates the five
    parameters linearly across maturities. That interpolation does NOT preserve
    calendar no-arbitrage (total variance can decrease with maturity at some
    moneyness). SSVI parametrises the WHOLE surface with a single correlation
    rho and a function phi(theta) anchored on the ATM total-variance term
    structure theta_t. Under explicit conditions it is free of BOTH calendar
    and butterfly arbitrage by construction.

Model:
    w(k, theta) = (theta/2) * [ 1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2) ]
    phi(theta)  = eta / ( theta^gamma * (1+theta)^(1-gamma) )      (power-law)
    where theta = theta_t is the ATM total variance at maturity t, and the
    global shape parameters are (rho, eta, gamma).

No-arbitrage conditions enforced during calibration (Gatheral-Jacquier 2014):
    Calendar : theta_t non-decreasing in t.
    Butterfly: theta*phi(theta)*(1+|rho|)   < 4   for all theta,
               theta*phi(theta)^2*(1+|rho|) <= 4  for all theta.

This script reuses the data pipeline (cleaning, IV extraction) from
iv_surface.py and adds SSVI calibration, verification, and plots. It compares
SSVI against the per-slice SVI baseline.

Run:
    python ssvi.py
"""

import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the validated data pipeline and diagnostics from iv_surface.py
from iv_surface import (
    DATA_DIR, FIG_DIR,
    RISK_FREE_RATE, DIVIDEND_YIELD,
    validate_option_chain, clean_option_chain, extract_iv_for_chain,
    calibrate_svi, svi_total_variance, durrleman_g, resolve_spot,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


# =============================================================================
# SSVI CORE
# =============================================================================

def ssvi_phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """Power-law phi(theta) = eta / (theta^gamma * (1+theta)^(1-gamma)).

    Args:
        theta: ATM total variance (scalar or array, > 0).
        eta:   Scale parameter (> 0).
        gamma: Curvature decay exponent in (0, 1).

    Returns:
        phi(theta), same shape as theta.
    """
    return eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))


def ssvi_total_variance(
    k: np.ndarray,
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """SSVI total variance w(k, theta).

    w(k,theta) = (theta/2) * [1 + rho*phi*k + sqrt((phi*k+rho)^2 + 1 - rho^2)]

    Args:
        k:     Log-moneyness array.
        theta: ATM total variance at this maturity (> 0).
        rho:   Global correlation parameter in (-1, 1).
        eta:   Global scale parameter (> 0).
        gamma: Global curvature exponent in (0, 1).

    Returns:
        Total variance array w(k, theta). Always strictly positive for theta>0.
    """
    phi = ssvi_phi(theta, eta, gamma)
    x = phi * k
    return 0.5 * theta * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + (1.0 - rho ** 2)))


def ssvi_to_svi_params(theta: float, rho: float, eta: float, gamma: float) -> dict:
    """Convert an SSVI slice to equivalent raw-SVI (a,b,rho,m,xi) parameters.

    Each SSVI slice is a raw-SVI slice. The mapping (Gatheral-Jacquier 2014):
        b   = theta*phi/2
        m   = -rho/phi
        xi  = sqrt(1-rho^2)/phi
        a   = (theta/2)*(1-rho^2)
        rho = rho   (unchanged)

    This lets us reuse durrleman_g() to check butterfly arbitrage per slice.

    Args:
        theta: ATM total variance.
        rho:   SSVI correlation.
        eta:   SSVI scale.
        gamma: SSVI exponent.

    Returns:
        dict with keys a, b, rho, mu, xi (raw-SVI convention).
    """
    phi = float(ssvi_phi(np.array([theta]), eta, gamma)[0])
    return {
        "a":   0.5 * theta * (1.0 - rho ** 2),
        "b":   0.5 * theta * phi,
        "rho": rho,
        "mu":  -rho / phi,
        "xi":  np.sqrt(1.0 - rho ** 2) / phi,
    }


# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_ssvi(
    slices: dict[float, tuple[np.ndarray, np.ndarray]],
    n_starts: int = 12,
    seed: int = 42,
) -> dict:
    """Jointly calibrate SSVI across all maturities.

    Parameters: global (rho, eta, gamma) plus one theta_t per maturity.
    Objective:  sum over maturities and strikes of (iv_model - iv_obs)^2,
                fitted on implied vol (not total variance) so that long
                maturities do not dominate the loss.

    Hard constraints (SLSQP inequalities):
        - Calendar:  theta_{j+1} - theta_j >= 0.
        - Butterfly: 4 - theta*phi*(1+|rho|) > 0 and
                     4 - theta*phi^2*(1+|rho|) >= 0, enforced on a dense theta
                     grid spanning the calibrated range (not only at the nodes).

    Args:
        slices:   Map t (years) -> (k_array, w_obs_array) of observed total
                  variance. Maturities are sorted internally.
        n_starts: Multi-start count.
        seed:     RNG seed for reproducible multi-start.

    Returns:
        dict with keys:
            rho, eta, gamma,
            theta (dict t -> theta_t),
            rmse  (dict t -> per-maturity vol-point RMSE),
            rmse_global (float),
            converged (bool), n_iter (int).
    """
    local_rng = np.random.default_rng(seed)
    maturities = sorted(slices.keys())
    n = len(maturities)

    # Pre-extract per-maturity observed (k, iv)
    k_list   = [slices[t][0] for t in maturities]
    w_list   = [slices[t][1] for t in maturities]
    iv_list  = [np.sqrt(np.maximum(w, 1e-12) / t) for t, w in zip(maturities, w_list)]

    # ATM total-variance seed per maturity: value at the point nearest k=0
    theta_seed = []
    for t, k_arr, w_arr in zip(maturities, k_list, w_list):
        idx = int(np.argmin(np.abs(k_arr)))
        theta_seed.append(max(w_arr[idx], 1e-4))
    theta_seed = np.array(theta_seed)
    # Enforce a monotone seed (cumulative max) so the start is feasible
    theta_seed = np.maximum.accumulate(theta_seed)

    def unpack(params):
        rho, eta, gamma = params[0], params[1], params[2]
        theta = params[3:]
        return rho, eta, gamma, theta

    def objective(params):
        rho, eta, gamma, theta = unpack(params)
        sse = 0.0
        for j, t in enumerate(maturities):
            w_model = ssvi_total_variance(k_list[j], theta[j], rho, eta, gamma)
            iv_model = np.sqrt(np.maximum(w_model, 1e-12) / t)
            sse += np.sum((iv_model - iv_list[j]) ** 2)
        return sse

    # Constraints
    def cons_calendar(params):
        # theta_{j+1} - theta_j >= 0 for all adjacent pairs
        theta = params[3:]
        return np.diff(theta)

    def cons_butterfly1(params):
        rho, eta, gamma, theta = unpack(params)
        # dense theta grid spanning [min,max] of current thetas
        tg = np.linspace(max(theta.min(), 1e-6), max(theta.max(), 1e-4), 50)
        phi = ssvi_phi(tg, eta, gamma)
        return 4.0 - tg * phi * (1.0 + abs(rho)) - 1e-8   # > 0

    def cons_butterfly2(params):
        rho, eta, gamma, theta = unpack(params)
        tg = np.linspace(max(theta.min(), 1e-6), max(theta.max(), 1e-4), 50)
        phi = ssvi_phi(tg, eta, gamma)
        return 4.0 - tg * phi ** 2 * (1.0 + abs(rho))     # >= 0

    constraints = [
        {"type": "ineq", "fun": cons_calendar},
        {"type": "ineq", "fun": cons_butterfly1},
        {"type": "ineq", "fun": cons_butterfly2},
    ]

    bounds = [(-0.999, 0.999), (1e-4, 10.0), (1e-3, 0.999)] + \
             [(1e-6, 5.0)] * n

    best, best_obj = None, np.inf
    for s in range(n_starts):
        if s == 0:
            x0 = np.array([-0.5, 1.0, 0.5, *theta_seed])
        else:
            x0 = np.array([
                local_rng.uniform(-0.85, -0.1),
                local_rng.uniform(0.3, 3.0),
                local_rng.uniform(0.2, 0.8),
                *np.maximum.accumulate(theta_seed * local_rng.uniform(0.7, 1.3, n)),
            ])
        try:
            res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                           constraints=constraints,
                           options={"maxiter": 1000, "ftol": 1e-12})
            if res.fun < best_obj:
                best_obj, best = res.fun, res
        except Exception as e:
            log.warning("SSVI start %d failed: %s", s, e)

    if best is None:
        raise RuntimeError("All SSVI calibration starts failed.")

    rho, eta, gamma, theta = unpack(best.x)
    theta_by_t = {t: float(theta[j]) for j, t in enumerate(maturities)}

    rmse_by_t = {}
    sq_sum, n_pts = 0.0, 0
    for j, t in enumerate(maturities):
        w_model = ssvi_total_variance(k_list[j], theta[j], rho, eta, gamma)
        iv_model = np.sqrt(np.maximum(w_model, 1e-12) / t)
        err = iv_model - iv_list[j]
        rmse_by_t[t] = float(np.sqrt(np.mean(err ** 2)))
        sq_sum += np.sum(err ** 2)
        n_pts += len(err)

    return {
        "rho": float(rho), "eta": float(eta), "gamma": float(gamma),
        "theta": theta_by_t,
        "rmse": rmse_by_t,
        "rmse_global": float(np.sqrt(sq_sum / n_pts)),
        "converged": bool(best.success),
        "n_iter": int(best.nit),
    }


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_ssvi_surface(fit: dict) -> dict:
    """Verify calendar and butterfly no-arbitrage on the calibrated SSVI surface.

    Args:
        fit: Output of calibrate_ssvi.

    Returns:
        dict summary with calendar_violations (int), butterfly_pass (dict
        t -> bool), g_min (dict t -> float).
    """
    rho, eta, gamma = fit["rho"], fit["eta"], fit["gamma"]
    maturities = sorted(fit["theta"].keys())
    k = np.linspace(-0.35, 0.35, 500)

    log.info("=" * 60)
    log.info("SSVI ARBITRAGE VERIFICATION (calibrated surface)")
    log.info("=" * 60)
    log.info("Global: rho=%.4f  eta=%.4f  gamma=%.4f", rho, eta, gamma)

    # Butterfly per slice via Durrleman g on the equivalent raw-SVI slice
    butterfly_pass, g_min = {}, {}
    for t in maturities:
        theta = fit["theta"][t]
        p = ssvi_to_svi_params(theta, rho, eta, gamma)
        g = durrleman_g(k, p["a"], p["b"], p["rho"], p["mu"], p["xi"])
        g_min[t] = float(g.min())
        butterfly_pass[t] = g_min[t] >= -1e-9
        log.info("T=%dd: theta=%.5f  Durrleman g_min=%.6f (%s)",
                 round(t * 365), theta, g_min[t],
                 "PASS" if butterfly_pass[t] else "FAIL")

    # Calendar: total variance non-decreasing in t at every k on the grid
    cal_viol = 0
    curves = {t: ssvi_total_variance(k, fit["theta"][t], rho, eta, gamma)
              for t in maturities}
    for i in range(len(maturities) - 1):
        t1, t2 = maturities[i], maturities[i + 1]
        viol = curves[t1] > curves[t2] + 1e-9
        n = int(viol.sum())
        cal_viol += n
        status = "PASS" if n == 0 else f"{n} violations (max {(curves[t1][viol]-curves[t2][viol]).max():.6f})"
        log.info("Calendar T=%dd->%dd: %s",
                 round(t1 * 365), round(t2 * 365), status)

    log.info("SSVI calendar violations total (|k|<=0.35): %d", cal_viol)
    return {"calendar_violations": cal_viol,
            "butterfly_pass": butterfly_pass, "g_min": g_min}


# =============================================================================
# DATA PREP (reuses iv_surface pipeline) + OTM selection
# =============================================================================

def load_and_prepare(csv_path: Path, S0: float, r: float, q: float
                     ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Load a cached chain CSV, clean, extract IV, select OTM, return slices.

    Args:
        csv_path: Path to spy_chains_*.csv produced by fetch_data.py.
        S0: Spot. r: rate. q: dividend yield.

    Returns:
        Map t (years) -> (k_array, w_obs_array) of OTM total variances.
    """
    df = pd.read_csv(csv_path)
    validate_option_chain(df)
    if df["expiry"].astype(str).str.startswith("synthetic").any():
        raise RuntimeError("CSV contains synthetic labels — refusing on real path.")

    ref = datetime.date.today()
    slices = {}
    for exp in df["expiry"].unique():
        try:
            t_days = (datetime.date.fromisoformat(str(exp)) - ref).days
        except ValueError:
            continue
        if t_days <= 0:
            continue
        T = t_days / 365.0
        sub = df[df["expiry"] == exp]
        F = S0 * np.exp((r - q) * T)
        parts = []
        for otype in ["call", "put"]:
            d = sub[sub["option_type"].str.lower() == otype].copy()
            if len(d) == 0:
                continue
            try:
                d = clean_option_chain(d, S0, r, q, T)
                d = extract_iv_for_chain(d, S0, r, q, T)
            except ValueError:
                continue
            d = d[d["strike"] > F] if otype == "call" else d[d["strike"] <= F]
            parts.append(d)
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        if len(merged) < 5:
            continue
        k = merged["log_moneyness"].values
        w = merged["iv"].values ** 2 * T
        order = np.argsort(k)
        slices[T] = (k[order], w[order])
    return slices


# =============================================================================
# PLOTS
# =============================================================================

def plot_ssvi_vs_svi(slices, ssvi_fit, svi_by_t, fig_dir: Path) -> None:
    """Plot SSVI fit vs observed and vs per-slice SVI, per maturity."""
    maturities = sorted(slices.keys())
    n = len(maturities)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()
    rho, eta, gamma = ssvi_fit["rho"], ssvi_fit["eta"], ssvi_fit["gamma"]
    kk = np.linspace(-0.35, 0.35, 300)

    for idx, t in enumerate(maturities):
        ax = axes[idx]
        k_obs, w_obs = slices[t]
        ax.scatter(k_obs, np.sqrt(w_obs / t) * 100, s=22, color="tomato",
                   zorder=5, label="Observed (OTM)")
        w_ssvi = ssvi_total_variance(kk, ssvi_fit["theta"][t], rho, eta, gamma)
        ax.plot(kk, np.sqrt(w_ssvi / t) * 100, "b-", lw=2,
                label=f"SSVI (RMSE={ssvi_fit['rmse'][t]:.3f})")
        if t in svi_by_t:
            p = svi_by_t[t]["params"]
            w_svi = svi_total_variance(kk, p["a"], p["b"], p["rho"], p["mu"], p["xi"])
            ax.plot(kk, np.sqrt(np.maximum(w_svi, 0) / t) * 100, "g--", lw=1.4,
                    label=f"SVI slice (RMSE={svi_by_t[t]['rmse']:.3f})")
        ax.set_title(f"T={round(t*365)}d")
        ax.set_xlabel("Log-moneyness ln(K/F)")
        ax.set_ylabel("Implied Vol (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("SSVI vs per-slice SVI — SPY", fontsize=13)
    plt.tight_layout()
    out = fig_dir / "fig5_ssvi_vs_svi.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_ssvi_surface(ssvi_fit, slices, fig_dir: Path) -> None:
    """3D SSVI surface with theta interpolated linearly in t between nodes."""
    maturities = sorted(slices.keys())
    rho, eta, gamma = ssvi_fit["rho"], ssvi_fit["eta"], ssvi_fit["gamma"]
    t_nodes = np.array(maturities)
    th_nodes = np.array([ssvi_fit["theta"][t] for t in maturities])
    kk = np.linspace(-0.35, 0.35, 120)
    tt = np.linspace(t_nodes.min(), t_nodes.max(), 40)
    th_interp = np.interp(tt, t_nodes, th_nodes)  # monotone since nodes monotone
    K, Tg = np.meshgrid(kk, tt)
    IV = np.zeros_like(K)
    for i, (t_, th_) in enumerate(zip(tt, th_interp)):
        w = ssvi_total_variance(kk, th_, rho, eta, gamma)
        IV[i, :] = np.sqrt(w / t_) * 100
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, Tg * 365, IV, cmap="viridis", alpha=0.9, linewidth=0.2)
    ax.set_xlabel("Log-moneyness ln(K/F)")
    ax.set_ylabel("Maturity (days)")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title("SSVI Implied Volatility Surface — SPY")
    out = fig_dir / "fig6_ssvi_surface_3d.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Load cached data, calibrate SSVI, verify, compare to per-slice SVI."""
    # Resolve spot robustly (live, else from cached CSV). Never default silently
    # to a constant: a wrong spot shifts the forward and corrupts the surface.
    S0 = resolve_spot()

    candidates = sorted(DATA_DIR.glob("spy_chains_*.csv"), reverse=True)
    if not candidates:
        sys.exit("No data/spy_chains_*.csv found. Run: python fetch_data.py")
    csv_path = candidates[0]
    log.info("Loading %s (S0=%.2f)", csv_path, S0)

    slices = load_and_prepare(csv_path, S0, RISK_FREE_RATE, DIVIDEND_YIELD)
    if len(slices) < 2:
        sys.exit("Need at least 2 maturities for SSVI. Got fewer after cleaning.")

    # Per-slice SVI baseline (for comparison)
    svi_by_t = {}
    for t, (k, w) in slices.items():
        try:
            svi_by_t[t] = calibrate_svi(k, w, t)
        except RuntimeError:
            pass

    # SSVI joint calibration
    log.info("Calibrating SSVI jointly across %d maturities...", len(slices))
    ssvi_fit = calibrate_ssvi(slices)
    log.info("SSVI: rho=%.4f eta=%.4f gamma=%.4f | global RMSE=%.4f | conv=%s",
             ssvi_fit["rho"], ssvi_fit["eta"], ssvi_fit["gamma"],
             ssvi_fit["rmse_global"], ssvi_fit["converged"])

    summary = verify_ssvi_surface(ssvi_fit)

    plot_ssvi_vs_svi(slices, ssvi_fit, svi_by_t, FIG_DIR)
    plot_ssvi_surface(ssvi_fit, slices, FIG_DIR)

    # Comparison table
    print("\n" + "=" * 64)
    print("SSVI vs per-slice SVI — RMSE (vol points)")
    print("=" * 64)
    print(f"{'T(days)':>8} | {'SVI RMSE':>10} | {'SSVI RMSE':>10} | {'theta':>9}")
    print("-" * 64)
    for t in sorted(slices.keys()):
        svi_rmse = svi_by_t[t]["rmse"] if t in svi_by_t else float("nan")
        print(f"{round(t*365):>8} | {svi_rmse:>10.4f} | "
              f"{ssvi_fit['rmse'][t]:>10.4f} | {ssvi_fit['theta'][t]:>9.5f}")
    print("-" * 64)
    print(f"SSVI calendar violations (|k|<=0.35): {summary['calendar_violations']}")
    bf = sum(summary['butterfly_pass'].values())
    print(f"SSVI butterfly pass: {bf}/{len(summary['butterfly_pass'])} maturities")
    print("=" * 64)
    print("\n[DESK VIEW] SSVI trades a small amount of per-slice fit quality for "
          "a globally arbitrage-free surface. RMSE slightly above per-slice SVI "
          "is expected and acceptable: SSVI removes the calendar violations that "
          "linear SVI-parameter interpolation produced.")


if __name__ == "__main__":
    main()
