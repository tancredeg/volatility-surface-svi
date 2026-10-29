"""Phase 0 audit diagnostics on the frozen SPY snapshots (no network).

Every number quoted in AUDIT.md section "Diagnostics chiffrés" is printed by
this script. Usage:

    python scripts/audit/audit_diagnostics.py data/raw out/diagnostics

Conventions (stated, not hidden):
    - Snapshot time of the 2026-06-03 file = max(lastTradeDate) in the file.
      Yahoo quotes are delayed; the true quote time is unknown to within
      ~15-20 min. Sensitivity of T to this choice is < 0.1 % at 28 days.
    - Expiry time = 16:00 America/New_York on the expiry date (SPY options
      are PM-settled). T = ACT/365 in calendar time.
    - Forward F and discount factor D per expiry from put-call parity on
      near-the-money strikes:  C - P = D (F - K)   (OLS of C-P on K).
      SPY options are American: parity is an approximation; restricting to
      |K/F - 1| <= 3 % keeps early-exercise premia small.
    - Implied vols from Black-76 on the forward: no spot, no r, no q needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

NY = "America/New_York"


def black76(F: float, K: np.ndarray, T: float, D: float, sigma: np.ndarray, is_call: np.ndarray):
    """Undiscounted-forward Black (1976) price, discounted by D."""
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s**2) / s
    d2 = d1 - s
    call = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def iv_black76(price: float, F: float, K: float, T: float, D: float, is_call: bool) -> float:
    """Implied vol by Brent on [1e-4, 5]; NaN if the price is outside no-arb bounds."""
    intrinsic = D * max(F - K, 0.0) if is_call else D * max(K - F, 0.0)
    upper = D * F if is_call else D * K
    if not (intrinsic < price < upper):
        return np.nan

    def f(s: float) -> float:
        return float(black76(F, np.array([K]), T, D, np.array([s]), np.array([is_call]))[0]) - price

    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-10)
    except ValueError:
        return np.nan


def year_fraction(expiry: str, t0: pd.Timestamp) -> float:
    t_exp = pd.Timestamp(f"{expiry} 16:00", tz=NY).tz_convert("UTC")
    return (t_exp - t0).total_seconds() / (365.0 * 86400.0)


# External discount rate (public source, frozen with the sample):
# US 3-month bill auction of 2026-06-15, high rate 3.64 % (discount basis),
# reported by Newsquawk. Used ONLY for the discount factor D; the forward is
# taken from the option quotes themselves. Sensitivity reported below.
R_EXT = 0.0364


def parity_forward(df: pd.DataFrame, T: float, band: float = 0.03) -> dict:
    """Forward from put-call parity on near-ATM strikes with two-sided quotes.

    Two estimators are reported:
      - OLS of (C - P) on K: identifies D and F jointly. On asynchronous
        Yahoo quotes D comes out > 1 (negative implied rate): D is NOT
        identifiable from this data, which is itself an audit finding.
      - Fixed-D estimator: D = exp(-R_EXT T), F = median_K [K + (C - P)/D].
        This is the forward used downstream.
    """
    q = df[(df.bid > 0) & (df.ask > 0) & (df.ask >= df.bid)].copy()
    q["mid"] = 0.5 * (q.bid + q.ask)
    c = q[q.option_type == "call"].set_index("strike")["mid"]
    p = q[q.option_type == "put"].set_index("strike")["mid"]
    both = pd.concat([c.rename("C"), p.rename("P")], axis=1).dropna()
    k_atm = (both.C - both.P).abs().idxmin()
    near = both[np.abs(both.index.values / k_atm - 1) <= band]
    K = near.index.values.astype(float)
    y = (near.C - near.P).values
    A = np.vstack([np.ones_like(K), -K]).T
    (dF, D_ols), *_ = np.linalg.lstsq(A, y, rcond=None)
    D = float(np.exp(-R_EXT * T))
    F_each = K + y / D
    F = float(np.median(F_each))
    return {
        "F": F,
        "D": D,
        "F_dispersion_usd_p90_p10": float(np.quantile(F_each, 0.9) - np.quantile(F_each, 0.1)),
        "D_ols": float(D_ols),
        "r_implied_ols": float(-np.log(D_ols) / T),
        "F_ols": float(dF / D_ols),
        "n_pairs": int(len(K)),
        "k_atm_guess": float(k_atm),
    }


def main() -> None:
    data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    # ------------------------------------------------------------------ 1. files
    files = sorted(data_dir.glob("spy_chains_*.csv"))
    out["files"] = {}
    for f in files:
        d = pd.read_csv(f)
        ltd = pd.to_datetime(d.lastTradeDate, utc=True)
        out["files"][f.name] = {
            "rows": len(d),
            "columns": list(d.columns),
            "has_underlying_spot": "underlying_spot" in d.columns,
            "share_bid_ask_zero": float(((d.bid == 0) & (d.ask == 0)).mean()),
            "last_trade_min_utc": str(ltd.min()),
            "last_trade_max_utc": str(ltd.max()),
            "expiries": sorted(d.expiry.unique().tolist()),
        }

    # ------------------------------------------- 2. forward & discount (06-03)
    d3 = pd.read_csv(data_dir / "spy_chains_20260603.csv")
    t0 = pd.to_datetime(d3.lastTradeDate, utc=True).max()
    out["snapshot_time_utc_0603"] = str(t0)
    fwd = {}
    for e in sorted(d3.expiry.unique()):
        T = year_fraction(e, t0)
        fwd[e] = {"T": T, "T_days": T * 365, **parity_forward(d3[d3.expiry == e], T)}
    exps = sorted(fwd)
    carry = {}
    for e1, e2 in zip(exps[:-1], exps[1:], strict=False):
        carry[f"{e1}->{e2}"] = float(np.log(fwd[e2]["F"] / fwd[e1]["F"]) / (fwd[e2]["T"] - fwd[e1]["T"]))
    out["forward_0603"] = fwd
    out["implied_carry_between_expiries"] = carry
    out["repo_assumed_carry_r_minus_q"] = 0.053 - 0.013

    # Spot implied by the repo convention, if F were exactly right at the
    # first expiry: S = F1 * exp(-(r-q) T1). Then repo forwards at later
    # expiries vs parity forwards -> moneyness shift in log terms.
    S_impl = fwd[exps[0]]["F"] * np.exp(-0.04 * fwd[exps[0]]["T"])
    out["repo_forward_error_logpts_if_anchored_on_first_expiry"] = {
        e: float(np.log(S_impl * np.exp(0.04 * fwd[e]["T"]) / fwd[e]["F"])) for e in exps
    }

    # ------------------------------ 3. call/put IV gap at same strikes (06-03)
    rows = []
    q = d3[(d3.bid > 0) & (d3.ask > 0) & (d3.ask >= d3.bid)].copy()
    q["mid"] = 0.5 * (q.bid + q.ask)
    q["spread_rel"] = (q.ask - q.bid) / q["mid"]
    for e in exps:
        F, D, T = fwd[e]["F"], fwd[e]["D"], fwd[e]["T"]
        sub = q[q.expiry == e].copy()
        sub["iv"] = [
            iv_black76(m, F, K, T, D, ot == "call")
            for m, K, ot in zip(sub["mid"], sub.strike, sub.option_type, strict=False)
        ]
        sub["k"] = np.log(sub.strike / F)
        sub["T"] = T
        rows.append(sub)
        piv = sub.pivot_table(index="strike", columns="option_type", values="iv")
        near = piv[np.abs(np.log(piv.index.values / F)) <= 0.03].dropna()
        out.setdefault("callput_iv_gap_bp_near_atm_parityF", {})[e] = {
            "n": int(len(near)),
            "median_abs_bp": float((near.call - near.put).abs().median() * 1e4),
            "max_abs_bp": float((near.call - near.put).abs().max() * 1e4),
        }
    allq = pd.concat(rows)
    allq.to_csv(out_dir / "iv_0603_black76_parityF.csv", index=False)

    # OTM smile, spread-filtered, as a clean calibration set
    F_row = allq.expiry.map({e: fwd[e]["F"] for e in exps})
    otm = allq[
        ((allq.option_type == "call") & (allq.strike > F_row))
        | ((allq.option_type == "put") & (allq.strike <= F_row))
    ]
    otm = otm.dropna(subset=["iv"])
    cov = {}
    for e in exps:
        s = otm[otm.expiry == e]
        cov[e] = {
            "n_otm_quotes": int(len(s)),
            "k_min": float(s.k.min()),
            "k_max": float(s.k.max()),
            "median_rel_spread": float(s.spread_rel.median()),
            "share_rel_spread_gt_20pct": float((s.spread_rel > 0.20).mean()),
        }
    out["otm_coverage_0603"] = cov

    # ---------------------------------- 4. Yahoo IV column vs recomputed (06-03)
    m = allq.dropna(subset=["iv"])
    m = m[(m.impliedVolatility > 0.01)]
    diff = (m.impliedVolatility - m.iv) * 100
    out["yahoo_iv_minus_black76_volpts_0603"] = {
        "n": int(len(m)),
        "median": float(diff.median()),
        "p05": float(diff.quantile(0.05)),
        "p95": float(diff.quantile(0.95)),
    }

    # --------------------------- 5. stale lastPrice: 06-03 mid vs 06-04 last
    d4 = pd.read_csv(data_dir / "spy_chains_20260604.csv")
    j = d3.merge(
        d4[["contractSymbol", "lastPrice", "lastTradeDate"]], on="contractSymbol", suffixes=("_03", "_04")
    )
    out["rows_identical_lastPrice_03_vs_04"] = float((j.lastPrice_03 == j.lastPrice_04).mean())
    out["underlying_spot_0604_file"] = float(d4.underlying_spot.iloc[0])

    # ------------------- 6. time-to-expiry convention of the published run
    # Published run: T = (expiry - 2026-06-04)/365 in whole days, but the
    # 06-04 file holds 06-03 closing trades (max lastTradeDate). Effective
    # data time = that max timestamp; expiry at 16:00 New York.
    t_data_04 = pd.to_datetime(d4.lastTradeDate, utc=True).max()
    tconv = {}
    for e in exps:
        T_pub = (pd.Timestamp(e) - pd.Timestamp("2026-06-04")).days / 365.0
        T_true = year_fraction(e, t_data_04)
        tconv[e] = {
            "T_pub_days": T_pub * 365,
            "T_true_days": T_true * 365,
            "iv_bias_rel_pct": float((np.sqrt(T_true / T_pub) - 1) * 100),
        }
    out["time_convention_published_run"] = tconv

    # ---------------- 7. SSVI power-law: butterfly condition as theta -> 0
    # Published global parameters (README; replay reproduces them).
    rho, eta, gamma = -0.494, 0.832, 0.582
    th = np.logspace(-7, -1, 200001)
    phi = eta / (th**gamma * (1 + th) ** (1 - gamma))
    c1 = th * phi * (1 + abs(rho))
    c2 = th * phi**2 * (1 + abs(rho))
    bad2 = th[c2 > 4]
    theta_star = float(bad2.max()) if bad2.size else None
    atm_vol_short = 0.129  # published SSVI ATM vol at 28d (README)
    out["ssvi_butterfly_small_theta"] = {
        "gamma_gt_half": gamma > 0.5,
        "theta_star_cond2_fails_below": theta_star,
        "maturity_days_at_theta_star_if_atm_vol_12.9pct": (theta_star / atm_vol_short**2 * 365)
        if theta_star
        else None,
        "cond1_max_on_grid": float(c1.max()),
        "theta_min_calibrated": 0.00127,
    }

    # ------------ 8. fixed-moneyness RR/BF expressed in standard deviations
    atm_pub = {"28": 0.138, "57": 0.150, "88": 0.154, "179": 0.167}  # README SVI ATM
    out["k_0.10_in_std_units"] = {d: float(0.10 / (v * np.sqrt(int(d) / 365))) for d, v in atm_pub.items()}

    (out_dir / "diagnostics.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
