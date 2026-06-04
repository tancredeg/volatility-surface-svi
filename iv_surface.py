"""
iv_surface.py
=============
Implied Volatility Surface — SVI Calibration Pipeline

End-to-end pipeline: SPY option chain download → IV extraction →
SVI calibration → arbitrage verification → surface construction.

Run:
    python iv_surface.py

Author:  <your name>
License: MIT
"""

# =============================================================================
# SECTION 0 — Configuration
# =============================================================================
# Imports, constants, and global configuration.
#
# Design notes:
#   - Single RNG: np.random.default_rng(42). No np.random.seed() anywhere.
#   - RuntimeWarnings are kept VISIBLE. A sqrt of a negative or a division
#     by zero inside the BS/vega/IV loop signals a real pipeline failure
#     that must not be silently swallowed in a portfolio project.
#   - DATA_SOURCE is resolved at runtime in Section 1 and printed explicitly.
# =============================================================================

import sys
import warnings
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — savefig() only
import matplotlib.pyplot as plt

# RuntimeWarnings from numpy signal real numerical failures — keep visible.
warnings.filterwarnings("default", category=RuntimeWarning)

# --- Logging setup -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# --- Paths -------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
FIG_DIR    = BASE_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# --- Single RNG (no legacy np.random.seed anywhere) -------------------------
rng = np.random.default_rng(42)

# --- Market / rate conventions -----------------------------------------------
# Risk-free rate: approximate 3-month T-bill rate (update if needed).
# Continuous dividend yield for SPY: approximate annual yield.
# Both are treated as flat curves — a known simplification.
# A desk implementation would use OIS discounting and a dividend term structure.
RISK_FREE_RATE     = 0.053   # approximate 3M T-bill rate
DIVIDEND_YIELD     = 0.013   # approximate SPY annual dividend yield

# Target maturities (calendar days). yfinance uses listed expiries;
# we select the four closest to these targets.
TARGET_MATURITIES_DAYS = [30, 60, 90, 180]

# DATA_SOURCE is set in Section 1 at runtime.
DATA_SOURCE: str = "unknown"


# =============================================================================
# SECTION 1 — Data Acquisition
# =============================================================================
# Financial motivation:
#   SPY options are the most liquid equity options globally. The chain
#   provides a cross-section of market-implied risk-neutral densities at
#   multiple horizons. We extract four slices to build a term structure.
#
# Conventions:
#   - All maturities in years unless suffixed _days.
#   - F = S * exp((r-q)*T)  (cost-of-carry forward)
#   - Log-moneyness m = ln(K/F), centred on the forward.
#
# Known limitations:
#   - Flat r and q curves (no OIS, no dividend term structure).
#   - yfinance implied_volatility column is used ONLY for schema validation,
#     never as the IV estimate — we recompute from mid prices.
# =============================================================================

def validate_option_chain(df: pd.DataFrame) -> None:
    """Verify that the DataFrame has the expected yfinance schema.

    Args:
        df: Raw option chain DataFrame from yfinance.

    Raises:
        KeyError: If any required column is absent, with a diagnostic message
                  listing both expected and received columns.

    Notes:
        This guard runs BEFORE any processing. If yfinance changes its API
        schema, the error is surfaced immediately rather than propagating
        as a silent NaN or KeyError later in the pipeline.
    """
    required = ["bid", "ask", "strike", "lastPrice", "openInterest",
                "impliedVolatility"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"yfinance API may have changed schema. "
            f"Expected columns: {required}. "
            f"Got: {list(df.columns)}. "
            f"Missing: {missing}."
        )


def _nearest_expiry(expiries: list, target_days: int,
                    reference_date: datetime.date) -> str | None:
    """Return the expiry string closest to target_days from reference_date."""
    best_exp, best_diff = None, float("inf")
    for exp_str in expiries:
        try:
            exp_date = datetime.date.fromisoformat(exp_str)
        except ValueError:
            continue
        diff = abs((exp_date - reference_date).days - target_days)
        if diff < best_diff:
            best_diff = diff
            best_exp = exp_str
    return best_exp


def download_spy_chains(
    reference_date: datetime.date,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Download SPY option chains for four target maturities via yfinance.

    Args:
        reference_date: Pricing date (today).

    Returns:
        Tuple of (chains_by_expiry, data_source_label).
        chains_by_expiry maps expiry string → combined call+put DataFrame.
        data_source_label is 'live' or 'cached'.

    Raises:
        RuntimeError: If both live download and cache are unavailable.

    Notes:
        Saves raw chains to data/spy_chains_{date}.csv on successful download.
        Loads from cache if download fails and a cache file exists.
        NEVER falls back to synthetic data silently.
    """
    import yfinance as yf

    cache_path = DATA_DIR / f"spy_chains_{reference_date.isoformat()}.csv"

    # --- Attempt live download -----------------------------------------------
    try:
        ticker = yf.Ticker("SPY")
        expiries = ticker.options
        if not expiries:
            raise ValueError("yfinance returned empty expiry list for SPY.")

        selected_expiries = []
        for target in TARGET_MATURITIES_DAYS:
            exp = _nearest_expiry(list(expiries), target, reference_date)
            if exp and exp not in selected_expiries:
                selected_expiries.append(exp)

        log.info("Selected expiries: %s", selected_expiries)

        frames = []
        for exp in selected_expiries:
            # yfinance >= 0.2.x returns a named tuple Options(calls, puts,
            # underlying) — THREE fields. Use attribute access, not 2-tuple
            # unpacking, to stay compatible across versions.
            oc = ticker.option_chain(exp)
            calls = oc.calls.copy()
            puts  = oc.puts.copy()
            calls["option_type"] = "call"
            puts["option_type"]  = "put"
            calls["expiry"] = exp
            puts["expiry"]  = exp
            frames.extend([calls, puts])

        combined = pd.concat(frames, ignore_index=True)
        validate_option_chain(combined)

        # Persist the underlying spot so re-running offline still works.
        # Without this, iv_surface.py's own live save would overwrite a
        # fetch_data.py CSV and strip the spot column.
        try:
            hist = ticker.history(period="1d")
            if not hist.empty:
                combined["underlying_spot"] = float(hist["Close"].iloc[-1])
        except Exception:
            pass   # not fatal; resolve_spot will fall back to live next run

        # Cache to disk
        combined.to_csv(cache_path, index=False)
        log.info("Saved live chains to %s", cache_path)

        # Reconstruct dict
        chains = {
            exp: combined[combined["expiry"] == exp].copy()
            for exp in selected_expiries
        }
        return chains, "live"

    except Exception as live_err:
        log.warning("Live download failed: %s", live_err)

    # --- Attempt cache load --------------------------------------------------
    # Look for any cache file matching today's date first, then most recent.
    cache_candidates = sorted(DATA_DIR.glob("spy_chains_*.csv"), reverse=True)
    if cache_candidates:
        cache_file = cache_candidates[0]
        log.info("Loading cached data from %s", cache_file)
        combined = pd.read_csv(cache_file)
        validate_option_chain(combined)
        # Refuse synthetic data on the real/cached path. A cache file whose
        # expiries are labelled 'synthetic_*' must never be treated as market data.
        if combined["expiry"].astype(str).str.startswith("synthetic").any():
            raise RuntimeError(
                f"Cache file {cache_file} contains synthetic expiry labels. "
                "Refusing to load synthetic data on the real-data path. "
                "Run fetch_data.py to obtain genuine market data."
            )
        expiries_in_cache = combined["expiry"].unique().tolist()
        chains = {
            exp: combined[combined["expiry"] == exp].copy()
            for exp in expiries_in_cache
        }
        return chains, "cached"

    # --- Both paths failed ---------------------------------------------------
    raise RuntimeError(
        "Both live download and cache are unavailable.\n"
        "To resolve:\n"
        "  1. Ensure internet access and re-run (live download).\n"
        "  2. Place a previously saved spy_chains_YYYY-MM-DD.csv "
        "     file in the data/ directory (cache fallback).\n"
        "  3. To test with synthetic data, call run_synthetic_pipeline() "
        "     explicitly — synthetic data is NEVER the default path.\n"
    )


# =============================================================================
# SECTION 2 — Data Cleaning and IV Extraction
# =============================================================================
# Financial motivation:
#   Raw option chains contain illiquid, stale, and misquoted strikes.
#   Each filter removes a specific class of pathological data. The order
#   matters: we must compute mid before applying the spread filter, and
#   we must have a valid mid before checking intrinsic value.
#
# Mathematical framework:
#   Black-Scholes price for European call/put with continuous dividend q:
#     d1 = [ln(S/K) + (r-q+σ²/2)T] / (σ√T)
#     d2 = d1 - σ√T
#     Call = S·e^{-qT}·N(d1) - K·e^{-rT}·N(d2)
#     Put  = K·e^{-rT}·N(-d2) - S·e^{-qT}·N(-d1)
#     Vega = S·e^{-qT}·φ(d1)·√T         (same for call and put)
#
# Known limitations:
#   - European Black-Scholes vega used for American SPY options.
#     SPY pays a discrete dividend. Early exercise premium is small
#     for liquid near-money strikes but non-negligible deep ITM.
# =============================================================================

def _bs_price(S: float, K: float, T: float, r: float, q: float,
              sigma: float, option_type: str) -> float:
    """Black-Scholes price for a European option with continuous dividend.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Continuously compounded risk-free rate.
        q: Continuously compounded dividend yield.
        sigma: Implied volatility (annualised).
        option_type: 'call' or 'put'.

    Returns:
        Theoretical option price. Returns 0.0 on numerical failure.

    Notes:
        Uses continuous dividend yield (Merton 1973 extension).
        Valid for European options; SPY options are American but
        early exercise premium is negligible near-the-money.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def _bs_vega(S: float, K: float, T: float, r: float, q: float,
             sigma: float) -> float:
    """Black-Scholes vega (∂Price/∂σ), identical for calls and puts.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.

    Returns:
        Vega. Returns 0.0 on numerical failure.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T


def implied_vol_nr(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Extract implied volatility via Newton-Raphson with Brent fallback.

    Algorithm:
        1. Newton-Raphson: σ_{n+1} = σ_n - (BS(σ_n) - price) / Vega(σ_n)
           Switch to Brent if Vega < 1e-10 (avoids divergence deep OTM).
        2. Brent's method on [1e-4, 10.0] as a guaranteed-convergent fallback.
           If the price is not bracketed in this interval, returns np.nan
           with an explicit log warning (not a silent NaN).

    Args:
        price: Observed option mid-price.
        S:     Spot.
        K:     Strike.
        T:     Time to expiry in years.
        r:     Risk-free rate.
        q:     Dividend yield.
        option_type: 'call' or 'put'.
        tol:   Convergence tolerance on σ.
        max_iter: Maximum NR iterations before switching to Brent.

    Returns:
        Implied volatility, or np.nan if no solution exists.

    Notes:
        NR diverges when Vega ≈ 0 (deep OTM, very short T).
        Brent is slower (O(log(1/tol)) function evals) but guaranteed
        to converge within [1e-4, 10.0] provided the price is bracketed.
        Reference: Jaeckel (2006) "By Implication" for a faster approach
        (not implemented here for clarity).
    """
    if T <= 0 or price <= 0:
        return np.nan

    # --- Newton-Raphson phase ------------------------------------------------
    sigma = 0.20   # initial guess: 20% vol
    for _ in range(max_iter):
        vega = _bs_vega(S, K, T, r, q, sigma)
        if vega < 1e-10:
            # NR diverges when Vega ≈ 0 (deep OTM). Brent is slower but
            # guaranteed convergent.
            break
        bs = _bs_price(S, K, T, r, q, sigma, option_type)
        diff = bs - price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        sigma = max(1e-6, sigma)   # keep positive

    # --- Brent fallback ------------------------------------------------------
    lo, hi = 1e-4, 10.0
    f_lo = _bs_price(S, K, T, r, q, lo, option_type) - price
    f_hi = _bs_price(S, K, T, r, q, hi, option_type) - price

    if f_lo * f_hi > 0:
        # No sign change: price is not bracketed. Log and return NaN.
        log.warning(
            "No IV solution in [1e-4, 10.0] for K=%.2f, T=%.4f, "
            "price=%.4f. Likely a data error — skipping.",
            K, T, price
        )
        return np.nan

    try:
        iv = brentq(
            lambda sig: _bs_price(S, K, T, r, q, sig, option_type) - price,
            lo, hi, xtol=tol, maxiter=200
        )
        return iv
    except Exception as e:
        log.warning("Brent failed for K=%.2f, T=%.4f: %s", K, T, e)
        return np.nan


def clean_option_chain(
    df: pd.DataFrame,
    S: float,
    r: float,
    q: float,
    T: float,
    min_price_abs: float = 0.05,
    max_spread_pct: float = 0.20,
    moneyness_range: float = 0.35,
) -> pd.DataFrame:
    """Apply a sequential filtering pipeline to a raw option chain slice.

    Filters are applied in a specific order (see inline comments).
    Each step logs the number of rows dropped and the reason.

    Args:
        df:             Raw option chain for a single expiry and type.
        S:              Spot price.
        r:              Risk-free rate (continuous).
        q:              Dividend yield (continuous).
        T:              Time to expiry in years.
        min_price_abs:  Minimum absolute mid-price (USD). Default 0.05.
        max_spread_pct: Maximum (ask-bid)/mid ratio. Default 0.20 (20%).
        moneyness_range: Maximum |ln(K/F)| retained. Default 0.35.

    Returns:
        Filtered DataFrame with a 'mid' column added.

    Raises:
        ValueError: If fewer than 5 rows survive all filters.

    Notes:
        Filter 3 (absolute price floor) is critical for deep OTM options.
        A call at K=800 with S=540 may have mid=0.01. The resulting IV
        can exceed 300% and still pass a naive IV<2.0 filter. The absolute
        price floor eliminates this pathology before IV extraction.
    """
    n0 = len(df)
    df = df.copy()

    # ---- Diagnostic: detect "all zero bid" (market closed / pre-/post-mkt) -
    # Yahoo populates bid/ask only during US market hours. Outside that
    # window the chain comes back with bid=0 / ask=0 everywhere but lastPrice
    # still carries the last trade. We log this loudly so the user knows the
    # cause is timing, not code, and we fall back to lastPrice.
    n_zero_quote = int(((df["bid"] == 0) & (df["ask"] == 0)).sum())
    if n0 > 0 and n_zero_quote / n0 > 0.7:
        log.warning(
            "  %d/%d rows (%.0f%%) have bid=ask=0. Quote feed is stale — "
            "most likely the US market is closed or pre-/post-market. "
            "Falling back to lastPrice where available; results will be "
            "based on the last trade prices, not live mid quotes. "
            "For best quality re-run between 15:30 and 22:00 Paris time.",
            n_zero_quote, n0, 100.0 * n_zero_quote / n0,
        )

    # ---- Build a usable price column -------------------------------------
    # mid where both bid and ask are positive; lastPrice otherwise.
    has_quote = (df["bid"] > 0) & (df["ask"] > 0) & (df["bid"] <= df["ask"])
    df["mid"] = np.where(
        has_quote,
        (df["bid"] + df["ask"]) / 2.0,
        df["lastPrice"].astype(float),
    )
    df["price_source"] = np.where(has_quote, "mid", "last")

    # Filter 1 — bid > ask guard (only meaningful when both > 0)
    mask = (df["bid"] > 0) & (df["ask"] > 0) & (df["bid"] > df["ask"])
    n_drop = int(mask.sum())
    if n_drop > 0:
        log.info("  Dropped %d rows: bid > ask (data pathology)", n_drop)
    df = df[~mask]

    # Filter 2 — no usable price at all (no live quote AND no last trade)
    mask = (df["mid"].isna()) | (df["mid"] <= 0)
    n_drop = int(mask.sum())
    if n_drop > 0:
        log.info("  Dropped %d rows: no usable price (no quote, no lastPrice)", n_drop)
    df = df[~mask]

    # Filter 3 — Minimum absolute price (eliminates deep-OTM noise)
    mask = df["mid"] < min_price_abs
    n_drop = int(mask.sum())
    if n_drop > 0:
        log.info(
            "  Dropped %d rows: mid < %.2f (deep OTM noise)", n_drop, min_price_abs
        )
    df = df[~mask]

    # Filter 4 — Spread filter, ONLY where we have a real two-sided quote.
    # Rows that fell back to lastPrice have no meaningful spread.
    has_mid = (df["price_source"] == "mid")
    mid_safe = df["mid"].where(has_mid, other=1.0)   # avoid div-by-zero on lastPrice rows
    spread_ratio = np.where(
        has_mid, (df["ask"] - df["bid"]) / mid_safe, 0.0
    )
    mask = spread_ratio > max_spread_pct
    n_drop = int(mask.sum())
    if n_drop > 0:
        log.info(
            "  Dropped %d rows: spread > %.0f%% of mid", n_drop, max_spread_pct * 100
        )
    df = df[~mask]

    # Filter 5 — Intrinsic value (stale quotes)
    F_disc = S * np.exp(-q * T)   # PV of forward delivery
    K_disc = df["strike"] * np.exp(-r * T)
    intrinsic_call = np.maximum(F_disc - K_disc, 0.0)
    intrinsic_put  = np.maximum(K_disc - F_disc, 0.0)

    otype = df["option_type"].str.lower()
    intrinsic = np.where(otype == "call", intrinsic_call, intrinsic_put)
    mask = df["mid"].values < intrinsic - 0.01
    n_drop = int(mask.sum())
    if n_drop > 0:
        log.info("  Dropped %d rows: mid below intrinsic (stale quote)", n_drop)
    df = df[~mask]

    # Filter 6 — Moneyness range
    F = S * np.exp((r - q) * T)
    df["log_moneyness"] = np.log(df["strike"] / F)
    mask = df["log_moneyness"].abs() > moneyness_range
    n_drop = int(mask.sum())
    if n_drop > 0:
        log.info(
            "  Dropped %d rows: |moneyness| > %.2f", n_drop, moneyness_range
        )
    df = df[~mask]

    # Report fallback usage if any rows came from lastPrice
    n_last = int((df["price_source"] == "last").sum())
    if n_last > 0:
        log.info(
            "  %d/%d retained rows use lastPrice (stale quote risk)",
            n_last, len(df)
        )

    n_final = len(df)
    if n_final < 5:
        raise ValueError(
            f"Insufficient clean data points ({n_final} remaining after filtering). "
            f"Cannot calibrate SVI for T={T:.4f}. Skipping this maturity."
        )

    log.info(
        "  T=%.4fy: %d raw → %d clean points retained", T, n0, n_final
    )
    return df.reset_index(drop=True)


def extract_iv_for_chain(
    df_clean: pd.DataFrame,
    S: float,
    r: float,
    q: float,
    T: float,
) -> pd.DataFrame:
    """Extract implied vols for all rows in a cleaned option chain slice.

    Args:
        df_clean: Cleaned DataFrame (output of clean_option_chain).
        S:        Spot price.
        r:        Risk-free rate.
        q:        Dividend yield.
        T:        Time to expiry in years.

    Returns:
        DataFrame with 'iv' column added; rows where IV extraction fails
        (np.nan) or iv is outside [0.01, 5.0] are dropped.

    Notes:
        IVs above 200% are logged as warnings but retained — they may
        indicate genuine stress pricing rather than data errors.
    """
    ivs = []
    for _, row in df_clean.iterrows():
        iv = implied_vol_nr(
            price=row["mid"],
            S=S,
            K=row["strike"],
            T=T,
            r=r,
            q=q,
            option_type=row["option_type"].lower(),
        )
        ivs.append(iv)

    df_clean = df_clean.copy()
    df_clean["iv"] = ivs

    # Drop failed extractions
    n_before = len(df_clean)
    df_clean = df_clean.dropna(subset=["iv"])
    n_failed = n_before - len(df_clean)
    if n_failed > 0:
        log.info("  Dropped %d rows: IV extraction returned NaN", n_failed)

    # Validity filter: 0.01 < iv < 5.0
    df_clean = df_clean[df_clean["iv"] > 0.01]
    df_clean = df_clean[df_clean["iv"] < 5.0]

    # Warn on IVs above 200% (retained but flagged)
    high_iv = df_clean[df_clean["iv"] > 2.0]
    for _, row in high_iv.iterrows():
        log.warning(
            "Warning: IV=%.2f > 200%% for K=%.2f, T=%.4f. "
            "Retained but flagged — may indicate data error.",
            row["iv"], row["strike"], T
        )

    return df_clean.reset_index(drop=True)


def verify_call_put_parity(
    df_calls: pd.DataFrame,
    df_puts: pd.DataFrame,
    T: float,
    tolerance_bp: float = 50.0,
) -> None:
    """Test call-put parity consistency at strikes where both are available.

    Parity condition: IV_call(K) ≈ IV_put(K)
    (Under continuous dividends and European exercise, identical strikes
    yield identical implied vols — violations signal pricing inconsistency.)

    Args:
        df_calls: Cleaned+IV-extracted call chain.
        df_puts:  Cleaned+IV-extracted put chain.
        T:        Time to expiry in years.
        tolerance_bp: Maximum acceptable IV difference in basis points.

    Notes:
        SPY exchanges typically quote calls OTM on the right and puts OTM
        on the left. After moneyness filtering, there may be no overlapping
        strikes. This is expected and reported explicitly — NOT treated as
        a test failure.
        DO NOT substitute a smile chart as a parity check proxy.
    """
    T_days = round(T * 365)
    strikes_calls = set(df_calls["strike"].round(2))
    strikes_puts  = set(df_puts["strike"].round(2))
    overlap = sorted(strikes_calls & strikes_puts)

    if not overlap:
        log.info(
            "No overlapping call/put strikes available for parity test "
            "at T=%dd. Parity test skipped — this is expected for exchanges "
            "quoting OTM options only.",
            T_days
        )
        return

    violations = 0
    max_violation_bp = 0.0
    n_tested = len(overlap)

    for K in overlap:
        iv_c = df_calls.loc[df_calls["strike"].round(2) == round(K, 2), "iv"].mean()
        iv_p = df_puts.loc[df_puts["strike"].round(2) == round(K, 2), "iv"].mean()
        diff_bp = abs(iv_c - iv_p) * 10_000
        if diff_bp > tolerance_bp:
            violations += 1
            max_violation_bp = max(max_violation_bp, diff_bp)

    log.info(
        "T=%dd call-put parity: %d pairs tested, %d violations, "
        "max violation %.1f bp (threshold: %.0f bp).",
        T_days, n_tested, violations, max_violation_bp, tolerance_bp
    )


# =============================================================================
# SECTION 3 — Descriptive Analysis (raw data, no model)
# =============================================================================
# Purpose: inspect the raw smile before fitting. This section reveals
# data quality, identifies skew and term structure patterns, and checks
# for calendar no-arbitrage violations in the raw observed data.
#
# Calendar no-arbitrage (raw check):
#   For a fixed moneyness m, total variance w(m,T) = σ²(m,T)·T must be
#   non-decreasing in T. Violations in raw data are expected due to noise
#   and sparse strikes — they are reported with count and magnitude, not
#   attributed vaguely to "noise".
# =============================================================================

def plot_raw_smiles(
    chains_by_maturity: dict[float, pd.DataFrame],
    fig_path: Path,
) -> None:
    """Plot raw implied vol smiles per maturity (no model fit).

    Args:
        chains_by_maturity: Map T (years) → cleaned+IV DataFrame.
        fig_path:           Output file path.
    """
    maturities = sorted(chains_by_maturity.keys())
    n = len(maturities)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, T in zip(axes, maturities):
        df = chains_by_maturity[T]
        calls = df[df["option_type"] == "call"]
        puts  = df[df["option_type"] == "put"]
        T_days = round(T * 365)

        ax.scatter(calls["log_moneyness"], calls["iv"] * 100,
                   s=20, alpha=0.7, color="steelblue", label="Calls")
        ax.scatter(puts["log_moneyness"], puts["iv"] * 100,
                   s=20, alpha=0.7, color="tomato", label="Puts")
        ax.set_title(f"T={T_days}d  (raw IV)")
        ax.set_xlabel("Log-moneyness ln(K/F)")
        ax.set_ylabel("Implied Vol (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Raw Implied Volatility Smiles — SPY", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", fig_path)


def check_raw_calendar_arbitrage(
    chains_by_maturity: dict[float, pd.DataFrame],
) -> None:
    """Check calendar no-arbitrage on raw observed total variances.

    For each pair of adjacent maturities (T1 < T2) and the common
    moneyness grid, verify w_obs(m, T1) <= w_obs(m, T2).

    Args:
        chains_by_maturity: Map T (years) → cleaned+IV DataFrame.

    Notes:
        Violations in raw data are EXPECTED due to noise, bid-ask spread,
        and sparse strikes at different maturities. They are reported with
        count and magnitude — NOT attributed vaguely to "noise".
        The calibrated SVI surface may resolve these violations.
    """
    log.info("--- Raw calendar no-arbitrage check ---")
    maturities = sorted(chains_by_maturity.keys())
    m_grid = np.linspace(-0.30, 0.30, 13)

    # Interpolate observed ATM total variance on a common m-grid
    # using nearest-neighbour (not smoothed) to avoid introducing artifacts.
    w_by_T: dict[float, np.ndarray] = {}
    for T in maturities:
        df = chains_by_maturity[T]
        if len(df) < 3:
            continue
        # Build total variance from observed IV
        w_obs = df["iv"] ** 2 * T
        # Sort by moneyness for interpolation
        idx = df["log_moneyness"].argsort()
        m_sorted  = df["log_moneyness"].values[idx]
        w_sorted  = w_obs.values[idx]
        # Linear interpolation on the grid (extrapolation set to NaN)
        w_interp = np.interp(m_grid, m_sorted, w_sorted,
                             left=np.nan, right=np.nan)
        w_by_T[T] = w_interp

    total_violations = 0
    for i in range(len(maturities) - 1):
        T1, T2 = maturities[i], maturities[i + 1]
        if T1 not in w_by_T or T2 not in w_by_T:
            continue
        w1, w2 = w_by_T[T1], w_by_T[T2]
        valid  = ~(np.isnan(w1) | np.isnan(w2))
        viol   = valid & (w1 > w2 + 1e-6)
        n_viol = viol.sum()
        max_mag = (w1[viol] - w2[viol]).max() if n_viol > 0 else 0.0
        total_violations += n_viol
        log.info(
            "  Raw calendar T=%.0fd→%.0fd: %d violations "
            "(max magnitude %.4f in total variance).",
            T1 * 365, T2 * 365, n_viol, max_mag
        )

    if total_violations == 0:
        log.info("  Raw calendar check: no violations detected.")
    else:
        log.info(
            "  Raw calendar check: %d total violations. "
            "Expected in noisy data — SVI calibration may resolve some.",
            total_violations
        )


# =============================================================================
# SECTION 4 — SVI Calibration
# =============================================================================
# Mathematical framework:
#   Gatheral (2004) SVI parametrization of total variance:
#
#     w(m) = a + b * [ ρ(m-μ) + √((m-μ)² + ξ²) ]
#
#   where m = ln(K/F), w = σ²·T (total variance), and:
#     a  ∈ ℝ          overall variance level (can be slightly negative
#                     at short maturities due to market microstructure)
#     b  > 0          slope/curvature
#     ρ  ∈ (-1,1)     skew (typically negative for equities)
#     μ  ∈ ℝ          location of minimum
#     ξ  > 0          smoothness / ATM curvature
#
#   Butterfly no-arbitrage (Lee 2004, necessary condition):
#     b(1 + |ρ|) < 4/T
#
#   Butterfly no-arbitrage (Roper 2010, sufficient condition):
#     g(m) ≥ 0 for all m, where:
#     g(m) = (1 - m·ρ/ξ_eff)² + (ρ²-1)·b²/ξ_eff²
#     ξ_eff = √((m-μ)² + ξ²)
#
#   The necessary condition can be satisfied while the sufficient
#   condition is violated for some m. Both are checked.
#
# Optimizer: SLSQP with an explicit inequality constraint enforces
#   butterfly_constraint as a hard condition. L-BFGS-B with a penalty
#   term can converge to a point that marginally violates the constraint
#   if the penalty weight is poorly scaled relative to the objective.
#
# SINGLE DEFINITION — do not redefine svi_total_variance anywhere else.
# =============================================================================

def svi_total_variance(
    m: np.ndarray,
    a: float,
    b: float,
    rho: float,
    mu: float,
    xi: float,
) -> np.ndarray:
    """Compute SVI total variance w(m) = a + b[ρ(m-μ) + √((m-μ)²+ξ²)].

    SINGLE DEFINITION — do not redefine this function elsewhere in the module.

    Args:
        m:   Log-moneyness array ln(K/F).
        a:   Vertical translation. May be slightly negative (see Notes).
        b:   Slope/curvature parameter. Must be positive.
        rho: Correlation/skew parameter. In (-1, 1).
        mu:  Shift parameter (location of variance minimum).
        xi:  Smoothness parameter. Must be positive.

    Returns:
        Total variance array w(m) with same shape as m.

    Notes:
        a can be slightly negative at short maturities (< 30d) due to
        market microstructure effects and bid-ask spreads. The calibration
        bounds allow a ∈ (-0.5, 2.0) to accommodate this.
        Reference: Gatheral, J. (2004). "A parsimonious arbitrage-free
        implied volatility parametrization with application to the valuation
        of volatility derivatives."
    """
    dm = m - mu
    return a + b * (rho * dm + np.sqrt(dm ** 2 + xi ** 2))


def butterfly_constraint(b: float, rho: float, xi: float, T: float) -> float:
    """Compute Lee (2004) butterfly no-arbitrage constraint value.

    Returns 4/T - b*(1+|ρ|). Must be > 0 for no-arbitrage.

    Args:
        b:   SVI slope parameter.
        rho: SVI skew parameter.
        xi:  SVI smoothness parameter (not used in Lee condition but
             included for API consistency with Roper check).
        T:   Time to expiry in years.

    Returns:
        Constraint value. Positive means no-arbitrage is satisfied
        (Lee necessary condition).

    Notes:
        This is a NECESSARY but not sufficient condition for butterfly
        no-arbitrage. The sufficient condition requires g(m) ≥ 0 for all m,
        where g(m) = (1 - m·ρ/ξ_eff)² + (ρ²-1)·b²/ξ_eff²
        and ξ_eff = √((m-μ)²+ξ²). See Roper (2010) "Arbitrage-Free Implied
        Volatility Surfaces" for the full characterization.
    """
    return 4.0 / T - b * (1.0 + abs(rho))


def svi_total_variance_derivatives(
    m: np.ndarray,
    b: float,
    rho: float,
    mu: float,
    xi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic first and second derivatives of SVI total variance w(m).

    w'(m)  = b * [ ρ + (m-μ)/√((m-μ)²+ξ²) ]
    w''(m) = b * ξ² / ((m-μ)²+ξ²)^{3/2}

    Args:
        m:   Log-moneyness array.
        b:   SVI b parameter.
        rho: SVI rho parameter.
        mu:  SVI mu parameter.
        xi:  SVI xi parameter.

    Returns:
        Tuple (w', w'') of first and second derivatives w.r.t. m.
    """
    dm = m - mu
    root = np.sqrt(dm ** 2 + xi ** 2)
    w_prime = b * (rho + dm / root)
    w_double_prime = b * xi ** 2 / root ** 3
    return w_prime, w_double_prime


def durrleman_g(
    m: np.ndarray,
    a: float,
    b: float,
    rho: float,
    mu: float,
    xi: float,
) -> np.ndarray:
    """Durrleman butterfly no-arbitrage function g(m).

    A slice is free of butterfly arbitrage (equivalently, the implied
    risk-neutral density is non-negative) if and only if g(m) ≥ 0 for all m:

        g(m) = (1 - m·w'(m)/(2·w(m)))²
               - (w'(m)²/4)·(1/w(m) + 1/4)
               + w''(m)/2

    Args:
        m:   Log-moneyness array.
        a:   SVI a parameter (needed because g depends on w, not just w').
        b:   SVI b parameter.
        rho: SVI rho parameter.
        mu:  SVI mu parameter.
        xi:  SVI xi parameter.

    Returns:
        g(m) array.

    Notes:
        This is the SUFFICIENT-AND-NECESSARY condition for the absence of
        butterfly arbitrage on a single SVI slice. It supersedes the
        formula `(1 - m·ρ/ξ_eff)² + (ρ²-1)·b²/ξ_eff²` that appeared in an
        earlier version: that expression is NOT the Durrleman g-function and
        produced false butterfly-arbitrage alarms (it reported g<0 on
        arbitrage-free surfaces). The Lee (2004) condition b(1+|ρ|) < 4/T
        remains a quick necessary screen.
        References: Durrleman (2010); Gatheral & Jacquier (2014),
        "Arbitrage-free SVI volatility surfaces", eq. (2.2)-(2.3).
    """
    dm = m - mu
    w = a + b * (rho * dm + np.sqrt(dm ** 2 + xi ** 2))
    w_prime, w_double_prime = svi_total_variance_derivatives(m, b, rho, mu, xi)
    term1 = (1.0 - m * w_prime / (2.0 * w)) ** 2
    term2 = (w_prime ** 2 / 4.0) * (1.0 / w + 0.25)
    term3 = w_double_prime / 2.0
    return term1 - term2 + term3


def calibrate_svi(
    m_arr: np.ndarray,
    w_arr: np.ndarray,
    T: float,
    n_starts: int = 8,
    seed: int = 42,
) -> dict:
    """Calibrate SVI parameters by minimising sum-of-squared errors on w.

    Objective: min_θ Σᵢ (w_SVI(mᵢ; θ) - w_obs_i)²

    Optimizer: SLSQP with the butterfly constraint as an explicit
    inequality (not L-BFGS-B with penalty), ensuring the constraint is
    enforced as a hard condition.

    Args:
        m_arr:     Log-moneyness array (calibration nodes).
        w_arr:     Observed total variance array (σ²·T).
        T:         Time to expiry in years (for butterfly constraint).
        n_starts:  Number of random restarts for the multi-start.
        seed:      Seed for the local RNG (reproducible multi-start).

    Returns:
        dict with keys:
            params (dict: a, b, rho, mu, xi),
            rmse (float, in vol points),
            butterfly_satisfied (bool),
            n_iter (int),
            converged (bool).

    Notes:
        Bounds: a ∈ (-0.5, 2.0), b ∈ (1e-4, 2.0), ρ ∈ (-0.999, 0.999),
                μ ∈ (-0.5, 0.5), ξ ∈ (1e-4, 2.0).
        a can be slightly negative at short maturities (see svi_total_variance
        docstring). The comment and code bounds are consistent — the prior
        discrepancy (comment said a∈(0,2), code had (-0.5,2.0)) is fixed here.
        Multi-start uses a local RNG from np.random.default_rng(seed) to
        avoid mutating the module-level rng state.
    """
    local_rng = np.random.default_rng(seed)

    # Bounds: (a, b, rho, mu, xi)
    # a ∈ (-0.5, 2.0): slightly negative permitted at short maturities
    # b ∈ (1e-4, 2.0): strictly positive
    # rho ∈ (-0.999, 0.999): open interval (±1 makes SVI degenerate)
    # mu ∈ (-0.5, 0.5): shift parameter
    # xi ∈ (1e-4, 2.0): strictly positive smoothness
    bounds = [
        (-0.5,   2.0),   # a — can be slightly negative (see docstring)
        (1e-4,   2.0),   # b — strictly positive
        (-0.999, 0.999), # rho
        (-0.5,   0.5),   # mu
        (1e-4,   2.0),   # xi — strictly positive
    ]

    def objective(params: np.ndarray) -> float:
        a, b, rho, mu, xi = params
        w_fit = svi_total_variance(m_arr, a, b, rho, mu, xi)
        return float(np.sum((w_fit - w_arr) ** 2))

    # Butterfly constraint as explicit inequality for SLSQP
    constraints = [
        {
            "type": "ineq",
            "fun": lambda p: butterfly_constraint(p[1], p[2], p[4], T),
        }
    ]

    best_result = None
    best_obj    = np.inf

    # Generate random starting points (reproducible)
    for _ in range(n_starts):
        x0 = np.array([
            local_rng.uniform(0.001, 0.10),   # a
            local_rng.uniform(0.05,  0.30),   # b
            local_rng.uniform(-0.90, -0.20),  # rho (negative for equities)
            local_rng.uniform(-0.05, 0.05),   # mu
            local_rng.uniform(0.05,  0.30),   # xi
        ])
        try:
            res = minimize(
                objective, x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if res.fun < best_obj:
                best_obj    = res.fun
                best_result = res
        except Exception as e:
            log.warning("SVI optimisation start failed: %s", e)

    if best_result is None:
        raise RuntimeError(f"All {n_starts} SVI calibration starts failed for T={T:.4f}.")

    a, b, rho, mu, xi = best_result.x
    bf_value = butterfly_constraint(b, rho, xi, T)
    butterfly_ok = bf_value > 0

    if not butterfly_ok:
        log.warning(
            "T=%.4fy: All starts violate butterfly constraint. "
            "Returning best result with butterfly_satisfied=False "
            "(constraint value=%.6f).",
            T, bf_value
        )

    # RMSE in vol points: sqrt(w) = sigma*sqrt(T), so vol = sqrt(w/T)
    w_fit = svi_total_variance(m_arr, a, b, rho, mu, xi)
    iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
    iv_obs = np.sqrt(np.maximum(w_arr, 0.0) / T)
    rmse = float(np.sqrt(np.mean((iv_fit - iv_obs) ** 2)))

    return {
        "params": {"a": a, "b": b, "rho": rho, "mu": mu, "xi": xi},
        "rmse": rmse,
        "butterfly_satisfied": butterfly_ok,
        "n_iter": best_result.nit,
        "converged": bool(best_result.success),
    }


# =============================================================================
# SECTION 5 — Arbitrage Verification on Calibrated Surface
# =============================================================================
# All checks run on the CALIBRATED SVI surface, not on the raw noisy data.
#
# Checks performed:
#   1. Butterfly (Lee necessary condition):  4/T - b(1+|ρ|) > 0
#   2. Positivity of w on calibrated surface over m ∈ [-0.40, +0.40]
#   3. Calendar no-arbitrage on calibrated surface: w(m, T1) < w(m, T2)
#   4. RMSE per maturity (quality of fit)
#   5. Durrleman butterfly condition g(m) >= 0 (necessary AND sufficient)
#
# Design note on w < 0 handling:
#   np.maximum(w, 0) would clip silently — this is NOT done here.
#   A warning is raised that is visible to the user. The w < 0 region
#   outside the calibration range [-0.35, +0.35] indicates SVI
#   extrapolation failure.
# =============================================================================

def verify_svi_surface(
    svi_results: dict[float, dict],
    maturities: list[float],
) -> int:
    """Run all five arbitrage checks on the calibrated SVI surface.

    Args:
        svi_results: Map T (years) → calibrate_svi() output dict.
        maturities:  List of maturities in years (sorted ascending).

    Returns:
        Total number of calendar no-arbitrage violations on the calibration
        range |m| <= 0.35. This single value is reused by the desk-view
        summary so the two never report different counts.

    Notes:
        w < 0 outside calibration range raises a visible warning.
        np.maximum(w, 0) is NOT applied — silent clipping masks a real
        failure mode of the SVI parametrisation.
    """
    log.info("=" * 60)
    log.info("SECTION 5 — Arbitrage Verification (Calibrated Surface)")
    log.info("=" * 60)

    m_dense = np.linspace(-0.40, 0.40, 500)

    # Table header
    log.info(
        "%-8s  %-22s  %-20s  %-10s  %-15s",
        "T(days)", "butterfly_necessary", "g_min(sufficient)", "RMSE", "calendar_arb"
    )
    log.info("-" * 80)

    calendar_results: dict[float, np.ndarray] = {}

    for T in maturities:
        if T not in svi_results:
            continue
        res    = svi_results[T]
        p      = res["params"]
        a, b, rho, mu, xi = p["a"], p["b"], p["rho"], p["mu"], p["xi"]
        T_days = round(T * 365)

        # Check 1 — Butterfly (Lee necessary condition)
        bf_val  = butterfly_constraint(b, rho, xi, T)
        bf_pass = bf_val > 0
        log.info(
            "T=%dd butterfly constraint (Lee): %.6f (%s)",
            T_days, bf_val, "PASS" if bf_pass else "FAIL"
        )

        # Check 2 — Positivity of w
        w_dense = svi_total_variance(m_dense, a, b, rho, mu, xi)
        neg_mask = w_dense < 0
        if neg_mask.any():
            m_neg = m_dense[neg_mask]
            log.warning(
                "T=%dd: w(m) < 0 detected at m ∈ [%.4f, %.4f]. "
                "SVI extrapolation failure — NOT silently clipped.",
                T_days, m_neg.min(), m_neg.max()
            )
        else:
            log.info("T=%dd: w(m) >= 0 over [-0.40, +0.40] (PASS)", T_days)

        # Store for calendar check
        calendar_results[T] = w_dense

        # Check 4 — RMSE
        rmse = res["rmse"]
        rmse_status = "PASS" if rmse < 0.005 else "WARNING: RMSE > 50bp"
        log.info(
            "T=%dd SVI RMSE: %.4f vol points (%s)", T_days, rmse, rmse_status
        )

        # Check 5 — Durrleman butterfly condition g(m) >= 0 (necessary AND
        # sufficient on a slice). Replaces the prior mis-specified formula.
        g_vals = durrleman_g(m_dense, a, b, rho, mu, xi)
        g_min  = float(g_vals.min())
        g_pass = g_min >= -1e-9
        log.info(
            "T=%dd Durrleman g(m) min = %.6f (%s). "
            "g>=0 everywhere ⇔ no butterfly arbitrage (necessary & sufficient).",
            T_days, g_min, "PASS" if g_pass else "FAIL"
        )

    # Check 3 — Calendar no-arbitrage on calibrated surface.
    # IMPORTANT: evaluated on the CALIBRATION range [-0.35, +0.35] only.
    # Beyond that range the SVI fit is extrapolation, and crossings there
    # are extrapolation artifacts rather than genuine surface arbitrage.
    # A single grid (m_cal) is used so that the count reported here matches
    # the count used in the desk-view summary — no second, inconsistent grid.
    log.info("--- Calendar no-arbitrage (calibrated surface, |m| <= 0.35) ---")
    m_cal = np.linspace(-0.35, 0.35, 500)
    cal_curves = {
        T: svi_total_variance(
            m_cal,
            svi_results[T]["params"]["a"], svi_results[T]["params"]["b"],
            svi_results[T]["params"]["rho"], svi_results[T]["params"]["mu"],
            svi_results[T]["params"]["xi"],
        )
        for T in maturities if T in svi_results
    }
    total_cal_violations = 0
    for i in range(len(maturities) - 1):
        T1, T2 = maturities[i], maturities[i + 1]
        if T1 not in cal_curves or T2 not in cal_curves:
            continue
        w1, w2  = cal_curves[T1], cal_curves[T2]
        viol    = w1 > w2 + 1e-6
        n_viol  = int(viol.sum())
        total_cal_violations += n_viol
        if n_viol > 0:
            m_viol   = m_cal[viol]
            max_mag  = (w1[viol] - w2[viol]).max()
            log.warning(
                "T=%.0fd→%.0fd calendar violation: %d points, "
                "max magnitude %.6f. m range: [%.4f, %.4f]. "
                "Calendar violations indicate linear parameter interpolation "
                "is insufficient between these maturities. "
                "SSVI (Gatheral & Jacquier 2014) would resolve this.",
                T1 * 365, T2 * 365, n_viol, max_mag,
                m_viol.min(), m_viol.max()
            )
        else:
            log.info(
                "T=%.0fd→%.0fd calendar: PASS (no violations on |m|<=0.35)",
                T1 * 365, T2 * 365
            )

    if total_cal_violations == 0:
        log.info("Calendar no-arbitrage: PASS (all pairs, |m|<=0.35)")
    else:
        log.info(
            "Calendar no-arbitrage: %d total violations on calibrated surface "
            "(|m|<=0.35).",
            total_cal_violations
        )

    return total_cal_violations


# =============================================================================
# SECTION 6 — Surface Construction and Visualization
# =============================================================================
# The calibrated SVI parameters are used to:
#   (a) Construct a continuous surface by linear interpolation in T.
#   (b) Produce four diagnostic figures for the portfolio.
#
# Linear parameter interpolation:
#   For T* between calibrated maturities T1 < T2:
#   p*(T*) = p1 + (T*-T1)/(T2-T1) * (p2-p1)
#
#   This is a SIMPLIFICATION. It does not guarantee that the interpolated
#   surface is arbitrage-free between calibrated maturities. The SSVI
#   (Surface SVI) parametrisation of Gatheral & Jacquier (2014) provides
#   calendar-arbitrage-free interpolation but is not implemented here.
#
# ATM term structure: read from CALIBRATED MODEL at m=0, not from noisy data.
#   The two can differ by 0.5–1% on short maturities.
#
# Risk reversal and butterfly approximations use fixed moneyness m=±0.10.
#   True 25Δ RR requires solving N(d₂(K))=0.25 for K — not implemented.
# =============================================================================

def _interpolate_svi_params(
    T_star: float,
    T1: float,
    p1: dict,
    T2: float,
    p2: dict,
) -> dict:
    """Linearly interpolate SVI parameters between two calibrated maturities.

    Args:
        T_star: Target maturity in years.
        T1, T2: Adjacent calibrated maturities (T1 < T_star <= T2).
        p1, p2: Parameter dicts at T1, T2.

    Returns:
        Interpolated parameter dict.

    Notes:
        Linear parameter interpolation is a simplification. It does not
        guarantee calendar-arbitrage-free surfaces between calibrated
        maturities. See Gatheral & Jacquier (2014) SSVI for a proper approach.
    """
    alpha = (T_star - T1) / (T2 - T1)
    return {
        key: p1[key] + alpha * (p2[key] - p1[key])
        for key in p1
    }


def get_iv(
    m: float,
    T: float,
    svi_results: dict[float, dict],
    maturities: list[float],
) -> float:
    """Interpolate SVI surface and return implied vol at (m, T).

    Args:
        m:           Log-moneyness.
        T:           Time to expiry in years.
        svi_results: Map T → calibrate_svi() result.
        maturities:  Sorted list of calibrated maturities in years.

    Returns:
        Implied volatility (annualised). Raises ValueError if w < 0.

    Notes:
        ATM vol (m=0) is read from this function — not from noisy observed
        data interpolation. The two can differ by 0.5–1% at short maturities.
    """
    T = max(T, 1e-6)

    if T <= maturities[0]:
        p = svi_results[maturities[0]]["params"]
    elif T >= maturities[-1]:
        p = svi_results[maturities[-1]]["params"]
    else:
        for i in range(len(maturities) - 1):
            if maturities[i] <= T <= maturities[i + 1]:
                p = _interpolate_svi_params(
                    T, maturities[i], svi_results[maturities[i]]["params"],
                    maturities[i + 1], svi_results[maturities[i + 1]]["params"]
                )
                break

    m_arr = np.array([m])
    w = svi_total_variance(m_arr, p["a"], p["b"], p["rho"], p["mu"], p["xi"])[0]

    if w < 0:
        raise ValueError(
            f"SVI total variance w={w:.6f} < 0 at m={m:.4f}, T={T:.4f}. "
            "Extrapolation failure — w<0 is not clipped silently. "
            "Consider restricting the moneyness range."
        )
    return float(np.sqrt(w / T))


def build_svi_surface(
    svi_results: dict[float, dict],
    maturities: list[float],
    fig_dir: Path,
) -> None:
    """Construct and visualise the SVI implied volatility surface.

    Produces four figures:
        fig1: 2×2 smile fits (observed vs SVI per maturity)
        fig2: 3D surface iv(m, T)
        fig3: ATM term structure (raw observed vs SVI model)
        fig4: Approximate risk reversal and butterfly vs maturity

    Args:
        svi_results: Map T (years) → calibrate_svi() output dict.
        maturities:  Sorted list of calibrated maturities in years.
        fig_dir:     Directory to save figures.

    Notes:
        RR/BF use fixed moneyness m=±0.10.
        Approximate RR/BF using fixed moneyness m=±0.10.
        True 25Δ RR requires solving N(d₂(K))=0.25 for K — not implemented.
        Desk would use the delta-strike mapping.
    """
    m_grid    = np.linspace(-0.35, 0.35, 300)
    T_days_list = [round(T * 365) for T in maturities]

    # --- Figure 1: SVI smile fits -------------------------------------------
    n = len(maturities)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axes1_flat = np.array(axes1).flatten() if n > 1 else [axes1]

    for idx, T in enumerate(maturities):
        ax   = axes1_flat[idx]
        res  = svi_results[T]
        p    = res["params"]
        T_days = T_days_list[idx]

        w_fit = svi_total_variance(m_grid, p["a"], p["b"], p["rho"], p["mu"], p["xi"])
        iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T) * 100  # pct

        bf_sym = "✓" if res["butterfly_satisfied"] else "✗"
        ax.plot(m_grid, iv_fit, "b-", lw=2, label="SVI fit")

        # Overlay raw data if available in svi_results (stored separately)
        if "m_obs" in res and "iv_obs" in res:
            ax.scatter(res["m_obs"], np.array(res["iv_obs"]) * 100,
                       s=25, color="tomato", zorder=5, label="Observed")

        ax.set_title(
            f"T={T_days}d | RMSE={res['rmse']:.3f} | butterfly={bf_sym}",
            fontsize=10
        )
        ax.set_xlabel("Log-moneyness ln(K/F)")
        ax.set_ylabel("Implied Vol (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(maturities), len(axes1_flat)):
        axes1_flat[idx].set_visible(False)

    fig1.suptitle("SVI Smile Calibration — SPY", fontsize=13)
    plt.tight_layout()
    p1 = fig_dir / "fig1_svi_smile_fits.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    log.info("Saved %s", p1)

    # --- Figure 2: 3D surface -----------------------------------------------
    T_fine    = np.linspace(maturities[0], maturities[-1], 40)
    M_grid, T_grid = np.meshgrid(m_grid, T_fine)
    IV_grid   = np.zeros_like(M_grid)

    for i, T_val in enumerate(T_fine):
        for j, m_val in enumerate(m_grid):
            try:
                IV_grid[i, j] = get_iv(m_val, T_val, svi_results, maturities) * 100
            except ValueError:
                IV_grid[i, j] = np.nan

    fig2 = plt.figure(figsize=(10, 7))
    ax3d = fig2.add_subplot(111, projection="3d")
    ax3d.plot_surface(
        M_grid, T_grid * 365, IV_grid,
        cmap="RdYlBu_r", alpha=0.85, linewidth=0.2
    )
    ax3d.set_xlabel("Log-moneyness ln(K/F)")
    ax3d.set_ylabel("Maturity (days)")
    ax3d.set_zlabel("Implied Vol (%)")
    ax3d.set_title("SVI Implied Volatility Surface — SPY")
    p2 = fig_dir / "fig2_svi_surface_3d.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info("Saved %s", p2)

    # --- Figure 3: ATM term structure ---------------------------------------
    iv_atm_model = []
    for T in maturities:
        try:
            iv_atm_model.append(get_iv(0.0, T, svi_results, maturities) * 100)
        except ValueError:
            iv_atm_model.append(np.nan)

    iv_atm_raw = []
    for T in maturities:
        if "atm_iv_raw" in svi_results[T]:
            iv_atm_raw.append(svi_results[T]["atm_iv_raw"] * 100)
        else:
            iv_atm_raw.append(np.nan)

    fig3, (ax_ts, ax_gap) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    T_days_arr = np.array(T_days_list, dtype=float)

    ax_ts.plot(T_days_arr, iv_atm_model, "b-o", lw=2, ms=6, label="SVI model ATM")
    if not all(np.isnan(iv_atm_raw)):
        ax_ts.plot(T_days_arr, iv_atm_raw, "r--s", lw=1.5, ms=6, alpha=0.8,
                   label="Raw observed ATM (noisy)")
    ax_ts.set_ylabel("ATM Implied Vol (%)")
    ax_ts.set_title("ATM Term Structure — SVI Model vs Raw Observed")
    ax_ts.legend()
    ax_ts.grid(True, alpha=0.3)

    gap = np.array(iv_atm_model) - np.array(iv_atm_raw)
    ax_gap.bar(T_days_arr, gap, width=8, color="purple", alpha=0.6)
    ax_gap.axhline(0, color="k", lw=0.8)
    ax_gap.set_xlabel("Maturity (days)")
    ax_gap.set_ylabel("Model − Raw ATM (%)")
    ax_gap.set_title("ATM Model vs Raw Difference")
    ax_gap.grid(True, alpha=0.3)

    plt.tight_layout()
    p3 = fig_dir / "fig3_atm_term_structure.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    log.info("Saved %s", p3)

    # --- Figure 4: Approximate RR and BF ------------------------------------
    # Approximate RR/BF using fixed moneyness m=±0.10.
    # True 25Δ RR requires solving N(d₂(K))=0.25 for K — not implemented.
    # Desk would use the delta-strike mapping.
    rr_approx, bf_approx = [], []
    for T in maturities:
        try:
            iv_otm_call = get_iv(+0.10, T, svi_results, maturities)
            iv_otm_put  = get_iv(-0.10, T, svi_results, maturities)
            iv_atm_     = get_iv( 0.00, T, svi_results, maturities)
            rr_approx.append((iv_otm_call - iv_otm_put) * 100)
            bf_approx.append((iv_otm_call + iv_otm_put - 2 * iv_atm_) * 100)
        except ValueError:
            rr_approx.append(np.nan)
            bf_approx.append(np.nan)

    fig4, (ax_rr, ax_bf) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_rr.bar(T_days_arr, rr_approx, width=8, color="steelblue", alpha=0.8)
    ax_rr.axhline(0, color="k", lw=0.8)
    ax_rr.set_ylabel("Approx RR (%)")
    ax_rr.set_title(
        "Approximate Risk Reversal: iv(m=+0.10) − iv(m=−0.10)\n"
        "[APPROXIMATION: fixed moneyness, not true 25Δ RR]"
    )
    ax_rr.grid(True, alpha=0.3)

    ax_bf.bar(T_days_arr, bf_approx, width=8, color="tomato", alpha=0.8)
    ax_bf.axhline(0, color="k", lw=0.8)
    ax_bf.set_xlabel("Maturity (days)")
    ax_bf.set_ylabel("Approx BF (%)")
    ax_bf.set_title(
        "Approximate Butterfly: iv(m=+0.10) + iv(m=−0.10) − 2·iv(m=0)\n"
        "[APPROXIMATION: fixed moneyness, not true 25Δ BF]"
    )
    ax_bf.grid(True, alpha=0.3)

    plt.tight_layout()
    p4 = fig_dir / "fig4_skew_analysis.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    log.info("Saved %s", p4)

    return {
        "iv_atm_model": iv_atm_model,
        "rr_approx": rr_approx,
        "bf_approx": bf_approx,
    }


# =============================================================================
# SECTION 7 — Desk Interpretation
# =============================================================================
# Concise trading desk commentary on the calibrated surface.
# These are printed to stdout as [DESK VIEW] blocks.
# =============================================================================

def print_desk_view_smile(
    T_days: int,
    iv_atm: float,
    rr: float,
    bf: float,
) -> None:
    """Print desk-style commentary on a single smile slice."""
    print(
        f"\n[DESK VIEW] T={T_days}d: "
        f"ATM vol = {iv_atm * 100:.1f}%, "
        f"skew (approx RR m=±0.10) = {rr * 100:.1f}%, "
        f"butterfly (approx BF) = {bf * 100:.2f}%. "
        "Negative skew confirms put premium: "
        "the market prices tail risk asymmetrically downward."
    )


def print_desk_view_term_structure(
    maturities_days: list[int],
    iv_atm_list: list[float],
) -> None:
    """Print desk-style commentary on the ATM term structure."""
    term_str = " → ".join(
        f"{T}d={iv * 100:.1f}%"
        for T, iv in zip(maturities_days, iv_atm_list)
        if not np.isnan(iv)
    )
    # Determine structure shape
    valid_ivs = [iv for iv in iv_atm_list if not np.isnan(iv)]
    if len(valid_ivs) >= 2:
        if valid_ivs[0] > valid_ivs[-1]:
            shape_desc = (
                "Structure is inverted: short-dated vol elevated, "
                "consistent with near-term event risk."
            )
        else:
            shape_desc = (
                "Structure is normal: long-dated vol elevated, "
                "consistent with long-run uncertainty premium."
            )
    else:
        shape_desc = "Insufficient maturities to characterise term structure shape."

    print(
        f"\n[DESK VIEW] ATM vol term structure: {term_str}. "
        f"{shape_desc}"
    )


def print_desk_view_arbitrage(
    svi_results: dict[float, dict],
    maturities: list[float],
    n_cal_violations: int,
) -> None:
    """Print desk-style commentary on arbitrage checks."""
    n_bf_pass = sum(
        1 for T in maturities
        if T in svi_results and svi_results[T]["butterfly_satisfied"]
    )
    n_total = len([T for T in maturities if T in svi_results])

    cal_msg = (
        f"{n_cal_violations} violations on calibrated surface. "
        "Calendar violations indicate the linear parameter interpolation "
        "is insufficient between calibrated maturities. SSVI would resolve this."
        if n_cal_violations > 0
        else "No calendar violations on calibrated surface."
    )

    print(
        f"\n[DESK VIEW] Arbitrage summary: "
        f"butterfly [{n_bf_pass}/{n_total} maturities pass], "
        f"calendar [{cal_msg}]"
    )


# =============================================================================
# SYNTHETIC DATA PIPELINE (explicit opt-in only — NOT the default path)
# =============================================================================
# This path is never triggered automatically.
# The known SVI parameters allow testing calibration correctness
# against a KNOWN ground truth, not against its own generating process.
# =============================================================================

SYNTHETIC_PARAMS = {
    30:  {"a": 0.020, "b": 0.120, "rho": -0.70, "m": 0.00, "xi": 0.08},
    60:  {"a": 0.025, "b": 0.110, "rho": -0.65, "m": 0.00, "xi": 0.09},
    90:  {"a": 0.030, "b": 0.100, "rho": -0.60, "m": 0.00, "xi": 0.10},
    180: {"a": 0.040, "b": 0.090, "rho": -0.55, "m": 0.00, "xi": 0.12},
}


def generate_synthetic_chains() -> dict[float, pd.DataFrame]:
    """Generate synthetic option chains from known SVI parameters.

    Returns:
        Map T (years) → synthetic DataFrame with columns matching
        validate_option_chain() requirements.

    Notes:
        Adds realistic noise: σ_noise ~ N(0, 0.005) on implied vols.
        Adds 5% bid > ask pathology and 3% stale quotes.
        DATA_SOURCE must be printed as 'synthetic_with_noise' by the caller.
        This function must NOT be called in the default pipeline.
    """
    chains = {}
    m_nodes = np.linspace(-0.30, 0.30, 25)

    for T_days, sp in SYNTHETIC_PARAMS.items():
        T = T_days / 365.0
        a, b, rho, mu, xi = sp["a"], sp["b"], sp["rho"], sp["m"], sp["xi"]

        w_true = svi_total_variance(m_nodes, a, b, rho, mu, xi)
        iv_true = np.sqrt(np.maximum(w_true, 0.0) / T)

        # Realistic noise on IV
        noise = rng.normal(0, 0.005, size=len(m_nodes))
        iv_noisy = np.maximum(iv_true + noise, 0.01)

        # Reconstruct mid prices from noisy IV (using S=540 reference)
        S_ref = 540.0
        F_ref = S_ref * np.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * T)
        K_nodes = F_ref * np.exp(m_nodes)

        mids  = []
        bids  = []
        asks  = []
        otypes = []
        for K, iv_ in zip(K_nodes, iv_noisy):
            otype = "call" if K >= F_ref else "put"
            mid_  = _bs_price(S_ref, K, T, RISK_FREE_RATE, DIVIDEND_YIELD, iv_, otype)
            spread = mid_ * 0.02   # 2% spread
            bid_ = mid_ - spread
            ask_ = mid_ + spread
            mids.append(mid_)
            bids.append(bid_)
            asks.append(ask_)
            otypes.append(otype)

        bids_arr = np.array(bids)
        asks_arr = np.array(asks)

        # Inject 5% bid > ask pathology
        n_inv = max(1, int(0.05 * len(m_nodes)))
        inv_idx = rng.choice(len(m_nodes), size=n_inv, replace=False)
        bids_arr[inv_idx], asks_arr[inv_idx] = (
            asks_arr[inv_idx].copy(), bids_arr[inv_idx].copy()
        )

        # Inject 3% stale (mid < intrinsic)
        n_stale = max(1, int(0.03 * len(m_nodes)))
        stale_idx = rng.choice(len(m_nodes), size=n_stale, replace=False)
        for idx_ in stale_idx:
            bids_arr[idx_] = 0.001
            asks_arr[idx_] = 0.002

        df = pd.DataFrame({
            "strike":          K_nodes,
            "bid":             bids_arr,
            "ask":             asks_arr,
            "lastPrice":       np.array(mids),
            "openInterest":    rng.integers(100, 5000, size=len(m_nodes)),
            "impliedVolatility": iv_noisy,
            "option_type":     otypes,
            "expiry":          f"synthetic_{T_days}d",
        })
        chains[T] = df

    return chains


def run_synthetic_pipeline() -> None:
    """Run the full pipeline on synthetic data (explicit opt-in only)."""
    global DATA_SOURCE
    DATA_SOURCE = "synthetic_with_noise"
    log.info("DATA_SOURCE = synthetic_with_noise (explicitly requested)")
    chains = generate_synthetic_chains()
    _run_pipeline(chains, S0=540.0, r=RISK_FREE_RATE, q=DIVIDEND_YIELD)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def _run_pipeline(
    chains_by_expiry: dict,
    S0: float,
    r: float,
    q: float,
) -> None:
    """Execute the full IV → SVI → arbitrage → surface pipeline.

    Args:
        chains_by_expiry: Map of expiry label → raw option chain DataFrame.
        S0: Spot price.
        r:  Risk-free rate.
        q:  Dividend yield.
    """
    reference_date = datetime.date.today()
    log.info("Reference date  : %s", reference_date)
    log.info("S0              : %.4f", S0)
    log.info("Risk-free rate  : %.4f", r)
    log.info("Dividend yield  : %.4f", q)
    log.info("DATA_SOURCE     : %s", DATA_SOURCE)

    # --- Resolve expiry → T mapping -----------------------------------------
    # For live/cached data: parse expiry string to date.
    # For synthetic: T is already the dict key (float in years).
    maturity_map: dict[float, tuple[str, pd.DataFrame]] = {}

    for key, df in chains_by_expiry.items():
        if isinstance(key, float):
            # Synthetic path: key is T in years
            T_years = key
            label   = f"synthetic_{round(key * 365)}d"
        else:
            # Live/cached path: key is expiry date string
            try:
                exp_date = datetime.date.fromisoformat(str(key))
                T_days   = (exp_date - reference_date).days
                if T_days <= 0:
                    log.warning("Expiry %s is in the past — skipping.", key)
                    continue
                T_years = T_days / 365.0
                label   = str(key)
            except ValueError:
                log.warning("Cannot parse expiry '%s' — skipping.", key)
                continue
        maturity_map[T_years] = (label, df)

    maturities = sorted(maturity_map.keys())
    if not maturities:
        raise RuntimeError("No valid maturities found in chains.")

    # --- Sections 2 & 3: cleaning, IV extraction, descriptive analysis ------
    log.info("=" * 60)
    log.info("SECTION 2 — Data Cleaning and IV Extraction")
    log.info("=" * 60)

    chains_iv: dict[float, pd.DataFrame] = {}
    svi_results: dict[float, dict] = {}

    for T in maturities:
        label, df_raw = maturity_map[T]
        T_days = round(T * 365)
        log.info("--- T=%dd (%s) ---", T_days, label)

        # Validate schema
        validate_option_chain(df_raw)

        # Split by option type, clean, extract IV
        dfs_by_type = {}
        for otype in ["call", "put"]:
            df_type = df_raw[df_raw["option_type"].str.lower() == otype].copy()
            if len(df_type) == 0:
                continue
            try:
                df_clean = clean_option_chain(df_type, S0, r, q, T)
                df_iv    = extract_iv_for_chain(df_clean, S0, r, q, T)
                dfs_by_type[otype] = df_iv
            except ValueError as e:
                log.warning("T=%dd %s: %s", T_days, otype, e)
                continue

        if not dfs_by_type:
            log.warning("T=%dd: no valid data after cleaning. Skipping.", T_days)
            continue

        # Call-put parity check on ALL cleaned strikes (diagnostic).
        # Large violations here are EXPECTED on real data: at any strike the
        # in-the-money option is illiquid with a stale/wide quote, so its IV
        # is unreliable. This motivates the OTM selection below.
        verify_call_put_parity(
            dfs_by_type.get("call", pd.DataFrame()),
            dfs_by_type.get("put",  pd.DataFrame()),
            T
        )

        # --- OTM selection (desk standard) ----------------------------------
        # Build the calibration smile from OUT-OF-THE-MONEY options only:
        #   calls with K > F, puts with K <= F.
        # ITM options carry mostly intrinsic value with tiny, noisy time value,
        # so their inverted IV is unstable and corrupts the fit (it flips the
        # skew sign and inflates RMSE). Using OTM options is the standard desk
        # construction of the implied-vol smile.
        F = S0 * np.exp((r - q) * T)
        otm_parts = []
        if "call" in dfs_by_type:
            c = dfs_by_type["call"]
            otm_parts.append(c[c["strike"] > F])
        if "put" in dfs_by_type:
            p = dfs_by_type["put"]
            otm_parts.append(p[p["strike"] <= F])
        df_combined = pd.concat(otm_parts, ignore_index=True)
        log.info(
            "  T=%dd: OTM selection → %d points (OTM calls K>F, OTM puts K<=F)",
            T_days, len(df_combined)
        )

        if len(df_combined) < 5:
            log.warning(
                "T=%dd: only %d OTM points after selection. Skipping maturity.",
                T_days, len(df_combined)
            )
            continue

        chains_iv[T] = df_combined

    if not chains_iv:
        raise RuntimeError("No maturities survived cleaning and IV extraction.")

    maturities = sorted(chains_iv.keys())

    # --- Section 3: raw smile plots + raw calendar check --------------------
    log.info("=" * 60)
    log.info("SECTION 3 — Descriptive Analysis (raw IV)")
    log.info("=" * 60)

    plot_raw_smiles(chains_iv, FIG_DIR / "fig0_raw_smiles.png")
    check_raw_calendar_arbitrage(chains_iv)

    # --- Section 4: SVI calibration -----------------------------------------
    log.info("=" * 60)
    log.info("SECTION 4 — SVI Calibration")
    log.info("=" * 60)

    for T in maturities:
        df  = chains_iv[T]
        T_days = round(T * 365)

        m_arr = df["log_moneyness"].values
        w_arr = (df["iv"].values) ** 2 * T

        log.info("Calibrating SVI for T=%dd (%d points)...", T_days, len(m_arr))
        try:
            result = calibrate_svi(m_arr, w_arr, T)
        except RuntimeError as e:
            log.error("T=%dd calibration failed: %s", T_days, e)
            continue

        # Attach raw observations to result for plotting
        result["m_obs"]  = list(m_arr)
        result["iv_obs"] = list(df["iv"].values)

        # ATM raw estimate (closest to m=0)
        idx_atm = np.argmin(np.abs(m_arr))
        result["atm_iv_raw"] = float(df["iv"].values[idx_atm])

        p = result["params"]
        log.info(
            "T=%dd: a=%.5f b=%.5f rho=%.4f mu=%.5f xi=%.5f | "
            "RMSE=%.4f | butterfly=%s | converged=%s | n_iter=%d",
            T_days,
            p["a"], p["b"], p["rho"], p["mu"], p["xi"],
            result["rmse"],
            "PASS" if result["butterfly_satisfied"] else "FAIL",
            result["converged"],
            result["n_iter"]
        )

        svi_results[T] = result

    if not svi_results:
        raise RuntimeError("SVI calibration failed for all maturities.")

    maturities = sorted(svi_results.keys())

    # --- Section 5: arbitrage verification ----------------------------------
    # verify_svi_surface returns the single authoritative calendar-violation
    # count (computed on |m|<=0.35). Reusing it guarantees the desk-view
    # summary cannot report a different number from Section 5.
    n_cal_viol = verify_svi_surface(svi_results, maturities)

    # --- Section 6: surface construction and visualization ------------------
    log.info("=" * 60)
    log.info("SECTION 6 — Surface Construction and Visualization")
    log.info("=" * 60)

    surface_metrics = build_svi_surface(svi_results, maturities, FIG_DIR)

    # --- Section 7: desk interpretation -------------------------------------
    log.info("=" * 60)
    log.info("SECTION 7 — Desk Interpretation")
    log.info("=" * 60)

    for i, T in enumerate(maturities):
        T_days_ = round(T * 365)
        try:
            iv_atm_ = get_iv(0.0, T, svi_results, maturities)
            rr_     = (
                get_iv(+0.10, T, svi_results, maturities)
                - get_iv(-0.10, T, svi_results, maturities)
            )
            bf_     = (
                get_iv(+0.10, T, svi_results, maturities)
                + get_iv(-0.10, T, svi_results, maturities)
                - 2 * get_iv(0.0, T, svi_results, maturities)
            )
        except ValueError:
            continue
        print_desk_view_smile(T_days_, iv_atm_, rr_, bf_)

    T_days_list_     = [round(T * 365) for T in maturities]
    iv_atm_list_     = surface_metrics["iv_atm_model"]
    iv_atm_fracs     = [iv / 100.0 for iv in iv_atm_list_]  # convert back from pct
    print_desk_view_term_structure(T_days_list_, iv_atm_fracs)
    print_desk_view_arbitrage(svi_results, maturities, n_cal_viol)

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"DATA_SOURCE : {DATA_SOURCE}")
    print(f"Figures     : {FIG_DIR}/")
    print("=" * 60)


def resolve_spot() -> float:
    """Resolve the SPY spot price robustly, never silently guessing.

    Resolution order:
        1. Live yfinance quote.
        2. 'underlying_spot' column of the most recent cached chain CSV
           (written by fetch_data.py).

    Returns:
        Spot price.

    Raises:
        RuntimeError: if neither source is available. This is deliberate —
            a wrong spot silently shifts the forward and corrupts the entire
            surface (the OTM/ITM split, moneyness, calibration). Failing loudly
            is far safer than defaulting to an arbitrary constant.
    """
    # 1. Live
    try:
        import yfinance as yf
        hist = yf.Ticker("SPY").history(period="1d")
        if not hist.empty:
            s = float(hist["Close"].iloc[-1])
            log.info("SPY spot: %.4f (from yfinance)", s)
            return s
    except Exception as e:
        log.warning("Live spot fetch failed: %s", e)

    # 2. Cached underlying_spot column
    for f in sorted(DATA_DIR.glob("spy_chains_*.csv"), reverse=True):
        try:
            head = pd.read_csv(f, nrows=1)
        except Exception:
            continue
        if "underlying_spot" in head.columns:
            s = float(head["underlying_spot"].iloc[0])
            log.info("SPY spot: %.4f (from cached %s)", s, f.name)
            return s

    raise RuntimeError(
        "Cannot resolve SPY spot: yfinance is unavailable and no cached CSV "
        "carries an 'underlying_spot' column. Run `python fetch_data.py` once "
        "while online to record the spot. A wrong spot corrupts the whole "
        "surface, so the pipeline refuses to guess."
    )


def main() -> None:
    """Entry point: download SPY chains and run the full pipeline."""
    global DATA_SOURCE

    reference_date = datetime.date.today()

    # Resolve spot robustly (live, else from cached CSV). Never default silently.
    S0 = resolve_spot()

    chains, DATA_SOURCE = download_spy_chains(reference_date)
    log.info("DATA_SOURCE = %s", DATA_SOURCE)

    _run_pipeline(chains, S0=S0, r=RISK_FREE_RATE, q=DIVIDEND_YIELD)


if __name__ == "__main__":
    main()
