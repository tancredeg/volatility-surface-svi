"""
fetch_data.py
=============
Standalone REAL DATA extractor for the SVI pipeline.

Run this on YOUR machine (VS Code, terminal) where there is no network
restriction. It downloads live SPY option chains via yfinance, verifies the
data is genuinely real (not synthetic), and writes it to data/.

    python fetch_data.py

It NEVER generates synthetic data. If the download fails it exits with a
non-zero status and an explicit diagnostic — it does not fabricate a chain.

Why a separate script:
    The main pipeline (iv_surface.py) consumes whatever is in data/.
    Keeping acquisition separate means you can refresh data on a schedule
    (cron / Task Scheduler) without re-running calibration, and the
    provenance of every CSV is unambiguous.
"""

import sys
import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TARGET_MATURITIES_DAYS = [30, 60, 90, 180]
REQUIRED_COLUMNS = ["bid", "ask", "strike", "lastPrice",
                    "openInterest", "impliedVolatility"]


def _nearest_expiry(expiries, target_days, ref_date):
    best, best_diff = None, float("inf")
    for e in expiries:
        try:
            d = datetime.date.fromisoformat(e)
        except ValueError:
            continue
        diff = abs((d - ref_date).days - target_days)
        if diff < best_diff:
            best, best_diff = e, diff
    return best


def fetch_real_spy_chains() -> Path:
    """Download live SPY chains and write a single CSV. Returns its path.

    Raises:
        SystemExit: on any failure, with a diagnostic message. Never falls
                    back to synthetic data.
    """
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed. Run: pip install -r requirements.txt")

    ref_date = datetime.date.today()
    ticker = yf.Ticker("SPY")

    # --- spot ---------------------------------------------------------------
    hist = ticker.history(period="1d")
    if hist.empty:
        sys.exit(
            "yfinance returned no price history for SPY. "
            "Likely causes: no internet, rate limiting, or a yfinance/Yahoo "
            "API change. Retry in a few minutes or upgrade yfinance."
        )
    spot = float(hist["Close"].iloc[-1])

    # --- expiries -----------------------------------------------------------
    expiries = list(ticker.options)
    if not expiries:
        sys.exit(
            "yfinance returned an empty expiry list for SPY. "
            "This usually means Yahoo is rate-limiting you or the API changed. "
            "Retry later or run: pip install --upgrade yfinance"
        )

    selected = []
    for tgt in TARGET_MATURITIES_DAYS:
        e = _nearest_expiry(expiries, tgt, ref_date)
        if e and e not in selected:
            selected.append(e)

    # --- chains -------------------------------------------------------------
    frames = []
    for e in selected:
        chain = ticker.option_chain(e)
        for df, otype in [(chain.calls, "call"), (chain.puts, "put")]:
            df = df.copy()
            df["option_type"] = otype
            df["expiry"] = e
            frames.append(df)

    if not frames:
        sys.exit("No option chains retrieved. Aborting (no synthetic fallback).")

    combined = pd.concat(frames, ignore_index=True)

    # --- schema validation --------------------------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in combined.columns]
    if missing:
        sys.exit(
            f"yfinance schema changed. Missing columns: {missing}. "
            f"Got: {list(combined.columns)}."
        )

    # --- realness verification ---------------------------------------------
    verify_is_real(combined, spot)

    # Persist the spot so downstream scripts never have to guess it offline.
    combined["underlying_spot"] = spot

    out = DATA_DIR / f"spy_chains_{ref_date.isoformat()}.csv"
    combined.to_csv(out, index=False)

    print("=" * 60)
    print("REAL DATA DOWNLOAD COMPLETE")
    print(f"  reference date : {ref_date}")
    print(f"  SPY spot       : {spot:.2f}")
    print(f"  expiries       : {selected}")
    print(f"  rows           : {len(combined)}")
    print(f"  saved to       : {out}")
    print("=" * 60)
    return out


def verify_is_real(df: pd.DataFrame, spot: float) -> None:
    """Heuristic guards that the data is genuinely market data, not synthetic.

    Raises SystemExit if the data fails any realness check.

    Checks:
        1. Expiry labels are ISO dates, not 'synthetic_*'.
        2. Both calls and puts share at least some strikes (real chains do;
           the synthetic generator assigns each strike to exactly one type).
        3. Strikes span a plausible range around spot.
        4. A non-trivial fraction of rows carry open interest > 0.
    """
    # 1. No synthetic expiry labels
    if df["expiry"].astype(str).str.startswith("synthetic").any():
        sys.exit("Data contains 'synthetic_*' expiry labels — refusing to treat as real.")

    # 2. Overlapping call/put strikes
    calls = set(df.loc[df["option_type"] == "call", "strike"].round(2))
    puts  = set(df.loc[df["option_type"] == "put",  "strike"].round(2))
    overlap = calls & puts
    if len(overlap) < 5:
        sys.exit(
            f"Only {len(overlap)} overlapping call/put strikes found. "
            "Real SPY chains quote calls and puts on a shared strike ladder; "
            "this looks synthetic or malformed."
        )

    # 3. Strike range plausibility
    kmin, kmax = df["strike"].min(), df["strike"].max()
    if not (kmin < spot < kmax):
        sys.exit(
            f"Spot {spot:.2f} is outside the strike range "
            f"[{kmin:.2f}, {kmax:.2f}] — implausible for real data."
        )

    # 4. Open interest present
    frac_oi = (df["openInterest"].fillna(0) > 0).mean()
    if frac_oi < 0.05:
        sys.exit(
            f"Only {frac_oi:.1%} of rows have positive open interest — "
            "real liquid SPY chains have far more. Aborting."
        )

    print(f"Realness checks passed: {len(overlap)} shared strikes, "
          f"{frac_oi:.0%} rows with open interest.")


if __name__ == "__main__":
    fetch_real_spy_chains()
