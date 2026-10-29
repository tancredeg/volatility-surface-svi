"""Download a live SPY option chain from Yahoo Finance into data/raw/ (optional, needs yfinance).

    pip install -e .[fetch]
    python scripts/fetch_data.py

Run it during the US session (15:30-22:00 Paris time): outside it Yahoo
returns bid = ask = 0 and only stale last-trade prices, which the pipeline
rejects. The script refuses to save such a chain.

Outputs
    data/raw/spy_chains_<YYYYMMDD>.csv   raw quotes (local only: Yahoo's terms
                                         forbid redistribution; data/raw/ is git-ignored)
    data/snapshots/spy_<YYYYMMDD>.json   metadata skeleton: download time and
                                         SHA-256 are filled in; the discount rate
                                         (FRED DTB3 of the day) and the cash
                                         dividends must be added by hand.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "snapshots"
TARGET_MATURITIES_DAYS = (30, 60, 90, 180)
MIN_TWO_SIDED_SHARE = 0.5


def nearest_expiry(expiries: list[str], target_days: int, ref: dt.date) -> str | None:
    """Listed expiry whose distance to ref + target_days is smallest."""
    best, best_diff = None, float("inf")
    for e in expiries:
        try:
            diff = abs((dt.date.fromisoformat(e) - ref).days - target_days)
        except ValueError:
            continue
        if diff < best_diff:
            best, best_diff = e, diff
    return best


def main() -> None:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed: pip install -e .[fetch]")

    fetched_at = dt.datetime.now(dt.UTC).replace(microsecond=0)
    ticker = yf.Ticker("SPY")
    expiries = list(ticker.options)
    if not expiries:
        sys.exit("Yahoo returned no SPY expiries (rate limiting or API change). Retry later.")
    selected: list[str] = []
    for target in TARGET_MATURITIES_DAYS:
        e = nearest_expiry(expiries, target, fetched_at.date())
        if e and e not in selected:
            selected.append(e)

    frames = []
    for e in selected:
        chain = ticker.option_chain(e)  # named tuple (calls, puts, underlying)
        for df, otype in ((chain.calls, "call"), (chain.puts, "put")):
            df = df.copy()
            df["option_type"] = otype
            df["expiry"] = e
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["fetched_at_utc"] = fetched_at.isoformat()

    two_sided = float(((combined["bid"] > 0) & (combined["ask"] > combined["bid"])).mean())
    if two_sided < MIN_TWO_SIDED_SHARE:
        sys.exit(
            f"Only {two_sided:.1%} of rows carry a two-sided quote: the US market is probably closed. "
            "Nothing saved; rerun during the US session."
        )

    tag = fetched_at.strftime("%Y%m%d")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"spy_chains_{tag}.csv"
    combined.to_csv(out, index=False)
    meta = {
        "underlying": "SPY",
        "description": f"SPY listed options, expiries {', '.join(selected)}, Yahoo Finance, downloaded "
        f"{fetched_at.isoformat()}.",
        "chain_file": str(out.relative_to(ROOT).as_posix()),
        "chain_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "valuation_time_utc": fetched_at.isoformat().replace("+00:00", "Z"),
        "valuation_time_rule": "Download time (Yahoo quotes are delayed by about 15 minutes).",
        "rate": {
            "rate_cc": None,
            "source": "TO FILL: FRED DTB3 of the day, converted with "
            "volsurf.forward.rate_from_tbill_discount",
        },
        "dividends": [],
        "synthetic": False,
    }
    meta_path = META_DIR / f"spy_{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"Saved {len(combined)} quotes ({two_sided:.0%} two-sided) for {selected} to {out.relative_to(ROOT)}"
    )
    print(f"Metadata skeleton: {meta_path.relative_to(ROOT)} (fill in the rate and the dividends)")


if __name__ == "__main__":
    main()
