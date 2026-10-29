# Data

| Folder | Versioned | Content |
|---|---|---|
| `snapshots/` | yes | Metadata of the real SPY snapshot: valuation time, discount rate, cash dividends (with sources) and the SHA-256 of the raw chain file. |
| `raw/` | **no** | Raw SPY option chains downloaded from Yahoo Finance (`scripts/fetch_data.py`). Yahoo's terms forbid redistributing its data, so these files stay local. |
| `sample/` | yes | Synthetic **SPY-like** chain (`spy_like_20260603.csv`) and its metadata. Built by `scripts/make_sample.py` from the constrained-SVI fits of the real snapshot: same expiries, forwards, discounting, strike coverage, quote noise and bid-ask widths. Not market data. |

## Real snapshot (`snapshots/spy_20260603.json`)

* **Underlying:** SPY (SPDR S&P 500 ETF Trust), listed options, PM-settled.
* **Expiries:** 2026-07-02, 2026-07-31, 2026-08-31, 2026-11-30 (29 to 180 days).
* **Valuation time:** 2026-06-03 14:35:53 UTC, the latest `lastTradeDate` in the chain. Yahoo quotes are delayed, so the true quote time is uncertain by 15-20 minutes (effect on T below 0.1% at 29 days).
* **Time to expiry:** ACT/365 from the valuation time to 16:00 America/New_York on the expiry date.
* **Discount rate:** FRED DTB3 (3-month T-bill, discount basis) on 2026-06-03 = 3.63%, converted to a continuously compounded rate r = -ln(1 - d·91/360)·365/91 = 3.697%.
* **Dividends (barrier pricer only):** SPY cash distributions with ex-dates 2026-06-18 (USD 1.903516) and 2026-09-18 (USD 1.888834), from the SSGA 2026 SPDR distribution schedule.
* **Forwards:** read from the quotes by put-call parity, F = median over near-the-money strikes of K + (C - P)/D. Spot, rate and dividend yield therefore do not enter the implied vols.

A second chain downloaded on 2026-06-04 before the US open (98.5% of rows with bid = ask = 0, i.e. only stale last-trade prices) is the one the v0 code used; it is now rejected by design (`StaleQuotesError`). See `AUDIT.md`.

## Cleaning (applied per expiry, `volsurf.data.prepare_snapshot`)

| Step | Rule | Default |
|---|---|---|
| 1 | Minimum maturity | 7 calendar days |
| 2 | Two-sided quote: bid > 0 and ask > bid (last-trade prices are never used) | chain rejected if fewer than 50% of rows qualify |
| 3 | Parity forward on strikes within ±3% of the ATM strike | at least 3 call/put pairs |
| 4 | Out-of-the-money options only: calls K > F, puts K ≤ F | |
| 5 | Minimum bid (removes minimum-tick quotes) | USD 0.05 |
| 6 | Black-76 implied vol exists at bid, mid and ask | |
| 7 | Moneyness filter in forward delta | \|Δ\| ≥ 0.5% |
| 8 | Bid-ask width in vol | ≤ 1 vol point |

The number of quotes kept after each step is written to `results/<snapshot>/filter_log.csv`. On the real snapshot, 1,215 rows become 680 calibration quotes (134, 217, 225 and 104 per expiry).

## Schema of a chain file

`expiry` (YYYY-MM-DD), `option_type` (`call`/`put`), `strike`, `bid`, `ask`. Any other column (Yahoo's `lastPrice`, `impliedVolatility`, ...) is ignored.
