<!-- BEGIN:key-results -->
- **Fit.** On 785 out-of-the-money quotes, constrained (arbitrage-free) SVI has an implied-vol RMSE of 0.09-0.20 vol points per expiry, against 0.09-0.20 for unconstrained SVI and 0.71-1.41 for SSVI. Removing arbitrage costs at most 0.000 vol point per expiry; SSVI's single set of shape parameters for all expiries costs 0.62-1.28.
- **Arbitrage.** Unconstrained SVI shows butterfly arbitrage (negative density) on 1 of 4 expiries (g(k) as low as -0.00, at k = 0.40 to 0.40) and no calendar crossing; none of these violations lie inside the quoted strike range (k <= 0.19). The fit looks clean on the data and breaks in the extrapolated wings, which Dupire local vol and every wing-sensitive product use. Constrained SVI and SSVI have no violation, confirmed by two independent checks (Durrleman g and convexity of call prices in strike); the SSVI surface is arbitrage-free at every maturity, not only at the listed ones (gamma <= 1/2).
- **Speed.** A full surface takes 0.06 s with SSVI, 0.4 s with raw SVI and 1.4 s with constrained SVI (median, one CPU core).
- **Structuring.** On a down-and-in put expiring in 89 days (strike 100%, barrier 90%), Dupire local vol gives 2.38% of notional against 1.60% for Black-Scholes at the strike vol (15.6%): the flat-vol desk would under-charge the barrier by 0.79% of notional, i.e. overstate a reverse-convertible coupon by about 3.2% per annum. Using the barrier-level vol instead (23.3%) gives 3.62%.
- **Model check.** The local-vol Monte Carlo reprices the SSVI vanillas within 0.05 vol point (2.5 standard errors at most); SSVI itself is off the market by 1.68 vol point at the 90% strike, which bounds the precision of the barrier prices.
<!-- END:key-results -->

<!-- BEGIN:fit-table -->
| Expiry | T (days) | Quotes | ATM vol | RMSE SVI raw | RMSE SVI constrained | RMSE SSVI | Half bid-ask |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-02 | 29 | 159 | 13.29% | 0.20 | 0.20 | 1.31 | 0.037 |
| 2026-07-31 | 58 | 198 | 14.39% | 0.12 | 0.12 | 1.34 | 0.026 |
| 2026-08-31 | 89 | 215 | 15.15% | 0.13 | 0.13 | 1.41 | 0.024 |
| 2026-11-30 | 180 | 213 | 16.59% | 0.09 | 0.09 | 0.71 | 0.024 |
<!-- END:fit-table -->

<!-- BEGIN:arbitrage-table -->
| Model | Slices with g(k) < 0 | of which inside quoted range | Min g(k) | Negative-density points (price check) | Calendar crossings: quoted / extrapolated (grid points) |
|---|---:|---:|---:|---:|---:|
| SVI raw | 1 / 4 | 0 | -0.003 | 105 | 0 / 0 |
| SVI constrained | 0 / 4 | 0 | +0.000 | 0 | 0 / 0 |
| SSVI | 0 / 4 | 0 | +0.254 | 0 | 0 / 0 |
<!-- END:arbitrage-table -->

<!-- BEGIN:timing-table -->
| Model | Median time, full surface (s) | Per slice (ms) | Runs |
|---|---:|---:|---:|
| SVI raw | 0.41 | 103 | 10 |
| SVI constrained | 1.36 | 339 | 10 |
| SSVI | 0.06 | 14 | 10 |
<!-- END:timing-table -->

<!-- BEGIN:stability-table -->
| Model | Std ATM vol | Std put wing k=-0.5 | Std call wing k=+0.3 | Fits at a bound |
|---|---:|---:|---:|---:|
| SVI raw | 0.013 | 0.228 | 1.055 | 0 |
| SVI constrained | 0.013 | 0.189 | 0.801 | 0 |
| SSVI | 0.061 | 0.089 | 0.072 | 0 |
<!-- END:stability-table -->

<!-- BEGIN:barrier-table -->
| Barrier | DIP local vol | DIP BS strike vol | DIP BS barrier vol | LV - BS(strike) | Coupon effect p.a. | DOC local vol | DOC BS strike vol | DOC gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 75% | 1.14% | 0.00% | 2.09% | +1.14% | +4.7% | 3.41% | 3.41% | +0.1% |
| 80% | 1.54% | 0.07% | 2.72% | +1.47% | +6.0% | 3.40% | 3.41% | -0.0% |
| 85% | 1.97% | 0.48% | 3.30% | +1.49% | +6.1% | 3.38% | 3.41% | -0.8% |
| 90% | 2.38% | 1.60% | 3.62% | +0.79% | +3.2% | 3.27% | 3.40% | -3.8% |
| 95% | 2.69% | 2.62% | 3.43% | +0.07% | +0.3% | 2.77% | 3.03% | -8.6% |
<!-- END:barrier-table -->

<!-- BEGIN:repricing-table -->
| Strike | Type | Market vol | SSVI vol | Local-vol MC implied vol | MC error (std errors) |
|---:|---|---:|---:|---:|---:|
| 80% | put | 29.53% | 29.98% | 29.98% | -0.1 |
| 90% | put | 21.58% | 23.26% | 23.29% | +0.5 |
| 100% | call | 15.19% | 15.64% | 15.69% | +2.5 |
| 110% | call | 12.28% | 12.70% | 12.73% | +1.2 |
<!-- END:repricing-table -->
