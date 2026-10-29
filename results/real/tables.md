<!-- BEGIN:key-results -->
- **Fit.** On 680 out-of-the-money quotes, constrained (arbitrage-free) SVI has an implied-vol RMSE of 0.09-0.21 vol points per expiry, against 0.06-0.14 for unconstrained SVI and 0.73-1.42 for SSVI. Removing arbitrage costs at most 0.065 vol point per expiry; SSVI's single set of shape parameters for all expiries costs 0.64-1.34.
- **Arbitrage.** Unconstrained SVI shows butterfly arbitrage (negative density) on 3 of 4 expiries (g(k) as low as -0.44, at k = 0.95 to 1.19) and 1 of 3 pairs of expiries cross in total variance (for k >= 0.27); none of these violations lie inside the quoted strike range (k <= 0.18). The fit looks clean on the data and breaks in the extrapolated wings, which Dupire local vol and every wing-sensitive product use. Constrained SVI and SSVI have no violation, confirmed by two independent checks (Durrleman g and convexity of call prices in strike); the SSVI surface is arbitrage-free at every maturity, not only at the listed ones (gamma <= 1/2).
- **Speed.** A full surface takes 0.05 s with SSVI, 1.8 s with raw SVI and 3.4 s with constrained SVI (median, one CPU core).
- **Structuring.** On a down-and-in put expiring in 89 days (strike 100%, barrier 90%), Dupire local vol gives 2.35% of notional against 1.57% for Black-Scholes at the strike vol (15.5%): the flat-vol desk would under-charge the barrier by 0.78% of notional, i.e. overstate a reverse-convertible coupon by about 3.2% per annum. Using the barrier-level vol instead (23.0%) gives 3.56%.
- **Model check.** The local-vol Monte Carlo reprices the SSVI vanillas within 0.05 vol point (2.3 standard errors at most); SSVI itself is off the market by 1.17 vol point at the 90% strike, which bounds the precision of the barrier prices.
<!-- END:key-results -->

<!-- BEGIN:fit-table -->
| Expiry | T (days) | Quotes | ATM vol | RMSE SVI raw | RMSE SVI constrained | RMSE SSVI | Half bid-ask |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-02 | 29 | 134 | 13.43% | 0.14 | 0.21 | 1.31 | 0.025 |
| 2026-07-31 | 58 | 217 | 14.40% | 0.06 | 0.12 | 1.38 | 0.019 |
| 2026-08-31 | 89 | 225 | 15.07% | 0.08 | 0.14 | 1.42 | 0.019 |
| 2026-11-30 | 180 | 104 | 16.54% | 0.09 | 0.09 | 0.73 | 0.021 |
<!-- END:fit-table -->

<!-- BEGIN:arbitrage-table -->
| Model | Slices with g(k) < 0 | of which inside quoted range | Min g(k) | Negative-density points (price check) | Calendar crossings: quoted / extrapolated (grid points) |
|---|---:|---:|---:|---:|---:|
| SVI raw | 3 / 4 | 0 | -0.439 | 2398 | 0 / 1234 |
| SVI constrained | 0 / 4 | 0 | +0.000 | 0 | 0 / 0 |
| SSVI | 0 / 4 | 0 | +0.254 | 0 | 0 / 0 |
<!-- END:arbitrage-table -->

<!-- BEGIN:timing-table -->
| Model | Median time, full surface (s) | Per slice (ms) | Runs |
|---|---:|---:|---:|
| SVI raw | 1.81 | 453 | 10 |
| SVI constrained | 3.44 | 859 | 10 |
| SSVI | 0.05 | 14 | 10 |
<!-- END:timing-table -->

<!-- BEGIN:stability-table -->
| Model | Std ATM vol | Std put wing k=-0.5 | Std call wing k=+0.3 | Fits at a bound |
|---|---:|---:|---:|---:|
| SVI raw | 0.018 | 0.126 | 0.716 | 0 |
| SVI constrained | 0.016 | 0.108 | 0.418 | 0 |
| SSVI | 0.075 | 0.095 | 0.062 | 0 |
<!-- END:stability-table -->

<!-- BEGIN:barrier-table -->
| Barrier | DIP local vol | DIP BS strike vol | DIP BS barrier vol | LV - BS(strike) | Coupon effect p.a. | DOC local vol | DOC BS strike vol | DOC gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 75% | 1.10% | 0.00% | 2.03% | +1.10% | +4.5% | 3.39% | 3.38% | +0.1% |
| 80% | 1.50% | 0.06% | 2.65% | +1.43% | +5.9% | 3.38% | 3.38% | -0.0% |
| 85% | 1.93% | 0.46% | 3.23% | +1.47% | +6.0% | 3.36% | 3.38% | -0.7% |
| 90% | 2.35% | 1.57% | 3.56% | +0.78% | +3.2% | 3.26% | 3.38% | -3.6% |
| 95% | 2.67% | 2.59% | 3.39% | +0.07% | +0.3% | 2.77% | 3.02% | -8.2% |
<!-- END:barrier-table -->

<!-- BEGIN:repricing-table -->
| Strike | Type | Market vol | SSVI vol | Local-vol MC implied vol | MC error (std errors) |
|---:|---|---:|---:|---:|---:|
| 80% | put | 29.16% | 29.73% | 29.71% | -0.2 |
| 90% | put | 21.88% | 23.04% | 23.07% | +0.4 |
| 100% | call | 15.42% | 15.53% | 15.58% | +2.3 |
| 110% | call | 12.41% | 13.12% | 13.15% | +1.0 |
<!-- END:repricing-table -->
