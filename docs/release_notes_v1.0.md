# v1.0 — release notes

Complete rebuild of the v0 pipeline (commit `3519491`) after the audit in `AUDIT.md`.

## Data
* Calibration on two-sided mid quotes only; chains without quotes (market closed) are rejected (`StaleQuotesError`). v0 used stale last-trade prices for 100% of its points.
* Forwards from put-call parity per expiry and Black-76 implied vols; discounting at the FRED DTB3 rate of the snapshot day (3.63% discount basis, 3.697% continuous).
* Time to expiry ACT/365 from the snapshot time to 16:00 New York; no clock, no network in the computation. Snapshot metadata (time, rate, dividends, SHA-256 of the chain) versioned in `data/snapshots/`.
* Raw Yahoo quotes are no longer meant to be redistributed (Yahoo terms); a synthetic SPY-like chain built from the real fits is bundled in `data/sample/`.

## Models
* Raw SVI with quasi-explicit starts; constrained SVI with Durrleman g(k) >= 0, Lee's bound b(1 ± ρ) <= 2 (v0 used 4/T, never active) and no calendar crossing; penalty continuation + SLSQP.
* SSVI with power-law φ, θ pinned to market ATM total variance, γ <= 1/2 and the exact Theorem 4.2 bound on η: free of static arbitrage at every maturity.
* Arbitrage audit split between quoted and extrapolated strikes; independent price-convexity check.
* New: Dupire local vol from SSVI (σ²_loc = ∂_T w / g), Monte-Carlo barrier pricer with Brownian-bridge monitoring and SPY cash dividends; Reiner-Rubinstein closed forms cross-checked against QuantLib.

## Engineering
* Package `volsurf` (src layout), pinned dependencies, Python 3.11-3.13.
* pytest suite (financial identities, counterexamples, parameter recovery, regression), coverage above 95%; ruff, mypy; GitHub Actions.
* `scripts/run_results.py` regenerates every table and figure of the README; a test checks that the README tables equal the generated ones.
