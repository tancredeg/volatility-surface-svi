# v1.0.1 — release notes

Maintenance release after a full review of v1.0. The published fits, SSVI parameters, arbitrage counts and barrier prices are unchanged.

## Results
* Coupon effect of the barrier study discounted to maturity: DIP gap / (D(T) T) instead of / T (90% barrier: 3.19% → 3.22% per annum, still "about 3.2%").
* Stability table: SSVI fits at a parameter bound are now counted (the 0 previously shown was a default, it is now a measurement).
* Timings regenerated on the release machine.

## Engineering
* `constraints.txt` pins every dependency, transitive ones included; the CI installs with `pip install -c constraints.txt -e ".[dev,crosscheck]"`.
* `run_results.py --fast` writes to `out/fast/` and never overwrites the versioned results; `fetch_data.py` writes git-ignored `*.draft.json` metadata skeletons.
* SVI Jacobian derived from `SVIParams.w` (no duplicated formula); explicit Brownian-bridge flag in the barrier study; typing and docstring fixes.
* Tests (77): Black-76 against QuantLib and its limits, crossed and locked quotes, SVI Jacobian, SSVI bounds, README calendar sentence; the real-snapshot regression now also covers η, the arbitrage counts and the 90% barrier prices; tests no longer write data files into the repository.
* README: no hand-written numbers outside the generated blocks; PowerShell-safe install commands; caveat on raw SVI parameter identifiability.
