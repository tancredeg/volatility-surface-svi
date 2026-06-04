# Implied Volatility Surface — SVI Calibration Pipeline

End-to-end pipeline from raw SPY option chain data to a calibrated, arbitrage-verified implied volatility surface using the Gatheral (2004) SVI parametrisation.

---

## Repository Structure

```
iv-surface-svi/
├── iv_surface.py          # Per-slice SVI pipeline (clean → IV → SVI → arbitrage)
├── ssvi.py                # Surface SVI: globally arbitrage-free calibration
├── fetch_data.py          # Standalone REAL-data extractor (run in VS Code)
├── requirements.txt       # Pinned dependencies
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── spy_chains_YYYY-MM-DD.csv   # Written by fetch_data.py (carries spot)
└── figures/
    ├── fig0_raw_smiles.png          # Raw IV smiles before SVI fit
    ├── fig1_svi_smile_fits.png      # Per-slice SVI fit per maturity
    ├── fig2_svi_surface_3d.png      # 3D SVI surface iv(m, T)
    ├── fig3_atm_term_structure.png  # ATM term structure: model vs raw
    ├── fig4_skew_analysis.png       # Approximate RR and butterfly vs maturity
    ├── fig5_ssvi_vs_svi.png         # SSVI vs per-slice SVI per maturity
    └── fig6_ssvi_surface_3d.png     # 3D SSVI surface (arbitrage-free)
```

---

## Installation and Usage

```bash
git clone https://github.com/<your-handle>/iv-surface-svi
cd iv-surface-svi
pip install -r requirements.txt

# Step 1 — fetch REAL market data (writes data/spy_chains_<date>.csv)
python fetch_data.py

# Step 2 — per-slice SVI calibration + arbitrage checks + figures
python iv_surface.py

# Step 3 — Surface SVI (globally arbitrage-free) + comparison to per-slice SVI
python ssvi.py
```

`fetch_data.py` downloads live SPY option chains, runs realness checks (shared
call/put strikes, plausible strike range, open interest present, no synthetic
labels), records the underlying spot into the CSV, and writes a dated file to
`data/`. It never fabricates data: on any failure it exits with a diagnostic.

`iv_surface.py` runs the per-slice pipeline. It attempts a live download; if
that fails it loads the most recent CSV in `data/`. Spot is resolved live, then
from the CSV's stored spot — never silently defaulted, because a wrong spot
corrupts the forward and the whole surface. It **refuses** any cached file with
`synthetic_*` labels.

`ssvi.py` calibrates the Surface SVI parametrisation jointly across maturities,
verifies the surface is calendar- and butterfly-arbitrage-free, and prints an
RMSE comparison against the per-slice SVI baseline.

**Synthetic self-test (opt-in only, never the default):**
```python
from iv_surface import run_synthetic_pipeline
run_synthetic_pipeline()   # tests the calibrator against known SVI params
```
The synthetic path exists solely to verify that calibration recovers known
parameters. Its output is labelled `DATA_SOURCE = synthetic_with_noise` and it
never writes to `data/`.

---

## Mathematical Background

### 1. Log-Moneyness

The smile is parametrised in **log-moneyness**:

```
m = ln(K / F),   where F = S * exp((r - q) * T)
```

- `S` is the spot price, `K` the strike, `T` time to expiry in years.
- `F` is the cost-of-carry forward (continuous dividend yield `q`, risk-free rate `r`).
- Log-moneyness is preferred over simple moneyness `K/S - 1` because it is symmetric around zero, dimensionless, and consistent with the log-normal distribution underlying Black-Scholes. ATM corresponds to `m = 0` exactly.

### 2. SVI Parametrisation

Gatheral (2004) introduced the Stochastic Volatility Inspired (SVI) parametrisation of **total variance** `w = σ²T`:

```
w(m) = a + b * [ ρ(m - μ) + sqrt((m - μ)² + ξ²) ]
```

| Parameter | Domain        | Interpretation                                         |
|-----------|---------------|--------------------------------------------------------|
| `a`       | (-0.5, 2.0)   | Overall variance level. May be slightly negative at short maturities due to microstructure. |
| `b`       | (0, 2.0)      | Slope / curvature. Controls the speed of the wings.    |
| `ρ`       | (-1, 1)       | Skew correlation. Negative for equities (put premium). |
| `μ`       | (-0.5, 0.5)   | Shift of the minimum. Near zero for index options.     |
| `ξ`       | (0, 2.0)      | ATM smoothness. Larger values produce a flatter ATM region. |

The implied volatility is recovered as:

```
σ(m, T) = sqrt(w(m) / T)
```

SVI is used in practice because:
- It fits observed smiles accurately with only five parameters.
- It has a closed-form expression, making calibration fast.
- The no-arbitrage conditions are tractable to check and enforce.
- It generalises to a full surface via the SSVI extension (Gatheral & Jacquier 2014).

### 3. Black-Scholes IV Extraction

Implied volatility is extracted by inverting the Black-Scholes formula (Merton 1973 extension with continuous dividends):

```
d1 = [ln(S/K) + (r - q + σ²/2) * T] / (σ * sqrt(T))
d2 = d1 - σ * sqrt(T)

Call = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
Put  = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)
Vega = S * exp(-qT) * φ(d1) * sqrt(T)         (identical for call and put)
```

**Algorithm (Newton-Raphson with Brent fallback):**

1. Initialise `σ₀ = 0.20`.
2. Newton-Raphson step: `σ_{n+1} = σ_n - (BS(σ_n) - price) / Vega(σ_n)`
3. If `Vega(σ_n) < 1e-10` (deep OTM), switch immediately to Brent's method.
   - Brent is slower but guaranteed convergent on a bracketed interval.
4. Brent interval: `[1e-4, 10.0]`. If no sign change exists (price not bracketed), return `NaN` with an explicit log warning — never a silent `NaN`.

NR diverges near the wings because vega approaches zero; the fallback to Brent prevents this failure mode.

### 4. Butterfly No-Arbitrage Conditions

**Lee (2004) — necessary condition:**

For no butterfly arbitrage, the SVI parameters must satisfy:

```
b * (1 + |ρ|) < 4 / T
```

This is a necessary but not sufficient condition. It bounds the maximum slope of the smile wings.

**Durrleman butterfly condition (necessary and sufficient):**

A single SVI slice is free of butterfly arbitrage — equivalently, its implied
risk-neutral density is non-negative — if and only if the Durrleman function
`g(m) ≥ 0` for all `m`:

```
g(m) = (1 - m·w'(m)/(2·w(m)))²
       - (w'(m)²/4)·(1/w(m) + 1/4)
       + w''(m)/2
```

with analytic SVI derivatives:

```
w'(m)  = b·[ρ + (m-μ)/√((m-μ)²+ξ²)]
w''(m) = b·ξ² / ((m-μ)²+ξ²)^{3/2}
```

The Lee condition is a fast necessary screen; the Durrleman `g(m) ≥ 0` is the
complete characterisation and is what the pipeline reports in Section 5.

> **Note on a corrected formula.** An earlier version of this project used
> `g(m) = (1 - m·ρ/ξ_eff)² + (ρ²-1)·b²/ξ_eff²` and labelled it the "Roper
> sufficient condition". That expression is **not** the Durrleman/Gatheral
> butterfly function: tested against arbitrage-free calibrated surfaces it
> returned `g < 0` (false arbitrage alarms) on 3 of 4 maturities, while the
> correct Durrleman `g` was strictly positive everywhere. The formula above
> is the corrected, standard condition (Gatheral & Jacquier 2014, eq. 2.2–2.3).

Reference: Roper, M. (2010). *Arbitrage-Free Implied Volatility Surfaces*. Preprint.

**Calendar no-arbitrage:**

For fixed `m`, total variance must be non-decreasing in `T`:

```
w(m, T1) ≤ w(m, T2)   for T1 < T2
```

Violations on the raw data are expected due to noise and bid-ask spreads; violations on the calibrated per-slice SVI surface indicate that linear parameter interpolation is insufficient and that SSVI is needed.

### 5. Surface SVI (SSVI)

SSVI parametrises the entire surface rather than each slice independently:

```
w(k, θ) = (θ/2) · [ 1 + ρ·φ(θ)·k + √((φ(θ)·k + ρ)² + 1 − ρ²) ]
φ(θ)    = η / ( θ^γ · (1+θ)^(1−γ) )        (power-law form)
```

where `θ = θ_t` is the ATM total variance at maturity `t` (the backbone), and
`(ρ, η, γ)` are three global parameters shared by all maturities. `ssvi.py`
calibrates `(ρ, η, γ, θ_1, …, θ_n)` jointly by least squares on implied vol,
subject to the Gatheral–Jacquier (2014) no-arbitrage conditions enforced as
hard constraints:

- **Calendar:** `θ_t` non-decreasing in `t`.
- **Butterfly:** `θ·φ(θ)·(1+|ρ|) < 4` and `θ·φ(θ)²·(1+|ρ|) ≤ 4` for all `θ`.

Each SSVI slice maps to an equivalent raw-SVI slice
(`b = θφ/2, m = −ρ/φ, ξ = √(1−ρ²)/φ, a = (θ/2)(1−ρ²)`), so the same Durrleman
`g(m)` check validates butterfly no-arbitrage on the calibrated surface.

---

## Key Results

Results below are indicative. Actual values depend on market conditions at time of download.

### ATM Term Structure (model)

| Maturity | ATM Vol (model) | ATM Vol (raw) | Difference |
|----------|-----------------|---------------|------------|
| ~30d     | —               | —             | —          |
| ~60d     | —               | —             | —          |
| ~90d     | —               | —             | —          |
| ~180d    | —               | —             | —          |

*Populated at runtime. ATM vol is read from the SVI model at `m=0`, not from raw data interpolation. The two can differ by 0.5–1% on short maturities.*

### SVI Calibration Quality

| Maturity | RMSE (vol pts) | Butterfly (Lee) | g_min (Roper) |
|----------|----------------|-----------------|---------------|
| ~30d     | —              | —               | —             |
| ~60d     | —              | —               | —             |
| ~90d     | —              | —               | —             |
| ~180d    | —              | —               | —             |

*PASS threshold: RMSE < 50bp (0.005 in vol units).*

### Approximate Skew Metrics

These use **fixed moneyness `m = ±0.10`** — not true 25-delta strikes. True 25Δ RR requires solving `N(d₂(K)) = 0.25` for `K`; a desk implementation would use the delta-strike mapping.

| Maturity | Approx RR (m=±0.10) | Approx BF (m=±0.10) |
|----------|---------------------|---------------------|
| ~30d     | —                   | —                   |
| ~60d     | —                   | —                   |
| ~90d     | —                   | —                   |
| ~180d    | —                   | —                   |

---

## Known Limitations

### 1. SVI Parameter Interpolation vs SSVI

The per-slice pipeline (`iv_surface.py`) fits each maturity independently and
interpolates the five SVI parameters linearly across maturities:

```
p*(T*) = p1 + (T* - T1) / (T2 - T1) * (p2 - p1)
```

This does not guarantee a calendar-arbitrage-free surface, and in practice it
produces calendar violations in the wings (reported in Section 5). **This is
why `ssvi.py` is included:** the Surface SVI parametrisation (Gatheral &
Jacquier 2014) anchors the whole surface on the ATM total-variance term
structure with a single correlation, and is free of calendar and butterfly
arbitrage by construction. The trade-off is documented below: SSVI fits each
slice slightly worse than per-slice SVI (a single global `rho` cannot match
four individually-steeper skews) in exchange for global no-arbitrage.

### 2. Approximate RR/BF vs True 25Δ

Risk reversal and butterfly are computed at fixed moneyness `m = ±0.10`. A proper vol desk uses the 25-delta strike:

```
K_25Δ: solve N(d₂(K)) = 0.25   (call 25Δ)
       solve N(-d₂(K)) = 0.25  (put 25Δ)
```

This requires numerical root-finding per maturity and is not implemented.

### 3. Flat Rate and Dividend Curves

The pipeline uses scalar `r` and `q` for all maturities. A production implementation uses:
- OIS discount curve for `r` (bootstrapped from swap rates).
- Dividend term structure for `q` (bootstrapped from listed forwards or dividend swaps).

The error from flat curves is small for short maturities but material for T > 1y.

### 4. European Black-Scholes for American SPY Options

SPY options are American. The pipeline uses European Black-Scholes to extract IV. The early exercise premium is negligible near-the-money but non-negligible deep ITM. A desk would use a binomial tree or Barone-Adesi–Whaley approximation for American options.

### 5. Synthetic Self-Test (not a data source)

The synthetic path uses fixed SVI parameters with additive Gaussian noise to
verify that calibration recovers known parameters. It is not a data source and
never writes to `data/`. Two caveats: (a) the chosen parameters imply ATM vols
around 30–60% (much higher than real SPY ~12–20%) because the synthetic `a`/`b`
were illustrative, not calibrated to a realistic level — this is irrelevant
since real data replaces them; (b) the synthetic generator assigns each strike
to exactly one option type, so the call-put parity test is never exercised by
synthetic data. On real chains, where strikes carry both calls and puts, parity
runs normally.

---

## Interview Q&A

**Q1: What is implied volatility and how do you extract it?**

Implied volatility (IV) is the value of `σ` that, when substituted into the Black-Scholes formula, reproduces the observed market price of an option. It is the market's risk-neutral expectation of future realised volatility over the life of the option, expressed as an annualised standard deviation. Extraction inverts the BS formula numerically: Newton-Raphson for near-the-money strikes (fast convergence where vega is large), falling back to Brent's method for deep OTM strikes where vega approaches zero and NR diverges.

**Q2: What is the SVI model and why is it used in practice?**

SVI (Stochastic Volatility Inspired, Gatheral 2004) parametrises total variance `w(m) = a + b[ρ(m-μ) + √((m-μ)²+ξ²)]` in log-moneyness space. It is used because:
1. Five parameters fit observed smiles with RMSE typically under 50bp.
2. The formula is analytic — calibration via SLSQP is fast (milliseconds per maturity).
3. No-arbitrage conditions are explicit and enforceable.
4. It has the correct asymptotic behaviour: linear wings consistent with Roger Lee's moment formula.
5. The SSVI extension provides a full arbitrage-free surface.

**Q3: What is butterfly no-arbitrage and how do you verify it?**

Butterfly no-arbitrage requires that the implied risk-neutral density `q(K) ≥ 0` everywhere. In terms of the smile, this means the second derivative of the call price with respect to strike is non-negative: `∂²C/∂K² ≥ 0`. For SVI, the Lee (2004) necessary condition `b(1+|ρ|) < 4/T` provides a quick check. The Roper (2010) sufficient condition `g(m) ≥ 0` for all `m` (where `g` involves the first and second derivatives of `w`) provides a complete characterisation. This pipeline enforces the Lee condition as a hard constraint in the SLSQP optimiser and reports both conditions in the arbitrage verification section.

**Q4: Why does your smile show negative skew for SPY?**

The negative skew (implied vol increasing as moneyness decreases, i.e. for puts below the forward) reflects the **equity volatility skew**: out-of-the-money puts are more expensive relative to Black-Scholes than OTM calls, for three reasons:
1. **Demand for portfolio protection**: institutions systematically buy OTM puts to hedge equity exposure.
2. **Leverage effect**: falling equity prices are associated with rising volatility (Black 1976), making put payoffs positively correlated with volatility — puts are a natural hedge for volatility sellers, reducing their required risk premium.
3. **Jump risk**: the risk-neutral distribution has a fat left tail (crash risk), which is priced into OTM puts.

In the SVI model, negative skew is captured by `ρ < 0` (typically -0.5 to -0.8 for SPY). The steepness of the skew is proportional to `b|ρ|`.

**Q5: What would you do differently with more time?**

SSVI is already implemented (`ssvi.py`) — it replaces the linear SVI-parameter
interpolation with a globally calendar- and butterfly-arbitrage-free surface.
Remaining work, in priority order:

1. **eSSVI**: extend SSVI to a maturity-dependent correlation `ρ(θ)`
   (Hendriks & Martini 2019), recovering per-slice fit quality while remaining
   arbitrage-free. This addresses SSVI's main weakness — a single global `ρ`
   under-fits steep short-dated skews.

2. **True delta strikes**: replace the fixed-moneyness RR/BF approximation with
   proper 25-delta and 10-delta strikes, solving `N(d₂(K)) = Δ` numerically.
   This is what a desk actually trades.

3. **Dupire local volatility**: extract the local vol surface from the
   calibrated SSVI surface for exotic pricing.

4. **Heston calibration**: calibrate the Heston (1993) stochastic volatility model `dσ² = κ(θ-σ²)dt + ν√σ²dW` to the surface. Heston gives more stable model dynamics than SVI and is standard for path-dependent exotics.

5. **Rate and dividend curves**: replace flat `r` and `q` with bootstrapped OIS and dividend term structures.

---

## References

- Black, F. (1976). *Studies of stock price volatility changes.* Proceedings of the 1976 American Statistical Association.
- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parametrization with application to the valuation of volatility derivatives.* Presentation at Global Derivatives & Risk Management, Madrid.
- Gatheral, J. & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance, 14(1), 59–71.
- Heston, S. (1993). *A closed-form solution for options with stochastic volatility.* Review of Financial Studies, 6(2), 327–343.
- Jaeckel, P. (2006). *By Implication.* Wilmott Magazine, November 2006.
- Lee, R. (2004). *The moment formula for implied volatility at extreme strikes.* Mathematical Finance, 14(3), 469–480.
- Merton, R. (1973). *Theory of rational option pricing.* Bell Journal of Economics, 4(1), 141–183.
- Roper, M. (2010). *Arbitrage-free implied volatility surfaces.* Preprint, University of Sydney.

---

## License

MIT
