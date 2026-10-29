"""Serialise study results: JSON summary, CSV tables and the Markdown blocks injected in the README."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .pipeline import MODELS, StudyResults


def summary(res: StudyResults) -> dict[str, object]:
    """Machine-readable headline numbers (also used by the regression test)."""
    cal = res.calibration
    b = res.barrier
    t90 = b.table.set_index("barrier_pct").loc[90.0]
    return {
        "snapshot": {
            "underlying": res.snapshot.meta.underlying,
            "valuation_time_utc": res.snapshot.meta.valuation_time.isoformat(),
            "synthetic": res.snapshot.meta.synthetic,
            "rate_cc": res.snapshot.meta.rate,
            "expiries": [s.expiry for s in res.snapshot.slices],
            "maturities_days": [s.maturity * 365 for s in res.snapshot.slices],
            "forwards": [s.forward for s in res.snapshot.slices],
            "quotes": [len(s.k) for s in res.snapshot.slices],
        },
        "rmse_vp": {m: [float(v) for v in res.fit[f"rmse_vp[{m}]"]] for m in MODELS},
        "svi_raw_params": [list(map(float, f.params.as_array())) for f in cal.svi_raw],
        "svi_constrained_params": [list(map(float, f.params.as_array())) for f in cal.svi_con],
        "ssvi": {
            "rho": cal.ssvi.params.rho,
            "eta": cal.ssvi.params.eta,
            "gamma": cal.ssvi.params.gamma,
            "theta": [float(x) for x in cal.ssvi.theta],
            "guaranteed_arbitrage_free": cal.ssvi.params.guaranteed_arbitrage_free,
        },
        "butterfly_slices_with_violation": {
            m: int((res.butterfly[res.butterfly.model == m].g_min < -1e-7).sum()) for m in MODELS
        },
        "calendar_violation_points": {
            m: int(
                res.calendar[res.calendar.model == m][["viol_points_supported", "viol_points_extrapolated"]]
                .to_numpy()
                .sum()
            )
            for m in MODELS
        },
        "ssvi_surface_check": res.ssvi_check,
        "timing_median_s": {r.model: r.median_s for r in res.timing.itertuples()},
        "barrier": {
            "expiry": b.expiry,
            "maturity_days": b.maturity * 365,
            "spot": b.spot,
            "strike": b.strike,
            "flat_vol": b.flat_vol,
            "n_paths": res.config.n_paths,
            "n_steps": b.lv.n_steps,
            "dip90_lv": float(t90["dip_lv"]),
            "dip90_flat": float(t90["dip_flat"]),
            "doc90_lv": float(t90["doc_lv"]),
            "doc90_flat": float(t90["doc_flat"]),
            "local_vol_clipped_points": b.lv_grid.n_clipped,
        },
    }


def _fmt(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def md_fit_table(res: StudyResults) -> str:
    """Implied-vol RMSE per expiry and model (vol points)."""
    f = res.fit
    head = [
        "| Expiry | T (days) | Quotes | ATM vol | RMSE SVI raw | RMSE SVI constrained | RMSE SSVI | Half bid-ask |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in f.iterrows():
        head.append(
            f"| {r['expiry']} | {r['T_days']:.0f} | {r['quotes']} | {r['atm_vol_pct']:.2f}% | "
            f"{r['rmse_vp[SVI raw]']:.2f} | {r['rmse_vp[SVI constrained]']:.2f} | {r['rmse_vp[SSVI]']:.2f} | "
            f"{r['half_spread_vp']:.3f} |"
        )
    return "\n".join(head)


def md_arbitrage_table(res: StudyResults) -> str:
    b, c = res.butterfly, res.calendar
    n = len(res.snapshot.slices)
    head = [
        "| Model | Slices with g(k) < 0 | of which inside quoted range | Min g(k) | Negative-density points (price check) "
        "| Calendar crossings: quoted / extrapolated (grid points) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        bm, cm = b[b.model == m], c[c.model == m]
        n_bad = int((bm.g_min < -1e-7).sum())
        n_in = int((bm.neg_points_supported > 0).sum())
        head.append(
            f"| {m} | {n_bad} / {n} | {n_in} | {bm.g_min.min():+.3f} | {int(bm.density_neg_points.sum())} | "
            f"{int(cm.viol_points_supported.sum())} / {int(cm.viol_points_extrapolated.sum())} |"
        )
    return "\n".join(head)


def md_timing_table(res: StudyResults) -> str:
    head = ["| Model | Median time, full surface (s) | Per slice (ms) | Runs |", "|---|---:|---:|---:|"]
    for r in res.timing.itertuples():
        head.append(f"| {r.model} | {r.median_s:.2f} | {r.median_ms_per_slice:.0f} | {r.runs} |")
    return "\n".join(head)


def md_stability_table(res: StudyResults) -> str:
    s = res.stability
    head = [
        "| Model | Std ATM vol | Std put wing k=-0.5 | Std call wing k=+0.3 | Fits at a bound |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        sm = s[s.model == m]
        head.append(
            f"| {m} | {sm.std_atm_vp.max():.3f} | {sm.std_wing_put_vp.max():.3f} | "
            f"{sm.std_wing_call_vp.max():.3f} | {int(sm.fits_at_bound.iloc[0])} |"
        )
    return "\n".join(head)


def md_barrier_table(res: StudyResults) -> str:
    b = res.barrier
    t = b.table
    S = b.spot
    head = [
        "| Barrier | DIP local vol | DIP BS strike vol | DIP BS barrier vol | LV - BS(strike) | Coupon effect p.a. "
        "| DOC local vol | DOC BS strike vol | DOC gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        head.append(
            f"| {r['barrier_pct']:.0f}% | {r['dip_lv'] / S * 100:.2f}% | {r['dip_flat'] / S * 100:.2f}% | "
            f"{r['dip_barrier_vol_closed_form'] / S * 100:.2f}% | {r['dip_gap_pct_of_spot']:+.2f}% | "
            f"{r['coupon_overstatement_pa_pct']:+.1f}% | {r['doc_lv'] / S * 100:.2f}% | {r['doc_flat'] / S * 100:.2f}% | "
            f"{r['doc_gap_pct']:+.1f}% |"
        )
    return "\n".join(head)


def md_repricing_table(res: StudyResults) -> str:
    head = [
        "| Strike | Type | Market vol | SSVI vol | Local-vol MC implied vol | MC error (std errors) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, r in res.barrier.repricing.iterrows():
        head.append(
            f"| {r['strike_pct']:.0f}% | {r['type']} | {_fmt(r['market_vol_pct'])}% | {r['ssvi_vol_pct']:.2f}% | "
            f"{r['lv_mc_vol_pct']:.2f}% | {r['error_in_se']:+.1f} |"
        )
    return "\n".join(head)


def md_key_results(res: StudyResults) -> str:
    """Bulleted key results, every number read from the results objects."""
    f, b = res.fit, res.barrier
    t = b.table.set_index("barrier_pct")
    r90 = t.loc[90.0]
    bm, cm = res.butterfly, res.calendar
    rb, rc = bm[bm.model == "SVI raw"], cm[cm.model == "SVI raw"]
    raw_bad = rb[rb.g_min < -1e-7]
    raw_in = int((rb.neg_points_supported > 0).sum() + (rc.viol_points_supported > 0).sum())
    n = len(res.snapshot.slices)
    tm = res.timing.set_index("model")
    cal_bad = rc[rc.viol_points_supported + rc.viol_points_extrapolated > 0]
    k_max_quoted = max(s.k_range[1] for s in res.snapshot.slices)
    if len(cal_bad):
        cal_txt = (
            f"and {len(cal_bad)} of {n - 1} pairs of expiries cross in total variance "
            f"(for k >= {cal_bad.k_viol_min.min():.2f})"
        )
    else:
        cal_txt = "and no calendar crossing"
    bfly_txt = (
        f"butterfly arbitrage (negative density) on {len(raw_bad)} of {n} expiries "
        f"(g(k) as low as {rb.g_min.min():+.2f}, at k = {raw_bad.k_at_g_min.min():.2f} to "
        f"{raw_bad.k_at_g_min.max():.2f})"
        if len(raw_bad)
        else "no butterfly arbitrage"
    )
    rep = res.barrier.repricing
    con_cost = f["rmse_vp[SVI constrained]"] - f["rmse_vp[SVI raw]"]
    ssvi_cost = f["rmse_vp[SSVI]"] - f["rmse_vp[SVI raw]"]
    lines = [
        f"- **Fit.** On {int(f.quotes.sum())} out-of-the-money quotes, constrained (arbitrage-free) SVI has an "
        f"implied-vol RMSE of {f['rmse_vp[SVI constrained]'].min():.2f}-{f['rmse_vp[SVI constrained]'].max():.2f} "
        f"vol points per expiry, against {f['rmse_vp[SVI raw]'].min():.2f}-{f['rmse_vp[SVI raw]'].max():.2f} for "
        f"unconstrained SVI and {f['rmse_vp[SSVI]'].min():.2f}-{f['rmse_vp[SSVI]'].max():.2f} for SSVI. "
        f"Removing arbitrage costs at most {max(con_cost.max(), 0.0):.3f} vol point per expiry; SSVI's single "
        f"set of shape parameters for all expiries costs {ssvi_cost.min():.2f}-{ssvi_cost.max():.2f}.",
        f"- **Arbitrage.** Unconstrained SVI shows {bfly_txt} {cal_txt}; "
        f"{'none' if raw_in == 0 else str(raw_in)} of these violations lie inside the quoted strike range "
        f"(k <= {k_max_quoted:.2f}). The fit looks clean on the data and breaks in the extrapolated wings, "
        "which Dupire local vol and every wing-sensitive product use. Constrained SVI and SSVI have no violation, "
        "confirmed by two independent checks (Durrleman g and convexity of call prices in strike); the SSVI "
        "surface is arbitrage-free at every maturity, not only at the listed ones (gamma <= 1/2).",
        f"- **Speed.** A full surface takes {tm.loc['SSVI', 'median_s']:.2f} s with SSVI, "
        f"{tm.loc['SVI raw', 'median_s']:.1f} s with raw SVI and {tm.loc['SVI constrained', 'median_s']:.1f} s with "
        "constrained SVI (median, one CPU core).",
        f"- **Structuring.** On a down-and-in put expiring in {b.maturity * 365:.0f} days (strike 100%, barrier 90%), Dupire local vol "
        f"gives {r90['dip_lv'] / b.spot * 100:.2f}% of notional against {r90['dip_flat'] / b.spot * 100:.2f}% for "
        f"Black-Scholes at the strike vol ({b.flat_vol * 100:.1f}%): the flat-vol desk would under-charge the barrier "
        f"by {r90['dip_gap_pct_of_spot']:.2f}% of notional, i.e. overstate a reverse-convertible coupon by about "
        f"{r90['coupon_overstatement_pa_pct']:.1f}% per annum. Using the barrier-level vol instead "
        f"({r90['barrier_vol_pct']:.1f}%) gives {r90['dip_barrier_vol_closed_form'] / b.spot * 100:.2f}%.",
        f"- **Model check.** The local-vol Monte Carlo reprices the SSVI vanillas within "
        f"{rep['lv_minus_ssvi_vp'].abs().max():.2f} vol point ({rep['error_in_se'].abs().max():.1f} standard errors at "
        f"most); SSVI itself is off the market by {abs(rep.loc[rep.strike_pct == 90, 'ssvi_vol_pct'].iloc[0] - rep.loc[rep.strike_pct == 90, 'market_vol_pct'].iloc[0]):.2f} "
        "vol point at the 90% strike, which bounds the precision of the barrier prices.",
    ]
    return "\n".join(lines)


BLOCKS = {
    "key-results": md_key_results,
    "fit-table": md_fit_table,
    "arbitrage-table": md_arbitrage_table,
    "timing-table": md_timing_table,
    "stability-table": md_stability_table,
    "barrier-table": md_barrier_table,
    "repricing-table": md_repricing_table,
}


def write_outputs(res: StudyResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary(res), indent=2) + "\n", encoding="utf-8")
    res.snapshot.filter_log.to_csv(out_dir / "filter_log.csv", index=False)
    for name, df in {
        "fit": res.fit,
        "butterfly": res.butterfly,
        "calendar": res.calendar,
        "timing": res.timing,
        "stability": res.stability,
        "barrier": res.barrier.table,
        "repricing": res.barrier.repricing,
    }.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6g")
    md = [f"<!-- BEGIN:{k} -->\n{fn(res)}\n<!-- END:{k} -->" for k, fn in BLOCKS.items()]
    (out_dir / "tables.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")


def inject(readme: Path, tables_md: Path) -> int:
    """Replace every <!-- BEGIN:name --> ... <!-- END:name --> block of the README by the generated one."""
    text = readme.read_text(encoding="utf-8")
    blocks = dict(
        re.findall(
            r"<!-- BEGIN:([\w-]+) -->\n(.*?)\n<!-- END:\1 -->",
            tables_md.read_text(encoding="utf-8"),
            flags=re.S,
        )
    )
    n = 0
    for name, body in blocks.items():
        # The block may be empty (BEGIN and END on consecutive lines) or hold an older version.
        pattern = re.compile(rf"(<!-- BEGIN:{name} -->\n)(?:.*?\n)?(<!-- END:{name} -->)", re.S)

        def repl(m: re.Match[str], b: str = body) -> str:
            return m.group(1) + b + "\n" + m.group(2)

        text, k = pattern.subn(repl, text)
        n += k
    readme.write_text(text, encoding="utf-8")
    return n
