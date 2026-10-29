"""Replay the published pipeline (iv_surface.py + ssvi.py, unmodified) offline.

Purpose (Phase 0 audit): establish which input produced the figures and the
README tables, and whether the run is reproducible without network access.

Usage:
    python scripts/audit/reproduce_published_run.py REPO_DIR CSV_PATH REF_DATE SPOT OUT_DIR

Example, from the repo root (v0 code at tag v0-published; published run: 2026-06-04, spot 752.18):
    git worktree add ../v0 v0-published
    python scripts/audit/reproduce_published_run.py ../v0 data/raw/spy_chains_20260604.csv 2026-06-04 752.18 out/replay_0604

The repo code calls datetime.date.today() and a live yfinance spot. Both are
pinned here by patching the module, so that the replay is deterministic.
Nothing in the repo code is otherwise modified.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    repo, csv_path, ref_str, spot_str, out_dir = sys.argv[1:6]
    repo, csv_path, out_dir = Path(repo), Path(csv_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_date = _dt.date.fromisoformat(ref_str)
    spot = float(spot_str)

    sys.path.insert(0, str(repo))
    import iv_surface as ivs  # noqa: E402
    import ssvi  # noqa: E402

    # --- pin "today" and the output directory ------------------------------
    class _FixedDate(_dt.date):
        @classmethod
        def today(cls):
            return ref_date

    shim = types.SimpleNamespace(date=_FixedDate, datetime=_dt.datetime)
    ivs.datetime = shim
    ssvi.datetime = shim
    ivs.FIG_DIR = out_dir
    ivs.DATA_SOURCE = f"replay:{csv_path.name}"

    raw = pd.read_csv(csv_path)
    chains = {e: raw[raw["expiry"] == e].copy() for e in raw["expiry"].unique()}

    # --- per-slice SVI pipeline (iv_surface._run_pipeline, unmodified) ------
    captured: dict = {}
    orig_verify = ivs.verify_svi_surface

    def _verify(svi_results, maturities):
        captured["svi"] = svi_results
        captured["mats"] = maturities
        return orig_verify(svi_results, maturities)

    ivs.verify_svi_surface = _verify
    t0 = time.perf_counter()
    ivs._run_pipeline(chains, S0=spot, r=ivs.RISK_FREE_RATE, q=ivs.DIVIDEND_YIELD)
    t_svi_pipeline = time.perf_counter() - t0

    # --- SSVI (ssvi.main logic, unmodified functions) -----------------------
    slices = ssvi.load_and_prepare(csv_path, spot, ivs.RISK_FREE_RATE, ivs.DIVIDEND_YIELD)
    t0 = time.perf_counter()
    svi_by_t = {t: ivs.calibrate_svi(k, w, t) for t, (k, w) in slices.items()}
    t_svi = time.perf_counter() - t0
    t0 = time.perf_counter()
    fit = ssvi.calibrate_ssvi(slices)
    t_ssvi = time.perf_counter() - t0
    summ = ssvi.verify_ssvi_surface(fit)
    ssvi.plot_ssvi_vs_svi(slices, fit, svi_by_t, out_dir)

    # --- price-source accounting (what the IVs were actually computed from) -
    src_rows = []
    for e, df in chains.items():
        T = (_dt.date.fromisoformat(e) - ref_date).days / 365.0
        for ot in ("call", "put"):
            d = df[df["option_type"] == ot]
            try:
                c = ivs.clean_option_chain(d, spot, ivs.RISK_FREE_RATE, ivs.DIVIDEND_YIELD, T)
            except ValueError:
                continue
            F = spot * np.exp((ivs.RISK_FREE_RATE - ivs.DIVIDEND_YIELD) * T)
            otm = c[c["strike"] > F] if ot == "call" else c[c["strike"] <= F]
            ltd = pd.to_datetime(otm["lastTradeDate"], utc=True)
            age_days = (pd.Timestamp(ref_date, tz="UTC") - ltd).dt.total_seconds() / 86400
            src_rows.append(
                {
                    "expiry": e,
                    "type": ot,
                    "n_otm": len(otm),
                    "share_lastPrice": float((otm["price_source"] == "last").mean()),
                    "median_trade_age_days": float(age_days.median()),
                    "max_trade_age_days": float(age_days.max()),
                }
            )

    res = {
        "csv": csv_path.name,
        "ref_date": ref_str,
        "spot": spot,
        "svi": {
            str(round(t * 365)): {
                **{k: float(v) for k, v in r["params"].items()},
                "rmse": r["rmse"],
                "b_1plus_absrho": float(r["params"]["b"] * (1 + abs(r["params"]["rho"]))),
                "lee_4_over_T": 4.0 / t,
                "n_points": len(r["m_obs"]),
                "m_min": float(np.min(r["m_obs"])),
                "m_max": float(np.max(r["m_obs"])),
            }
            for t, r in captured["svi"].items()
        },
        "ssvi": {
            "rho": fit["rho"],
            "eta": fit["eta"],
            "gamma": fit["gamma"],
            "theta": {str(round(t * 365)): v for t, v in fit["theta"].items()},
            "rmse": {str(round(t * 365)): v for t, v in fit["rmse"].items()},
            "calendar_violations": summ["calendar_violations"],
        },
        "timing_s": {"svi_pipeline_total": t_svi_pipeline, "svi_4_slices": t_svi, "ssvi_joint": t_ssvi},
        "price_source": src_rows,
    }
    (out_dir / "replay_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
