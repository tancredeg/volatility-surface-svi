"""Regenerate every table and figure of the README.

Usage (from the repository root):
    python scripts/run_results.py                      # bundled synthetic sample -> results/sample, figures/sample
    python scripts/run_results.py --snapshot data/snapshots/spy_20260603.json --update-readme
                                                       # real SPY snapshot (raw chain required) -> README

The real SPY chain is not redistributed (Yahoo Finance terms); its metadata
(timestamp, rate, dividends, SHA-256 of the chain file) is versioned in
data/snapshots/. Anyone holding a chain with the same hash reproduces the
README numbers exactly; the bundled sample reproduces the full pipeline.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

from volsurf.data import load_snapshot
from volsurf.pipeline import StudyConfig, run_study
from volsurf.plots import plot_all
from volsurf.report import inject, write_outputs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default=str(ROOT / "data/sample/spy_like_20260603.json"))
    ap.add_argument("--name", default=None, help="Output sub-folder (default: 'real' or 'sample').")
    ap.add_argument("--fast", action="store_true", help="Fewer paths and runs (smoke test).")
    ap.add_argument("--update-readme", action="store_true", help="Inject the tables into README.md.")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", message="Values in x were outside bounds")
    t0 = time.perf_counter()
    snap = load_snapshot(args.snapshot)
    name = args.name or ("sample" if snap.meta.synthetic else "real")
    cfg = StudyConfig.fast() if args.fast else StudyConfig()
    res = run_study(snap, cfg)
    out = ROOT / "results" / name
    write_outputs(res, out)
    figs = plot_all(res, ROOT / "figures" / name)
    print(
        f"Snapshot {snap.meta.underlying} {snap.meta.valuation_time:%Y-%m-%d %H:%M} UTC "
        f"({'synthetic' if snap.meta.synthetic else 'market'}), {len(snap.slices)} expiries"
    )
    print(f"Results -> {out.relative_to(ROOT)}; figures -> {[str(f.relative_to(ROOT)) for f in figs]}")
    if args.update_readme:
        if snap.meta.synthetic:
            raise SystemExit("Refusing to inject synthetic-sample numbers into the README.")
        n = inject(ROOT / "README.md", out / "tables.md")
        print(f"README: {n} blocks updated")
    print(f"Done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
