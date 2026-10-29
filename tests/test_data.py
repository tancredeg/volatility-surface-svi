"""Snapshot metadata, time convention and the documented cleaning filters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsurf.data import (
    FilterConfig,
    PreparedSnapshot,
    SnapshotMeta,
    StaleQuotesError,
    load_chain,
    load_snapshot,
    prepare_snapshot,
    year_fraction,
)

from .conftest import SAMPLE_META


def test_year_fraction_uses_16h_new_york_and_dst() -> None:
    t0 = pd.Timestamp("2026-06-03T14:35:53Z")
    # 2026-07-02 16:00 EDT = 20:00 UTC: 29 days, 5 h 24 min 7 s
    expected = (29 * 86400 + 5 * 3600 + 24 * 60 + 7) / (365 * 86400)
    assert year_fraction("2026-07-02", t0) == pytest.approx(expected, rel=1e-12)
    # 2026-11-30 is after the switch to EST (UTC-5): 16:00 EST = 21:00 UTC
    assert year_fraction("2026-11-30", t0) * 365 * 86400 == pytest.approx(
        (pd.Timestamp("2026-11-30T21:00:00Z") - t0).total_seconds()
    )


def test_sample_slices_are_otm_and_filter_counts_decrease(sample: PreparedSnapshot) -> None:
    for s in sample.slices:
        assert np.all(s.strike[s.is_call] > s.forward)
        assert np.all(s.strike[~s.is_call] <= s.forward)
        assert np.all(np.diff(s.k) >= 0)
        assert np.all(s.iv_bid <= s.iv_mid) and np.all(s.iv_mid <= s.iv_ask)
    steps = ["raw", "two_sided", "otm", "min_bid", "iv_exists", "delta", "vol_spread", "final"]
    counts = sample.filter_log[steps].to_numpy(dtype=float)
    assert np.all(np.diff(counts, axis=1) <= 0)


def test_stale_chain_is_rejected(sample: PreparedSnapshot) -> None:
    chain = load_chain(sample.meta.chain_file)
    chain[["bid", "ask"]] = 0.0
    with pytest.raises(StaleQuotesError):
        prepare_snapshot(chain, sample.meta)


def test_min_maturity_filter(sample: PreparedSnapshot) -> None:
    chain = load_chain(sample.meta.chain_file)
    snap = prepare_snapshot(chain, sample.meta, FilterConfig(min_days=40.0))
    assert len(snap.slices) == len(sample.slices) - 1
    assert snap.filter_log.loc[0, "dropped"] == "maturity"


def test_hash_mismatch_and_missing_file(tmp_path: Path) -> None:
    meta = json.loads(SAMPLE_META.read_text(encoding="utf-8"))
    meta["chain_sha256"] = "0" * 64
    root = SAMPLE_META.parents[2]
    bad = root / "data" / "sample" / "_tmp_bad_meta.json"
    try:
        bad.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="SHA-256"):
            load_snapshot(bad)
        meta["chain_file"] = "data/sample/does_not_exist.csv"
        bad.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            load_snapshot(bad)
        meta["rate"]["rate_cc"] = None  # skeleton written by scripts/fetch_data.py, not filled in
        bad.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="rate"):
            load_snapshot(bad)
    finally:
        bad.unlink(missing_ok=True)


def test_missing_columns_raise(tmp_path: Path) -> None:
    p = tmp_path / "chain.csv"
    pd.DataFrame({"expiry": ["2026-07-02"], "strike": [700.0]}).to_csv(p, index=False)
    with pytest.raises(KeyError):
        load_chain(p)


def test_metadata_fields(sample: PreparedSnapshot) -> None:
    m = SnapshotMeta.from_json(SAMPLE_META)
    assert m.synthetic and m.rate == pytest.approx(0.036974)
    divs = m.cash_dividends()
    assert divs[0].amount == pytest.approx(1.903516) and 0 < divs[0].time < divs[1].time
