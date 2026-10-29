"""Shared fixtures: the bundled synthetic sample and its calibration (computed once per session)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from volsurf.data import PreparedSnapshot, load_snapshot
from volsurf.pipeline import Calibration, calibrate_all

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_META = ROOT / "data" / "sample" / "spy_like_20260603.json"


@pytest.fixture(autouse=True)
def _quiet_slsqp() -> None:
    warnings.filterwarnings("ignore", message="Values in x were outside bounds")


@pytest.fixture(scope="session")
def sample() -> PreparedSnapshot:
    return load_snapshot(SAMPLE_META)


@pytest.fixture(scope="session")
def sample_cal(sample: PreparedSnapshot) -> Calibration:
    warnings.filterwarnings("ignore", message="Values in x were outside bounds")
    return calibrate_all(sample)
