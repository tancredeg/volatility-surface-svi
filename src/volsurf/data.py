"""Option-chain loading, snapshot metadata and documented cleaning filters.

A snapshot is a JSON metadata file pointing to a CSV option chain. The
metadata fixes everything that the published code used to read from the
system clock or the network: valuation time, discount rate and dividends.

Cleaning pipeline (applied per expiry, counts logged at each step):
    1. maturity >= ``min_days`` calendar days;
    2. two-sided quote: bid > 0 and ask > bid (no last-trade prices, ever);
    3. forward from put-call parity on near-the-money strikes;
    4. out-of-the-money selection: calls K > F, puts K <= F;
    5. bid >= ``min_bid`` (removes minimum-tick quotes);
    6. Black-76 implied vol exists for bid, mid and ask;
    7. moneyness filter in delta: |forward delta| >= ``min_abs_delta``;
    8. vol spread (iv_ask - iv_bid) <= ``max_vol_spread``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .forward import CashDividend, ForwardEstimate, discount_factor, parity_forward
from .iv import FloatArray, forward_delta, implied_vol

BoolArray = NDArray[np.bool_]

REQUIRED_COLUMNS = ("expiry", "option_type", "strike", "bid", "ask")
EXPIRY_CUTOFF = "16:00"  # SPY options settle on the close (PM-settled)
EXPIRY_TZ = "America/New_York"


class StaleQuotesError(ValueError):
    """Raised when a chain does not carry enough two-sided quotes to be priced."""


@dataclass(frozen=True)
class Dividend:
    ex_date: date
    amount: float
    source: str = ""


@dataclass(frozen=True)
class SnapshotMeta:
    """Everything needed to price a frozen chain without clock or network."""

    underlying: str
    chain_file: Path
    valuation_time: pd.Timestamp
    rate: float
    dividends: tuple[Dividend, ...]
    description: str = ""
    synthetic: bool = False
    chain_sha256: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> SnapshotMeta:
        """Load metadata; ``chain_file`` is resolved relative to the JSON file's repo root."""
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        root = _repo_root(path)
        if raw.get("rate", {}).get("rate_cc") is None:
            raise ValueError(
                f"{path}: 'rate.rate_cc' is missing; fill in the discount rate of the snapshot day."
            )
        divs = tuple(
            Dividend(date.fromisoformat(d["ex_date"]), float(d["amount"]), d.get("source", ""))
            for d in raw.get("dividends", [])
        )
        return cls(
            underlying=raw["underlying"],
            chain_file=(root / raw["chain_file"]).resolve(),
            valuation_time=pd.Timestamp(raw["valuation_time_utc"]).tz_convert("UTC"),
            rate=float(raw["rate"]["rate_cc"]),
            dividends=divs,
            description=raw.get("description", ""),
            synthetic=bool(raw.get("synthetic", False)),
            chain_sha256=raw.get("chain_sha256"),
            extra={k: v for k, v in raw.items() if k not in _META_KEYS},
        )

    def cash_dividends(self) -> tuple[CashDividend, ...]:
        """Dividends as (year fraction from valuation time, amount), ex-date at the open."""
        out = []
        for d in self.dividends:
            t_ex = pd.Timestamp(f"{d.ex_date.isoformat()} 09:30", tz=EXPIRY_TZ).tz_convert("UTC")
            out.append(CashDividend(_years(t_ex - self.valuation_time), d.amount))
        return tuple(out)


_META_KEYS = {
    "underlying",
    "chain_file",
    "valuation_time_utc",
    "rate",
    "dividends",
    "description",
    "synthetic",
    "chain_sha256",
}


def _repo_root(path: Path) -> Path:
    for parent in path.resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return path.resolve().parent


def _years(delta: pd.Timedelta) -> float:
    return float(delta.total_seconds() / (365.0 * 86400.0))


def sha256_of(path: str | Path) -> str:
    """SHA-256 of a file, used to pin the exact chain a result was computed on."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def year_fraction(expiry: str | date, valuation_time: pd.Timestamp) -> float:
    """ACT/365 year fraction from the valuation time to 16:00 New York on the expiry date."""
    exp = expiry.isoformat() if isinstance(expiry, date) else str(expiry)
    t_exp = pd.Timestamp(f"{exp} {EXPIRY_CUTOFF}", tz=EXPIRY_TZ).tz_convert("UTC")
    return _years(t_exp - valuation_time)


def load_chain(path: str | Path) -> pd.DataFrame:
    """Read a CSV option chain and validate the columns the pipeline uses.

    Raises:
        KeyError: if a required column is missing.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Chain {path} lacks columns {missing}; has {list(df.columns)}.")
    df = df.copy()
    df["option_type"] = df["option_type"].str.lower().str.strip()
    df["expiry"] = df["expiry"].astype(str)
    for c in ("strike", "bid", "ask"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def two_sided_share(chain: pd.DataFrame) -> float:
    """Share of rows with a usable two-sided quote (bid > 0 and ask > bid)."""
    ok = (chain["bid"] > 0) & (chain["ask"] > chain["bid"])
    return float(ok.mean()) if len(chain) else 0.0


@dataclass(frozen=True)
class FilterConfig:
    """Cleaning thresholds (see module docstring for the order of application)."""

    min_days: float = 7.0
    min_bid: float = 0.05
    min_abs_delta: float = 0.005
    max_vol_spread: float = 0.01
    forward_band: float = 0.03
    min_quote_share: float = 0.5
    min_points: int = 8


@dataclass(frozen=True)
class Slice:
    """Cleaned out-of-the-money smile of one expiry, sorted by log-moneyness."""

    expiry: str
    maturity: float
    forward: float
    discount: float
    forward_estimate: ForwardEstimate
    strike: FloatArray
    k: FloatArray
    is_call: BoolArray
    mid: FloatArray
    iv_mid: FloatArray
    iv_bid: FloatArray
    iv_ask: FloatArray

    @property
    def total_variance(self) -> FloatArray:
        return np.asarray(self.iv_mid**2 * self.maturity)

    @property
    def k_range(self) -> tuple[float, float]:
        return float(self.k.min()), float(self.k.max())


@dataclass(frozen=True)
class PreparedSnapshot:
    meta: SnapshotMeta
    slices: tuple[Slice, ...]
    filter_log: pd.DataFrame

    @property
    def maturities(self) -> FloatArray:
        return np.array([s.maturity for s in self.slices])


def prepare_snapshot(
    chain: pd.DataFrame, meta: SnapshotMeta, config: FilterConfig | None = None
) -> PreparedSnapshot:
    """Apply the documented cleaning pipeline and return one smile per expiry.

    Raises:
        StaleQuotesError: if fewer than ``config.min_quote_share`` of the rows
            carry a two-sided quote (e.g. a chain downloaded with the market
            closed). Last-trade prices are never used as a substitute.
    """
    cfg = config or FilterConfig()
    share = two_sided_share(chain)
    if share < cfg.min_quote_share:
        raise StaleQuotesError(
            f"Only {share:.1%} of rows carry a two-sided quote (threshold "
            f"{cfg.min_quote_share:.0%}). The chain was probably captured with the market "
            "closed; last-trade prices are asynchronous and are not used."
        )
    slices: list[Slice] = []
    log_rows: list[dict[str, object]] = []
    for expiry in sorted(chain["expiry"].unique()):
        sub = chain[chain["expiry"] == expiry]
        T = year_fraction(expiry, meta.valuation_time)
        row: dict[str, object] = {"expiry": expiry, "T_days": T * 365.0, "raw": len(sub)}
        if cfg.min_days > T * 365.0:
            row["dropped"] = "maturity"
            log_rows.append(row)
            continue
        q = sub[(sub["bid"] > 0) & (sub["ask"] > sub["bid"])]
        row["two_sided"] = len(q)
        D = float(discount_factor(meta.rate, T))
        try:
            fwd = parity_forward(q, D, band=cfg.forward_band)
        except ValueError:
            row["dropped"] = "forward"
            log_rows.append(row)
            continue
        F = fwd.forward
        is_call = (q["option_type"] == "call").to_numpy()
        K = q["strike"].to_numpy(dtype=np.float64)
        otm = np.where(is_call, K > F, K <= F)
        q, is_call, K = q[otm], is_call[otm], K[otm]
        row["otm"] = len(q)
        keep = q["bid"].to_numpy() >= cfg.min_bid
        q, is_call, K = q[keep], is_call[keep], K[keep]
        row["min_bid"] = len(q)
        bid = q["bid"].to_numpy(dtype=np.float64)
        ask = q["ask"].to_numpy(dtype=np.float64)
        mid = 0.5 * (bid + ask)
        iv_m = implied_vol(mid, F, K, T, D, is_call)
        iv_b = implied_vol(bid, F, K, T, D, is_call)
        iv_a = implied_vol(ask, F, K, T, D, is_call)
        keep = np.isfinite(iv_m) & np.isfinite(iv_b) & np.isfinite(iv_a)
        row["iv_exists"] = int(keep.sum())
        delta = np.full_like(mid, np.nan)
        delta[keep] = forward_delta(F, K[keep], T, iv_m[keep], is_call[keep])
        keep &= np.abs(np.nan_to_num(delta)) >= cfg.min_abs_delta
        row["delta"] = int(keep.sum())
        keep &= (iv_a - iv_b) <= cfg.max_vol_spread
        row["vol_spread"] = int(keep.sum())
        if keep.sum() < cfg.min_points:
            row["dropped"] = "too_few_points"
            log_rows.append(row)
            continue
        k = np.log(K[keep] / F)
        order = np.argsort(k)
        slices.append(
            Slice(
                expiry=str(expiry),
                maturity=T,
                forward=F,
                discount=D,
                forward_estimate=fwd,
                strike=K[keep][order],
                k=k[order],
                is_call=is_call[keep][order],
                mid=mid[keep][order],
                iv_mid=iv_m[keep][order],
                iv_bid=iv_b[keep][order],
                iv_ask=iv_a[keep][order],
            )
        )
        row["final"] = int(keep.sum())
        log_rows.append(row)
    return PreparedSnapshot(meta=meta, slices=tuple(slices), filter_log=pd.DataFrame(log_rows))


def load_snapshot(meta_path: str | Path, config: FilterConfig | None = None) -> PreparedSnapshot:
    """Load metadata + chain, verify the chain hash when recorded, and clean it."""
    meta = SnapshotMeta.from_json(meta_path)
    if not meta.chain_file.exists():
        raise FileNotFoundError(
            f"Chain file {meta.chain_file} not found. Raw vendor quotes are not redistributed; "
            "use the bundled synthetic sample or place your own snapshot there."
        )
    if meta.chain_sha256 is not None:
        digest = sha256_of(meta.chain_file)
        if digest != meta.chain_sha256:
            raise ValueError(f"SHA-256 mismatch for {meta.chain_file}: {digest} != {meta.chain_sha256}")
    return prepare_snapshot(load_chain(meta.chain_file), meta, config)
