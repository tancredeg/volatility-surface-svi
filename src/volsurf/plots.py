"""Figures for the README (matplotlib, static PNG with author/title metadata)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .pipeline import MODELS, StudyResults  # noqa: E402

AUTHOR = "Tancrède Geli"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
SERIES = {"SVI raw": "#2a78d6", "SVI constrained": "#eb6834", "SSVI": "#1baf7a"}
MARKET = "#52514e"
SHADE = "#efeeea"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "lines.solid_capstyle": "round",
    }
)


def save(fig: Figure, path: Path, title: str) -> Path:
    """Save with PNG metadata (title, author) instead of library defaults."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Title": title, "Author": AUTHOR, "Software": "volsurf (matplotlib)"},
    )
    plt.close(fig)
    return path


def _shade_unquoted(ax: plt.Axes, k_lo: float, k_hi: float, x_lo: float, x_hi: float) -> None:
    ax.axvspan(x_lo, k_lo, color=SHADE, lw=0, zorder=0)
    ax.axvspan(k_hi, x_hi, color=SHADE, lw=0, zorder=0)


def plot_smiles(res: StudyResults, path: Path) -> Path:
    """Market quotes (mid, with bid-ask) and the three calibrated models, per expiry."""
    snap, cal = res.snapshot, res.calibration
    n = len(snap.slices)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.4 * nrows), squeeze=False)
    for i, s in enumerate(snap.slices):
        ax = axes[i // ncols][i % ncols]
        x_lo, x_hi = s.k_range[0] - 0.1, s.k_range[1] + 0.3
        kk = np.linspace(x_lo, x_hi, 600)
        _shade_unquoted(ax, *s.k_range, x_lo, x_hi)
        ax.errorbar(
            s.k,
            s.iv_mid * 100,
            yerr=[(s.iv_mid - s.iv_bid) * 100, (s.iv_ask - s.iv_mid) * 100],
            fmt="o",
            ms=2.2,
            color=MARKET,
            elinewidth=0.6,
            label="Market (mid, bid-ask)",
            zorder=3,
        )
        for m in MODELS:
            p = cal.slice_params(m)[i]
            ax.plot(
                kk, np.sqrt(np.maximum(p.w(kk), 0) / s.maturity) * 100, color=SERIES[m], label=m, zorder=4
            )
        rmse = {m: res.fit.loc[i, f"rmse_vp[{m}]"] for m in MODELS}
        ax.set_title(
            f"{s.expiry} ({s.maturity * 365:.0f} days)  RMSE vp: "
            + ", ".join(
                f"{m.split()[0] if m != 'SVI constrained' else 'SVI-c'} {v:.2f}" for m, v in rmse.items()
            ),
            loc="left",
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("log-moneyness k = ln(K/F)")
        ax.set_ylabel("implied vol (%)")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.01, "Shaded: outside the quoted strike range (extrapolation).", ha="center", color=INK_2)
    fig.tight_layout()
    return save(
        fig,
        path,
        f"SPY implied-vol smiles and fits, {snap.meta.underlying} {snap.meta.valuation_time:%Y-%m-%d}",
    )


def plot_arbitrage(res: StudyResults, path: Path) -> Path:
    """Left: Durrleman g(k) on the product expiry. Right: total-variance gap between the last two expiries."""
    snap, cal = res.snapshot, res.calibration
    idx = int(np.argmin([abs(s.maturity - res.config.product_target_years) for s in snap.slices]))
    s = snap.slices[idx]
    kk = np.linspace(-1.5, 1.5, 1201)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _shade_unquoted(a1, *s.k_range, -1.5, 1.5)
    for m in MODELS:
        a1.plot(kk, cal.slice_params(m)[idx].g(kk), color=SERIES[m], label=m)
    a1.axhline(0, color=INK, lw=0.8)
    a1.set_ylim(-0.6, 1.2)
    a1.set_title(
        f"Butterfly: Durrleman g(k), {s.maturity * 365:.0f} days (g < 0 = negative density)", loc="left"
    )
    a1.set_xlabel("k")
    s1, s2 = snap.slices[-2], snap.slices[-1]
    lo, hi = max(s1.k_range[0], s2.k_range[0]), min(s1.k_range[1], s2.k_range[1])
    _shade_unquoted(a2, lo, hi, -1.5, 1.5)
    for m in MODELS:
        p1, p2 = cal.slice_params(m)[-2], cal.slice_params(m)[-1]
        a2.plot(kk, p2.w(kk) - p1.w(kk), color=SERIES[m], label=m)
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_ylim(-0.1, 0.1)
    a2.legend(loc="lower left")
    a2.set_title(
        f"Calendar: w(k, {s2.maturity * 365:.0f}d) - w(k, {s1.maturity * 365:.0f}d) (< 0 = arbitrage)",
        loc="left",
    )
    a2.set_xlabel("k")
    a1.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, path, "Static-arbitrage audit of the calibrated slices")


def plot_local_vol(res: StudyResults, path: Path) -> Path:
    """Dupire local vol from SSVI against SSVI implied vol at several maturities."""
    b = res.barrier
    fit = res.calibration.ssvi
    from .localvol import ssvi_local_variance

    y = np.linspace(-0.45, 0.2, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ramp = ["#9cc2f0", "#5f9be3", "#2a78d6", "#1a4f91"]
    for t, c in zip((b.maturity / 8, b.maturity / 4, b.maturity / 2, b.maturity), ramp, strict=True):
        lv = np.sqrt(ssvi_local_variance(fit.params, fit.theta_curve(), float(t), y)) * 100
        a1.plot(y, lv, color=c, label=f"t = {t * 365:.0f}d")
    a1.set_title("Dupire local vol from SSVI", loc="left")
    a1.set_xlabel("y = ln(S/F(0,t))")
    a1.set_ylabel("vol (%)")
    a1.legend()
    T = b.maturity
    a2.plot(y, fit.implied_vol(y, T) * 100, color=SERIES["SSVI"], label=f"SSVI implied vol, {T * 365:.0f}d")
    a2.plot(
        y,
        np.sqrt(ssvi_local_variance(fit.params, fit.theta_curve(), T, y)) * 100,
        color="#1a4f91",
        label=f"local vol, t = {T * 365:.0f}d",
    )
    sl = [s for s in res.snapshot.slices if s.expiry == b.expiry][0]
    a2.plot(sl.k, sl.iv_mid * 100, "o", ms=2.2, color=MARKET, label="market mid")
    y_lo = a2.get_ylim()[0]
    for lvl in (0.8, 0.9):
        a2.axvline(np.log(lvl * b.spot / b.forward_T), color=INK_2, lw=0.6, ls=":")
        a2.text(
            np.log(lvl * b.spot / b.forward_T),
            y_lo + 1.0,
            f" barrier {lvl:.0%}",
            color=INK_2,
            va="bottom",
            fontsize=8,
        )
    a2.set_xlim(y[0], y[-1])
    h = 1e-3
    lv_fn = fit.theta_curve()
    lv_slope = np.diff(np.sqrt(ssvi_local_variance(fit.params, lv_fn, T, np.array([-h, h]))))[0] / (2 * h)
    iv_slope = np.diff(fit.implied_vol(np.array([-h, h]), T))[0] / (2 * h)
    a2.set_title(f"At the money, local-vol skew = {lv_slope / iv_slope:.1f} x implied skew", loc="left")
    a2.set_xlabel("k = ln(K/F)")
    a2.legend(loc="upper right")
    fig.tight_layout()
    return save(fig, path, "Dupire local volatility from the SSVI surface")


def plot_barrier(res: StudyResults, path: Path) -> Path:
    """Down-and-in put and down-and-out call prices (% of spot) across barrier levels."""
    b = res.barrier
    t = b.table
    x = t["barrier_pct"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    colors = {"lv": "#2a78d6", "flat": "#eb6834", "hvol": "#1baf7a"}
    for ax, prod, name in ((a1, "dip", "Down-and-in put"), (a2, "doc", "Down-and-out call")):
        lv = t[f"{prod}_lv"].to_numpy() / b.spot * 100
        se = t[f"{prod}_lv_se"].to_numpy() / b.spot * 100
        ax.errorbar(
            x, lv, yerr=1.96 * se, color=colors["lv"], marker="o", ms=4, label="Local vol (Dupire, MC)"
        )
        ax.plot(
            x,
            t[f"{prod}_flat"] / b.spot * 100,
            color=colors["flat"],
            marker="s",
            ms=4,
            label=f"BS flat vol at strike ({b.flat_vol * 100:.1f}%)",
        )
        if prod == "dip":
            ax.plot(
                x,
                t[f"{prod}_barrier_vol_closed_form"] / b.spot * 100,
                color=colors["hvol"],
                marker="^",
                ms=4,
                label="BS flat vol at barrier level",
            )
        ax.set_xticks(x)
        ax.set_xlabel("barrier (% of spot)")
        ax.set_ylabel("price (% of spot)")
        ax.set_title(f"{name}, strike 100%, {b.maturity * 365:.0f} days", loc="left")
    a1.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, path, "Barrier prices: local vol vs flat Black-Scholes")


def plot_all(res: StudyResults, out_dir: Path) -> list[Path]:
    return [
        plot_smiles(res, out_dir / "smiles.png"),
        plot_arbitrage(res, out_dir / "arbitrage.png"),
        plot_local_vol(res, out_dir / "local_vol.png"),
        plot_barrier(res, out_dir / "barrier.png"),
    ]
