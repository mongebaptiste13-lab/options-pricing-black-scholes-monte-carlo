"""Figures for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .blackscholes import bs_greeks, bs_price
from .montecarlo import convergence_table, mc_asian_arithmetic, mc_european

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20
BLUE, RED, GREEN, GREY = "#1f4e79", "#c0392b", "#4f9d69", "#7f8c8d"


def _style(ax):
    ax.grid(alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def plot_convergence(path: str | Path) -> None:
    sizes = [10 ** k for k in range(3, 7)] + [3 * 10 ** k for k in range(3, 6)]
    t = convergence_table(S, K, T, R, SIG, sizes=sorted(sizes), seed=42)
    exact = bs_price(S, K, T, R, SIG, "call")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.fill_between(t.index, t["ci95_low"], t["ci95_high"], color=BLUE, alpha=0.2, label="95% confidence interval")
    ax.plot(t.index, t["mc_price"], color=BLUE, marker="o", label="Monte Carlo")
    ax.axhline(exact, color=RED, ls="--", label=f"Black-Scholes ({exact:.4f})")
    ax.set_xscale("log"); ax.set_xlabel("Number of paths"); ax.set_ylabel("Call price")
    ax.set_title("Monte Carlo converges to Black-Scholes"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.loglog(t.index, t["std_error"], color=BLUE, marker="o", label="Standard error")
    ref = t["std_error"].iloc[0] * np.sqrt(t.index[0] / t.index.to_numpy())
    ax.loglog(t.index, ref, color=GREY, ls="--", label=r"$1/\sqrt{N}$ reference")
    ax.set_xlabel("Number of paths"); ax.set_title("Standard error falls as 1/sqrt(N)"); ax.legend(fontsize=8)
    for a in axes: _style(a)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_greeks(path: str | Path) -> None:
    spots = np.linspace(60, 140, 161)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    series = {
        "Price (call and put)": None, "Delta": "delta", "Gamma": "gamma", "Vega (per 1.00 vol)": "vega",
        "Theta (per year)": "theta", "Rho (per 1.00 rate)": "rho",
    }
    for ax, (title, key) in zip(axes.flat, series.items()):
        if key is None:
            ax.plot(spots, [bs_price(s, K, T, R, SIG, "call") for s in spots], color=BLUE, label="Call")
            ax.plot(spots, [bs_price(s, K, T, R, SIG, "put") for s in spots], color=RED, label="Put")
            ax.plot(spots, np.maximum(spots - K, 0), color=BLUE, ls=":", lw=1, label="Payoff")
            ax.plot(spots, np.maximum(K - spots, 0), color=RED, ls=":", lw=1)
            ax.legend(fontsize=8)
        else:
            ax.plot(spots, [bs_greeks(s, K, T, R, SIG, "call")[key] for s in spots], color=BLUE, label="Call")
            if key != "gamma" and key != "vega":
                ax.plot(spots, [bs_greeks(s, K, T, R, SIG, "put")[key] for s in spots], color=RED, label="Put")
                ax.legend(fontsize=8)
        ax.axvline(K, color=GREY, lw=0.8, ls=":"); ax.set_title(title); ax.set_xlabel("Spot"); _style(ax)
    fig.suptitle(f"Black-Scholes price and Greeks vs spot (K={K:.0f}, T={T:.0f}y, r={R:.0%}, sigma={SIG:.0%})")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_variance_reduction(path: str | Path) -> None:
    n = 100_000
    eu_plain = mc_european(S, K, T, R, SIG, n_paths=n, antithetic=False, seed=1).std_error
    eu_anti = mc_european(S, K, T, R, SIG, n_paths=n, antithetic=True, seed=1).std_error
    as_plain = mc_asian_arithmetic(S, K, T, R, SIG, n_paths=n, control_variate=False, seed=1).std_error
    as_cv = mc_asian_arithmetic(S, K, T, R, SIG, n_paths=n, control_variate=True, seed=1).std_error
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, title in ((axes[0], (eu_plain, eu_anti), "European call (100k paths)"),
                            (axes[1], (as_plain, as_cv), "Arithmetic Asian call (100k paths)")):
        labels = ["Plain", "Antithetic" if ax is axes[0] else "Control variate"]
        ax.bar(labels, vals, color=[GREY, GREEN])
        for k, v in enumerate(vals):
            ax.text(k, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title); ax.set_ylabel("Standard error"); _style(ax)
        ax.set_xlabel(f"Error reduced {vals[0] / vals[1]:.0f}x")
    fig.suptitle("Variance reduction")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
