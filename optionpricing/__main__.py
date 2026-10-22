"""Command line: python -m optionpricing  (prints a comparison table and writes figures to docs/)"""
from pathlib import Path

import pandas as pd

from .blackscholes import bs_greeks, bs_price, implied_volatility
from .montecarlo import mc_asian_arithmetic, mc_european, mc_european_greeks
from .report import plot_convergence, plot_greeks, plot_variance_reduction

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20


def main() -> None:
    out = Path("docs")
    out.mkdir(exist_ok=True)
    print(f"Inputs: S={S}, K={K}, T={T}y, r={R:.0%}, sigma={SIG:.0%}\n")
    rows = []
    for opt in ("call", "put"):
        mc = mc_european(S, K, T, R, SIG, opt, n_paths=1_000_000, seed=42)
        rows.append({"option": opt, "Black-Scholes": bs_price(S, K, T, R, SIG, opt), "Monte Carlo": mc.price,
                     "std error": mc.std_error, "95% CI low": mc.ci95[0], "95% CI high": mc.ci95[1]})
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(pd.DataFrame(rows).set_index("option").to_string(), "\n")

    g = bs_greeks(S, K, T, R, SIG, "call")
    mcg = mc_european_greeks(S, K, T, R, SIG, "call", n_paths=1_000_000, seed=42)
    print(f"Call delta: BS {g['delta']:.4f} | MC pathwise {mcg['delta'].price:.4f} +/- {mcg['delta'].std_error:.4f}")
    print(f"Call vega : BS {g['vega']:.4f} | MC pathwise {mcg['vega'].price:.4f} +/- {mcg['vega'].std_error:.4f}")
    asian = mc_asian_arithmetic(S, K, T, R, SIG, n_paths=200_000, seed=42)
    print(f"Arithmetic Asian call (control variate): {asian.price:.4f} +/- {asian.std_error:.4f}")
    print(f"Implied vol recovered from the BS call price: {implied_volatility(bs_price(S, K, T, R, SIG), S, K, T, R):.4%}")

    plot_convergence(out / "convergence.png")
    plot_greeks(out / "greeks.png")
    plot_variance_reduction(out / "variance_reduction.png")
    print("\nFigures written to docs/")


if __name__ == "__main__":
    main()
