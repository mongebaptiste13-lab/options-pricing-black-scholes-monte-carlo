"""Monte Carlo pricing under geometric Brownian motion (risk-neutral measure)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .blackscholes import bs_price


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def _payoff(ST: np.ndarray, K: float, option: str) -> np.ndarray:
    if option == "call":
        return np.maximum(ST - K, 0.0)
    if option == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError("option must be 'call' or 'put'")


def _terminal(S, T, r, sigma, q, z):
    return S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)


def _result(samples: np.ndarray) -> MCResult:
    n = len(samples)
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / math.sqrt(n)), n)


def mc_european(S, K, T, r, sigma, option="call", q=0.0, n_paths=100_000, antithetic=True, seed=None) -> MCResult:
    """European option by simulating terminal prices.

    With antithetic variates each normal draw z is paired with -z; the pair average is one
    independent sample, which lowers the variance for monotone payoffs.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    disc = math.exp(-r * T)
    pay = _payoff(_terminal(S, T, r, sigma, q, z), K, option)
    if antithetic:
        pay = 0.5 * (pay + _payoff(_terminal(S, T, r, sigma, q, -z), K, option))
    return _result(disc * pay)


def mc_european_greeks(S, K, T, r, sigma, option="call", q=0.0, n_paths=200_000, seed=None) -> dict:
    """Delta and vega with the pathwise estimator (no re-simulation needed)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    ST = _terminal(S, T, r, sigma, q, z)
    disc = math.exp(-r * T)
    sign = 1.0 if option == "call" else -1.0
    itm = (sign * (ST - K) > 0).astype(float)
    delta = _result(sign * disc * itm * ST / S)
    vega = _result(sign * disc * itm * ST * (z * math.sqrt(T) - sigma * T))
    return {"delta": delta, "vega": vega}


def _gbm_paths(S, T, r, sigma, q, n_steps, z):
    dt = T / n_steps
    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    return S * np.exp(np.cumsum(increments, axis=1))


def geometric_asian_price(S, K, T, r, sigma, n_steps, option="call", q=0.0) -> float:
    """Closed-form price of a discretely monitored geometric-average Asian option."""
    t = np.arange(1, n_steps + 1) * T / n_steps
    mean = math.log(S) + (r - q - 0.5 * sigma ** 2) * t.mean()
    var = sigma ** 2 * np.minimum.outer(t, t).sum() / n_steps ** 2
    sd = math.sqrt(var)
    d1 = (mean + var - math.log(K)) / sd
    d2 = d1 - sd
    disc = math.exp(-r * T)
    fwd = math.exp(mean + 0.5 * var)
    if option == "call":
        return disc * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))


def mc_asian_arithmetic(S, K, T, r, sigma, option="call", q=0.0, n_steps=50, n_paths=100_000,
                        control_variate=True, seed=None) -> MCResult:
    """Arithmetic-average Asian option (no closed form).

    The control variate is the geometric Asian option, which is highly correlated with the arithmetic
    one and has a known price; it removes most of the variance.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    paths = _gbm_paths(S, T, r, sigma, q, n_steps, z)
    disc = math.exp(-r * T)
    arith = disc * _payoff(paths.mean(axis=1), K, option)
    if not control_variate:
        return _result(arith)
    geo = disc * _payoff(np.exp(np.log(paths).mean(axis=1)), K, option)
    geo_exact = geometric_asian_price(S, K, T, r, sigma, n_steps, option, q)
    beta = np.cov(arith, geo)[0, 1] / geo.var(ddof=1)
    return _result(arith - beta * (geo - geo_exact))


def convergence_table(S, K, T, r, sigma, option="call", q=0.0, sizes=(1_000, 10_000, 100_000, 1_000_000),
                      antithetic=True, seed=42) -> pd.DataFrame:
    """Monte Carlo estimate vs Black-Scholes for increasing path counts."""
    exact = bs_price(S, K, T, r, sigma, option, q)
    rows = []
    for n in sizes:
        res = mc_european(S, K, T, r, sigma, option, q, n_paths=n, antithetic=antithetic, seed=seed)
        rows.append({"paths": n, "mc_price": res.price, "std_error": res.std_error,
                     "ci95_low": res.ci95[0], "ci95_high": res.ci95[1],
                     "bs_price": exact, "abs_error": abs(res.price - exact)})
    return pd.DataFrame(rows).set_index("paths")
