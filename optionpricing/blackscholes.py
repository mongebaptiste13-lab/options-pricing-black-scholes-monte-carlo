"""Black-Scholes-Merton pricing, Greeks and implied volatility for European options.

Notation: S spot, K strike, T maturity in years, r risk-free rate, sigma volatility, q continuous
dividend yield. All rates are decimals (0.05 = 5%).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _check(S, K, T, sigma):
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    _check(S, K, T, sigma)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option: str = "call", q: float = 0.0) -> float:
    """Price of a European call or put."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option: str = "call", q: float = 0.0) -> dict:
    """Delta, gamma, vega (per 1.00 of volatility), theta (per year) and rho (per 1.00 of rate).

    Divide vega and rho by 100 for the usual 'per 1%' convention and theta by 365 for per day.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
    gamma = disc_q * pdf / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T)
    if option == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T)) - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    elif option == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T)) + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def put_call_parity_gap(S, K, T, r, sigma, q=0.0) -> float:
    """C - P - (S e^{-qT} - K e^{-rT}); zero up to rounding for Black-Scholes prices."""
    c = bs_price(S, K, T, r, sigma, "call", q)
    p = bs_price(S, K, T, r, sigma, "put", q)
    return c - p - (S * math.exp(-q * T) - K * math.exp(-r * T))


def implied_volatility(price: float, S: float, K: float, T: float, r: float, option: str = "call",
                       q: float = 0.0, lo: float = 1e-6, hi: float = 5.0) -> float:
    """Volatility that reproduces a market price (Brent root finding)."""
    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
    if option == "call":
        lower, upper = max(S * disc_q - K * disc_r, 0.0), S * disc_q
    else:
        lower, upper = max(K * disc_r - S * disc_q, 0.0), K * disc_r
    if not lower < price < upper:
        raise ValueError(f"price {price:.4f} outside no-arbitrage bounds ({lower:.4f}, {upper:.4f})")
    return brentq(lambda s: bs_price(S, K, T, r, s, option, q) - price, lo, hi, xtol=1e-12)
