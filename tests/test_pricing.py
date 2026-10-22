import math

import numpy as np
import pytest

from optionpricing import (bs_greeks, bs_price, convergence_table, geometric_asian_price, implied_volatility,
                           mc_asian_arithmetic, mc_european, mc_european_greeks, put_call_parity_gap)

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20


def test_reference_values():
    # Textbook values for S=K=100, r=5%, sigma=20%, T=1y
    assert bs_price(S, K, T, R, SIG, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(S, K, T, R, SIG, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("q", [0.0, 0.03])
@pytest.mark.parametrize("K_", [80.0, 100.0, 125.0])
def test_put_call_parity(K_, q):
    assert put_call_parity_gap(S, K_, T, R, SIG, q) == pytest.approx(0.0, abs=1e-10)


def test_price_bounds_and_monotonicity():
    prices = [bs_price(s, K, T, R, SIG, "call") for s in (80, 90, 100, 110, 120)]
    assert all(b > a for a, b in zip(prices, prices[1:]))
    assert bs_price(S, K, T, R, SIG, "call") >= max(S - K * math.exp(-R * T), 0)
    assert bs_price(S, K, T, R, 0.4, "call") > bs_price(S, K, T, R, 0.2, "call")


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_greeks_match_finite_differences(option, q):
    g = bs_greeks(S, K, T, R, SIG, option, q)
    h = 1e-4
    price = lambda s=S, t=T, r=R, sig=SIG: bs_price(s, K, t, r, sig, option, q)
    delta = (price(s=S + h) - price(s=S - h)) / (2 * h)
    gamma = (price(s=S + h) - 2 * price() + price(s=S - h)) / h ** 2
    vega = (price(sig=SIG + h) - price(sig=SIG - h)) / (2 * h)
    rho = (price(r=R + h) - price(r=R - h)) / (2 * h)
    theta = -(price(t=T + h) - price(t=T - h)) / (2 * h)
    assert g["delta"] == pytest.approx(delta, abs=1e-6)
    assert g["gamma"] == pytest.approx(gamma, abs=1e-5)
    assert g["vega"] == pytest.approx(vega, abs=1e-5)
    assert g["rho"] == pytest.approx(rho, abs=1e-5)
    assert g["theta"] == pytest.approx(theta, abs=1e-5)


def test_delta_call_minus_put_is_one_without_dividends():
    c, p = bs_greeks(S, K, T, R, SIG, "call"), bs_greeks(S, K, T, R, SIG, "put")
    assert c["delta"] - p["delta"] == pytest.approx(1.0)
    assert c["gamma"] == pytest.approx(p["gamma"]) and c["vega"] == pytest.approx(p["vega"])


@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("option", ["call", "put"])
def test_implied_volatility_roundtrip(sigma, option):
    price = bs_price(S, 110, 0.5, R, sigma, option, 0.01)
    assert implied_volatility(price, S, 110, 0.5, R, option, 0.01) == pytest.approx(sigma, abs=1e-8)


def test_implied_volatility_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        implied_volatility(0.01, S, K, T, R, "call")      # below intrinsic value (forward-adjusted)
    with pytest.raises(ValueError):
        implied_volatility(S + 1, S, K, T, R, "call")     # above the spot


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bs_price(S, K, 0.0, R, SIG)
    with pytest.raises(ValueError):
        bs_price(S, K, T, R, -0.1)
    with pytest.raises(ValueError):
        bs_price(S, K, T, R, SIG, "straddle")


@pytest.mark.parametrize("option", ["call", "put"])
def test_monte_carlo_within_confidence_interval(option):
    exact = bs_price(S, K, T, R, SIG, option)
    res = mc_european(S, K, T, R, SIG, option, n_paths=400_000, seed=7)
    assert abs(res.price - exact) < 4 * res.std_error
    assert res.ci95[0] < res.price < res.ci95[1]


def test_antithetic_variates_reduce_standard_error():
    plain = mc_european(S, K, T, R, SIG, "call", n_paths=200_000, antithetic=False, seed=3)
    anti = mc_european(S, K, T, R, SIG, "call", n_paths=200_000, antithetic=True, seed=3)
    assert anti.std_error < plain.std_error


def test_standard_error_shrinks_like_inverse_sqrt_n():
    t = convergence_table(S, K, T, R, SIG, sizes=(10_000, 1_000_000), seed=5)
    ratio = t["std_error"].iloc[0] / t["std_error"].iloc[1]
    assert ratio == pytest.approx(10.0, rel=0.1)


def test_mc_is_reproducible_with_seed():
    a = mc_european(S, K, T, R, SIG, n_paths=10_000, seed=11).price
    b = mc_european(S, K, T, R, SIG, n_paths=10_000, seed=11).price
    assert a == b


@pytest.mark.parametrize("option", ["call", "put"])
def test_pathwise_greeks_match_black_scholes(option):
    g = bs_greeks(S, K, T, R, SIG, option)
    mc = mc_european_greeks(S, K, T, R, SIG, option, n_paths=400_000, seed=9)
    assert abs(mc["delta"].price - g["delta"]) < 4 * mc["delta"].std_error
    assert abs(mc["vega"].price - g["vega"]) < 4 * mc["vega"].std_error


def test_geometric_asian_closed_form_matches_simulation():
    n_steps = 12
    exact = geometric_asian_price(S, K, T, R, SIG, n_steps)
    rng = np.random.default_rng(1)
    z = rng.standard_normal((300_000, n_steps))
    dt = T / n_steps
    paths = S * np.exp(np.cumsum((R - 0.5 * SIG ** 2) * dt + SIG * math.sqrt(dt) * z, axis=1))
    pay = math.exp(-R * T) * np.maximum(np.exp(np.log(paths).mean(axis=1)) - K, 0)
    se = pay.std(ddof=1) / math.sqrt(len(pay))
    assert abs(pay.mean() - exact) < 4 * se


def test_asian_control_variate_cuts_variance_and_ordering_holds():
    plain = mc_asian_arithmetic(S, K, T, R, SIG, control_variate=False, n_paths=60_000, seed=2)
    cv = mc_asian_arithmetic(S, K, T, R, SIG, control_variate=True, n_paths=60_000, seed=2)
    assert cv.std_error < plain.std_error / 10
    assert abs(cv.price - plain.price) < 4 * plain.std_error
    # arithmetic average >= geometric average, and an Asian call is cheaper than the vanilla call
    assert cv.price >= geometric_asian_price(S, K, T, R, SIG, 50)
    assert cv.price < bs_price(S, K, T, R, SIG, "call")
