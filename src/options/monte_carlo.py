"""
Monte Carlo pricing for European options under geometric Brownian motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

OptionType = Literal["call", "put"]


@dataclass
class MonteCarloResult:
    price: float
    std_error: float
    paths: int


def simulate_gbm_paths(
    spot: float,
    rate: float,
    vol: float,
    maturity: float,
    steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate GBM paths.
    """
    rng = rng or np.random.default_rng()
    dt = maturity / steps
    drift = (rate - 0.5 * vol**2) * dt
    diffusion = vol * math.sqrt(dt)
    increments = drift + diffusion * rng.standard_normal(size=(n_paths, steps))
    log_paths = increments.cumsum(axis=1)
    s_paths = spot * np.exp(log_paths)
    return s_paths


def price_european_option_mc(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    option_type: OptionType = "call",
    steps: int = 252,
    n_paths: int = 50_000,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """
    Monte Carlo estimator for European option prices under GBM.
    """
    rng = rng or np.random.default_rng()
    paths = simulate_gbm_paths(spot, rate, vol, maturity, steps, n_paths, rng)
    terminal = paths[:, -1]
    if option_type == "call":
        payoff = np.maximum(terminal - strike, 0)
    elif option_type == "put":
        payoff = np.maximum(strike - terminal, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    discounted_payoff = math.exp(-rate * maturity) * payoff
    price = float(discounted_payoff.mean())
    std_error = float(discounted_payoff.std(ddof=1) / math.sqrt(n_paths))
    return MonteCarloResult(price=price, std_error=std_error, paths=n_paths)


def price_european_option_mc_ci(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    option_type: OptionType = "call",
    steps: int = 252,
    n_paths: int = 50_000,
    rng: np.random.Generator | None = None,
    rng_seed: int | None = None,
) -> dict[str, float]:
    """
    Monte Carlo estimator with standard error and 95% confidence interval.

    Returns
    -------
    dict
        Keys: price, stderr, ci_low, ci_high.
    """
    if rng is None:
        rng = np.random.default_rng(rng_seed)
    result = price_european_option_mc(
        spot=spot,
        strike=strike,
        rate=rate,
        vol=vol,
        maturity=maturity,
        option_type=option_type,
        steps=steps,
        n_paths=n_paths,
        rng=rng,
    )
    price = result.price
    stderr = result.std_error
    ci_radius = 1.96 * stderr
    return {
        "price": price,
        "stderr": stderr,
        "ci_low": price - ci_radius,
        "ci_high": price + ci_radius,
    }


def convergence_curve(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    option_type: OptionType,
    steps: int,
    path_grid: Tuple[int, ...],
    rng_seed: int = 42,
) -> dict[int, float]:
    """
    Price across a grid of path counts to study convergence.
    """
    rng = np.random.default_rng(rng_seed)
    prices: dict[int, float] = {}
    for n_paths in path_grid:
        result = price_european_option_mc(
            spot,
            strike,
            rate,
            vol,
            maturity,
            option_type=option_type,
            steps=steps,
            n_paths=n_paths,
            rng=rng,
        )
        prices[n_paths] = result.price
    return prices
