"""
Black–Scholes closed-form pricing for European options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass
class BlackScholesResult:
    price: float
    d1: float
    d2: float


def _d1_d2(spot: float, strike: float, rate: float, vol: float, maturity: float) -> Tuple[float, float]:
    if vol <= 0 or maturity <= 0:
        raise ValueError("Volatility and maturity must be positive.")
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol**2) * maturity) / (vol * math.sqrt(maturity))
    d2 = d1 - vol * math.sqrt(maturity)
    return d1, d2


def price_option(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    option_type: OptionType = "call",
) -> BlackScholesResult:
    """
    Price a European option using Black–Scholes.
    """
    d1, d2 = _d1_d2(spot, strike, rate, vol, maturity)
    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * math.exp(-rate * maturity) * norm.cdf(d2)
    elif option_type == "put":
        price = strike * math.exp(-rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return BlackScholesResult(price=price, d1=d1, d2=d2)
