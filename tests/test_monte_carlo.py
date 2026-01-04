from src.options.black_scholes import price_option
from src.options.monte_carlo import price_european_option_mc, price_european_option_mc_ci


def test_monte_carlo_converges_to_black_scholes():
    bs_price = price_option(spot=100, strike=100, rate=0.01, vol=0.2, maturity=1.0, option_type="call").price
    result_mc = price_european_option_mc(
        spot=100,
        strike=100,
        rate=0.01,
        vol=0.2,
        maturity=1.0,
        option_type="call",
        steps=252,
        n_paths=30_000,
        rng=None,
    )
    assert abs(result_mc.price - bs_price) < 0.5


def test_monte_carlo_confidence_interval_monotonic_shrink():
    # Using a fixed seed to make results reproducible
    res_low = price_european_option_mc_ci(
        spot=100,
        strike=100,
        rate=0.01,
        vol=0.2,
        maturity=1.0,
        option_type="call",
        steps=252,
        n_paths=5_000,
        rng_seed=42,
    )
    res_high = price_european_option_mc_ci(
        spot=100,
        strike=100,
        rate=0.01,
        vol=0.2,
        maturity=1.0,
        option_type="call",
        steps=252,
        n_paths=40_000,
        rng_seed=42,
    )
    assert res_low["stderr"] > res_high["stderr"]
    assert res_low["ci_high"] - res_low["price"] > res_high["ci_high"] - res_high["price"]
