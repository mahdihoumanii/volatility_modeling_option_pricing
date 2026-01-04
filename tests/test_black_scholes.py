import math

from src.options.black_scholes import price_option


def test_black_scholes_call_put_prices_close_to_reference():
    result_call = price_option(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0, option_type="call")
    result_put = price_option(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0, option_type="put")
    assert math.isclose(result_call.price, 10.4506, rel_tol=1e-3)
    assert math.isclose(result_put.price, 5.5735, rel_tol=1e-3)
