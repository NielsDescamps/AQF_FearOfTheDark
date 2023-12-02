import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def BS_price(sigma, S0, K, r, q, T, flag):
    """
    Compute Black-Scholes prices of call and put options.

    Parameters:
    - sigma: volatility
    - S0: current stock price
    - K: option strike price
    - r: risk-free interest rate
    - q: dividend yield
    - T: time to maturity
    - flag: 0 for call option, 1 for put option

    Returns:
    - Option price
    """
    d1 = (np.log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = np.exp(-q * T) * S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = -np.exp(-q * T) * S0 * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(-d2)

    # put option: flag = 1
    # call option: flag = 0
    out = call * (1 - flag) + put * flag

    return out

def calc_implied_vol(market_price, S0, K, r, q, T, flag):
    """
    This function returns the implied volatility of a single option using the brentq method 

    Parameters: 
    - market_price: market price of the put or call option
    - S0: current stock price
    - K: option strike price
    - r: risk-free interest rate
    - q: dividend yield
    - T: time to maturity
    - flag: 0 for call option, 1 for put option

    Output: 
    - implied volatility
    """ 

    def black_scholes(sigma):
        d1 = (np.log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call = np.exp(-q * T) * S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put = -np.exp(-q * T) * S0 * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(-d2)

        return call * (1 - flag) + put * flag

    # Use brentq method to find implied volatility within a specified interval
    try:
        implied_vol = brentq(lambda sigma: black_scholes(sigma) - market_price, a=0.01, b=2)
    except ValueError:
        print("Error in implied vol function")
        implied_vol = None


    return implied_vol


