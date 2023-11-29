from Heston_FFT import Heston_FFT

def sqrd_diff(kappa, eta, theta, rho, sigma0, K, T, S0, r, q, type, integration_rule, market_price):
    # Detailed explanation goes here
    out = (Heston_FFT(kappa, eta, theta, rho, sigma0, K, T, S0, r, q, type, integration_rule) - market_price) ** 2
    return out