from .Heston_FFT import Heston_FFT

def APE(kappa, eta, theta, rho, sigma0, K, T, S0, r, q, type, integration_rule, market_price):
    heston_price = [0] * len(K)
    for i in range(len(K)):
        heston_price[i] = Heston_FFT(kappa, eta, theta, rho, sigma0, K[i], T, S0, r, q, type, integration_rule)
    out = sum(abs((heston_price - market_price) / market_price) * 100)
    return out