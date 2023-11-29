from Heston_FFT import Heston_FFT

def sse(kappa, eta, theta, rho, sigma0, K, T, S0, r, q, type, integration_rule, market_price):
    heston_price = [0] * len(K)
    for i in range(len(K)):
        heston_price[i] = Heston_FFT(kappa, eta, theta, rho, sigma0, K[i], T, S0, r, q, type, integration_rule)
    out = sum([(heston_price[i] - market_price[i]) ** 2 for i in range(len(K))])
    return out