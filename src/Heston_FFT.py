import numpy as np
from scipy.interpolate import CubicSpline
from heston_characteristic import heston_characteristic

def Heston_FFT(kappa, eta, theta, rho, sigma0, K, T, S0, r, q, type, integration_rule):
    # integration_rule = 0 --> rectangular rule
    # integration_rule = 1 --> Simpson's rule
    # type = 0 --> Call option
    # type = 1 --> Put option

    # define parameters
    N = 4096
    alpha = 1.5
    eta_grid = 0.25
    lambda_val = 2 * np.pi / (N * eta_grid)
    b = lambda_val * N / 2

    # define grid of log-strikesAPE has size 4095
    #k = np.arange(-b, b - lambda_val, lambda_val)

    k = np.arange(-b, b, lambda_val)
    # compute rho
    v = np.arange(0, N * eta_grid, eta_grid)
    u = v - (alpha + 1) * 1j
    rho_val = np.exp(-r * T) * heston_characteristic(kappa, eta, theta, rho, sigma0, r, q, S0, T, u) / (
                alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v)

    if integration_rule == 0:
        a = np.real(np.fft.fft(rho_val * np.exp(1j * v * b) * eta_grid, N))
    elif integration_rule == 1:
        simpson_1 = 1 / 3
        simpson = (3 + (-1) ** np.arange(2, N + 1)) / 3
        simpson_int = np.concatenate(([simpson_1], simpson))
        a = np.real(np.fft.fft(rho_val * np.exp(1j * v * b) * eta_grid * simpson_int, N))

    CallPrices = (1 / np.pi) * np.exp(-alpha * k) * a

    # find C(K,T)
    #Added int on K, if this is the strike price why does it have to be an integer?
    KK = np.exp(k)
    #out = CubicSpline(KK, CallPrices, int(K))
    if not np.all(np.isfinite(KK)) or not np.all(np.isfinite(CallPrices)):
        print("paus")
    cs = CubicSpline(KK,CallPrices)
    out = cs(K)

    if type == 1:
        out = out + K * np.exp(-r * T) - np.exp(-q * T) * S0

    return out