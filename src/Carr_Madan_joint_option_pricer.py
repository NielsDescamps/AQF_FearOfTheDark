import numpy as np
from scipy.interpolate import CubicSpline
from jump_characteristic_function import joint_characteristic_function


"""
This function uses the Carr-Madan pricing formula to calculate the price of a option
The underlying model joint heston & contagoius jump model
This function is not independently verified, but compared to heston model where it obtains reasonable results

Params:
t = time of evaluation (only implemented for t=0)
T = end time
K = strike price of priced option
type = 0 -> Call option 1 -> Put option
eta = heston param
rho = heston param
sigma0 = heston param
r = riskfree rate
q = market param (what is this really? something with risk free measure?)
S0 = last close
params = jump model parameters
lambda_zero = initial jump intensity (i think)
h = time step for the evaluation of the set of ODEs
index = on which index the price is evaulated
jump_distribution = "Gaussian"/"Exponential" specifies the pdf of the jump distribution

"""
def Carr_Madan_joint_option_pricer(t, T, K, type, kappa, eta, theta, rho, sigma0, r, q, S0, params, lambda_zero, h, index, jump_distribution):
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

    k = np.arange(-b, b, lambda_val)
    # compute rho
    v = np.arange(0, N * eta_grid, eta_grid)
    u = v - (alpha + 1) * 1j
    rho_val = [np.exp(-r * T) * joint_characteristic_function(u[i], t, T, kappa, eta, theta, rho, sigma0, r, q, S0, params, lambda_zero, h, index, jump_distribution) / (
                alpha ** 2 + alpha - v[i] ** 2 + 1j * (2 * alpha + 1) * v[i]) for i in range(len(u))]

    #No need for this too be variable as the simpson is more accurate and not computationally complex
    integration_rule = 1
    if integration_rule == 0:
        a = np.real(np.fft.fft(rho_val * np.exp(1j * v * b) * eta_grid, N))
    elif integration_rule == 1:
        simpson_1 = 1 / 3
        simpson = (3 + (-1) ** np.arange(2, N + 1)) / 3
        simpson_int = np.concatenate(([simpson_1], simpson))
        a = np.real(np.fft.fft(rho_val * np.exp(1j * v * b) * eta_grid * simpson_int, N))

    CallPrices = (1 / np.pi) * np.exp(-alpha * k) * a

    # find C(K,T)
    KK = np.exp(k)

    if not np.all(np.isfinite(KK)) or not np.all(np.isfinite(CallPrices)):
        print("paus")
    cs = CubicSpline(KK,CallPrices)
    out = cs(K)

    if type == 1:
        out = out + K * np.exp(-r * T) - np.exp(-q * T) * S0

    return out