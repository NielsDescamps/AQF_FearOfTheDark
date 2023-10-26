import numpy as np



def heston_characteristic(r,q,kappa,eta,theta, rho, sigma0,S0,u,t):

    d = np.sqrt((rho * theta * u * 1j - kappa) ** 2 - theta ** 2 * (-1j * u - u ** 2))
    g = (kappa - rho * theta * u * 1j - d) / (kappa - rho * theta * u * 1j + d)

    p1 = 1j * u * (np.log(S0) + (r - q) * t)
    p2 = eta * kappa * theta ** (-2) * ((kappa - rho * theta * u * 1j - d) * t -
                                        2 * np.log((1 - g * np.exp(-d * t)) / (1 - g)))
    p3 = sigma0 ** 2 * theta ** (-2) * (kappa - rho * theta * u * 1j - d) * \
         (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))

    out = np.exp(p1) * np.exp(p2) * np.exp(p3)

    return out


r = 0.01
q=0 
kappa = 0.05
eta = 0.05
theta = 0.2
rho = -0.75
sigma0= 0.2
S0 = 100
t=2

res = heston_characteristic(r,q,kappa,eta,theta, rho, sigma0,S0,u,t)
print(res)

def Heston_FFT(kappa, eta, theta, rho, sigma0, K, T, S0, r, q, option_type, integration_rule):
    # Parameters
    N = 4096
    alpha = 1.5
    eta_grid = 0.25
    lambda_val = 2 * np.pi / (N * eta_grid)
    b = lambda_val * N / 2

    # Grid of log-strikes
    k = np.arange(-b, b, lambda_val)

    # Compute rho
    v = np.arange(0, N * eta_grid, eta_grid)
    u = v - (alpha + 1) * 1j
    heston_char = np.exp(-r * T) * heston_characteristic(r, q, kappa, eta, theta, rho, sigma0, S0, u, T)
    rho = heston_char / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)

    if integration_rule == 0:
        a = np.fft.ifft(rho * np.exp(1j * v * b) * eta_grid, N)

    elif integration_rule == 1:
        simpson_1 = 1 / 3
        simpson = ((3 + (-1) ** np.arange(2, N + 1)) / 3)
        simpson_int = np.concatenate(([simpson_1], simpson))
        a = np.fft.ifft(rho * np.exp(1j * v * b) * eta_grid * simpson_int, N)

    CallPrices = (1 / np.pi) * np.exp(-alpha * k) * np.real(a)

    # Interpolate to find C(K,T)
    KK = np.exp(k)
    tck = splrep(KK, CallPrices)
    out = splev(K, tck)

    if option_type == 1:
        out = out + K * np.exp(-r * T) - np.exp(-q * T) * S0

    return out

# def construct_rho(char_f, eta_grid):
#     """
#     Construct a vector of rho(vj) where vj is a vector of size N
#     """
#     for j in range(1,N):
#         vj = (j-1)*eta_grid
#         rho_j = (np.exp(-r*t)*char_f((vj - (alpha +1 )*1j),t))/(alpha**2 + alpha - vj**2 + 1j*(2*alpha + 1)*vj)



#     return rho

# def CarrMadan(char_f, eta_grid):
    # N=4096
    # eta_grid = 0.25
    # alpha = 1.5



    # return grid_prices

