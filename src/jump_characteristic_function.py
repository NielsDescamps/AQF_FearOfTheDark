from src.RungeKutta import ode_system_Gaussian
from src.RungeKutta import ode_system_Exponential
from src.RungeKutta import runge_kutta_4th_order_finalbc
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.integrate as int
import matplotlib.pyplot as plt
from src.heston_characteristic import heston_characteristic
import scipy
from src.BlackScholes import calc_implied_vol


def mutualjump_characteristic_function(params,lambda_zero, t, T, h, u, index, jump_distribution):
    """
    Returns values for the characteristic function of the m-dimensional Hawkes process
    Parameters:
    - u: characteristic function variable
    - T: maturity date
    - t: time of evaluation
    - h: time step for the evaluation of the set of ODEs
    - index: asset index we are assessing (1, 2, 3 or 4)
    - params: model parameters (alpha, delta, beta, sigma, lambda_bar)

    Output:
    - value of the characteristic function phi2_index(u,t,T)
    """

    m = len(params[0])

    # 1. Construct Yt
    if t==0:
        # Assuming t=0 at time of evaluation Yt can be simplified strongly
        Yt = np.concatenate((lambda_zero,np.zeros(m)),axis=0)
    
    else:
        ## If this would be implemented probalby do it autoregressive
        print("t not 0 is not implemented")
        return KeyError
    

    # 2. Calculate a(v,t,T) and b(v,t,T)
    b_finalBCs = np.zeros(2*m,dtype=np.complex128)
    ## I dont think it should be an i here, as that refers tho the joint dist in paper, but u alone is real
    ## I also think not only the index should be u but the whole vector
    b_finalBCs[m+index] = u*1j
    a_finalBCs = np.array([0])
    final_conditions= np.concatenate((b_finalBCs,a_finalBCs))

    
    t_span = (t,T)

    if jump_distribution == "Exponential":
        ode_system = ode_system_Exponential
    elif jump_distribution == "Gaussian":
        ode_system = ode_system_Gaussian

    [t_values, y_values] = runge_kutta_4th_order_finalbc(ode_system,final_conditions, t_span, h, params)
    at = y_values[:,2*m]
    bt = y_values[:,:2*m]

    a= at[0]

    b= bt[0,:]

    """
    ##Solver with inherit package, may be quicker if it is optimized?
    def tau_ode(t,y):
        return -ode_system(t,y,params)
    ##Method with solve_ivp make variable change to tau = T-t -> d_tau = -dt for initial cond
    solution = int.solve_ivp(fun = tau_ode, t_span= t_span, y0 = final_conditions, method='RK45', t_eval=[T-t])
    Yt = solution['y']
    a = Yt[0]
    b = Yt[1:]
    """
    PHI = np.exp(a + np.dot(Yt,b)) 
    
    return PHI
def joint_characteristic_function(u, t, T, kappa, eta, theta, rho, sigma0, r, q, S0, params, lambda_zero, h, index, jump_distribution):
    # This is under the presumption that r and q are constant
    return np.exp(1j*u*T*(r-q))*heston_characteristic(kappa, eta, theta, rho, sigma0, r, q, S0, T, u) * mutualjump_characteristic_function(params,lambda_zero, t, T, h, u, index, jump_distribution)
def calc_priceFFT(characteristic_function, model_params, K, T, r, q, S0, type, integration_rule):

    """
    Calculate the european option price using the Carr Madan option price for different characteristic functions

    Parameters:
    - characteristic_function(model_params, r,q,S0,T,u): a function object that takes the model parameters as inputs (already evaluated or not)
    - model_params: the parameters of the model (e.g Heston: kappa, eta, theta, rho and sigma0)
    - option_params: parameters that define the european option (K, T, S0, r, q, type)
    - integration rule: 0 or 1 (rectangular=0, simpson's trapezoidal rule=1)

    """

    # define parameters
    N = 4096
    alpha = 1.5
    eta_grid = 0.25
    lambda_val = 2 * np.pi / (N * eta_grid)
    b = lambda_val * N / 2

    # define grid of log-strikes
    k = np.arange(-b, b, lambda_val)

    # compute rho
    v = np.arange(0, N * eta_grid, eta_grid)
    u = v - (alpha + 1) * 1j
    rho_val = np.exp(-r * T) * characteristic_function(model_params, r, q, S0, T, u) / (
                alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v)

    # compute integration of CarrMadan formula (a) with FFT
    if integration_rule == 0:
        a = np.real(np.fft.fft(rho_val * np.exp(1j * v * b) * eta_grid, N))
    elif integration_rule == 1:
        simpson_1 = 1 / 3
        simpson = (3 + (-1) ** np.arange(2, N + 1)) / 3
        simpson_int = np.concatenate(([simpson_1], simpson))
        a = np.real(np.fft.fft(rho_val * np.exp(1j * v * b) * eta_grid * simpson_int, N))

    # compute the prices for the grid of log-strikes
    CallPrices = (1 / np.pi) * np.exp(-alpha * k) * a


    # find C(K,T) for the given strike value K
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



