import numpy as np

def heston_characteristic(kappa, eta, theta, rho, sigma0, r, q, S0, t, u):
    d = np.sqrt((rho*theta*u*1j-kappa)**2-theta**2*(-1j*u-u**2))
    g = (kappa-rho*theta*u*1j-d)/(kappa-rho*theta*u*1j+d)
    p1 = 1j*u*(np.log(S0)+(r-q)*t)
    p2 =  eta*kappa *theta**(-2)*((kappa-rho*theta*u*1j-d)*t - 2*np.log((1-g*np.exp(-d*t))/(1-g)))
    p3 = sigma0**2*theta**(-2)*(kappa-rho*theta*u*1j - d)*(1- np.exp(-d*t))/(1-g*np.exp(-d*t))
    out = np.exp(p1)*np.exp(p2)*np.exp(p3)
    return out