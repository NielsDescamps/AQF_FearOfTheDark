import numpy as np

# This file contains functions to generate price paths for the different dynamics

## 0. Functions to generate random increments
# Diffusion process
# - standard normal generator 
# - correlated standard normal generator
# Jump frequency 
# - poisson generator
# Jump size 
# - negative exponential generator
# - location scale normal generator

def gen_standard_normal():
    return

def gen_corr_standard_normal():
    return

def gen_poisson():
    return

def gen_negative_exponential():
    return 

def gen_normal():
    return 


## 1. Jump dynamics
# - Simulate lambda
# - Simulate jump processd

def simulate_jump_lambda_paths(m, num_paths, num_timesteps, tau):
    """
    Simulate paths for lambda and the pure jump process for a series of assets
    
    Parameters:
    - num_paths: number of paths to generate 
    - num_timesteps: number of timesteps 
    - tau: time until maturity expressed in (?)

    Output:
    - Jump process paths: matrix with dimensions (num_paths, num_timesteps)
    - Lambda process paths: matrix with dimensions(num_paths, num_timesteps)
    """

    # populate random increments
    Epsilon = gen_normal() # or gen_negative_exponential()

    # output variable initialisation: m matrices for both J and Lambda
    dJ = [np.zeros((num_paths,num_timesteps+1)) for i in range(m)]
    Lambda = [np.zeros((num_paths,num_timesteps+1)) for i in range(m)]

    # parameters
    dt = tau/num_timesteps

    # SDE numerical implementation (still need to add iteration over assets i)
    for t in range(1,num_timesteps+1): # iterate over timesteps
        
        SumDelta = np.zeros((num_paths, m))

        for i in range(m): # iterate over all assets
            
            SumDelta[:, i] = np.sum(delta[i, :] * dJ[i][:, t-1], axis=1) # Compute SumDelta for each asset i
            
            Lambda[i][:,t] = Lambda[i][:,t-1] + alpha[i]*(lambda_bar[i]-Lambda[i][:,t-1])*dt + SumDelta

            dJ[i][:,t] =  Epsilon[i][:,t-1]*k[:,t-1]

        # TO BE FINISHED
    return

# 2. Full (Heston) dynamics
# - Simulate log price process 
# - Simulate volatility 

def simulate_price_vol_paths(num_paths, num_timesteps, tau):

    dt = tau/num_timesteps

    m = 100000

    S = np.zeros((num_paths, num_timesteps+1))
    v = np.zeros((num_paths, num_timesteps+1))
    S[:, 0] = S0
    v[:, 0] = v0

    # generate correlated random numbers
    eps = np.random.normal(0, 1, size=(m, n))
    epsS = np.random.normal(0, 1, size=(m, n))
    eps1 = eps
    eps2 = rho * eps + np.sqrt(1 - rho**2) * epsS

    if method == 1:  # Euler
        for j in range(1, n + 1):
            S[:, j] = S[:, j - 1] * (1 + (r - q) * dt + np.sqrt(v[:, j - 1]) * eps1[:, j - 1])
            v[:, j] = np.abs(v[:, j - 1] + (kappa * (eta - v[:, j - 1])) * dt + theta * np.sqrt(v[:, j - 1]) * eps2[:, j - 1])

    else:  # Milstein
        for j in range(1, n + 1):
            S[:, j] = S[:, j - 1] * (1 + (r - q) * dt + np.sqrt(v[:, j - 1]) * eps1[:, j - 1])
            v[:, j] = np.abs(v[:, j - 1] + (kappa * (eta - v[:, j - 1]) - theta**2 / 4) * dt +
                            theta * np.sqrt(v[:, j - 1]) * eps2[:, j - 1] +
                            theta**2 * dt * (eps2[:, j - 1]**2) / 4)


    return S, v