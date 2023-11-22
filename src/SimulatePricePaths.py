import numpy as np
import matplotlib.pyplot as plt

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

def gen_standard_normal(n_rows, n_columns):

    return np.random.standard_normal(size=(n_rows, n_columns))

def gen_corr_standard_normal():
    return

def gen_poisson(lam, n_rows,n_columns):

    return np.random.poisson(lam, size=(n_rows, n_columns))

def gen_negative_exponential(scale, n_rows,n_columns):

    return np.random.exponential(scale=scale, size=(n_rows,n_columns))

def gen_normal(mu,sigma,n_rows, n_columns):
    
    return np.random.normal(loc=mu, scale=sigma, size=(n_rows, n_columns))


## 1. Jump dynamics
# - Simulate lambda
# - Simulate jump processd

def simulate_jump_lambda_paths(m, num_paths, num_timesteps, tau, jump_params):
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

    [alpha,delta,beta,sigma,lambda_bar] = jump_params

    # populate random increments
    Epsilon = [np.random.normal(beta[i],sigma[i],(num_paths,num_timesteps)) for i in range(m)] # or gen_negative_exponential()

    # output variable initialisation: m matrices for both J and Lambda
    dJ = [np.zeros((num_paths,num_timesteps+1)) for i in range(m)]
    Lambda = [np.zeros((num_paths,num_timesteps+1)) for i in range(m)]

    #populate Lambda and dJ with first values
    for i in range(m):
        Lambda[i][:,0] = lambda_bar[i]
        dJ[i][:,0] = 0

    # parameters
    dt = tau/num_timesteps

    # SDE numerical implementation (still need to add iteration over assets i)
    for t in range(1,num_timesteps+1): # iterate over timesteps
        
        SumDelta = [np.zeros(num_paths) for i in range(m)]

        for i in range(m): # iterate over all assets
            # Get intermediate sum of delta*dJ called 'SumDelta'
            deltai_vec = np.array([delta[i,j] for j in range(m)])

            # Perform element-wise multiplication
            for p in range(num_paths):
                dJi_vec = np.array([dJ[i][p,t-1] for i in range(m)])
                SumDelta[i][p] = np.dot(deltai_vec,dJi_vec)
        
            # Compute Lambda timestep  
            Lambda[i][:,t] = np.maximum(0, Lambda[i][:, t-1] + alpha[i]*(lambda_bar[i]-Lambda[i][:, t-1])*dt + SumDelta[i])

            # Generate row vector of Poisson draws with intensity equal to corresponding Lambda[t]
            k = [np.random.poisson(Lambda[i][j,t-1]) for j in range(num_paths)]

            # Compute dJ 
            dJ[i][:,t] =  Epsilon[i][:,t-1]*k

    return Lambda, dJ

def test_dimensions(Lambda, dJ):
    m = len(Lambda)
    num_timesteps = Lambda[0].shape[1] - 1  # Subtract 1 for initial time step

    print("Dimensions:")
    for i in range(m):
        print(f"Lambda[{i}]: {Lambda[i].shape}")
        print(f"dJ[{i}]: {dJ[i].shape}")

def plot_paths(Lambda, dJ, num_paths, num_timesteps):
    m = len(Lambda)

    # Plot Lambda paths
    plt.figure(figsize=(12, 6))
    for i in range(m):
        plt.subplot(2, m, i + 1)
        plt.plot(Lambda[i][:, 1:])
        plt.title(f"Lambda[{i}] Paths")
        plt.xlabel("Time")
        plt.ylabel("Lambda")
        plt.grid(True)

    # Plot dJ paths
    for i in range(m):
        plt.subplot(2, m, m + i + 1)
        plt.plot(dJ[i][:, 1:])
        plt.title(f"dJ[{i}] Paths")
        plt.xlabel("Time")
        plt.ylabel("dJ")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

#-------- RUN THIS-------
# # Test code for simulate_jump_lambda
# alpha = np.array([0.5, 0.5]) 
# beta = np.array([0.1 , 0.1]) 
# sigma = np.array([0.2 , 0.2]) 
# delta = np.array([[0.1, 0.2], [0.2, 0.1]]) 
# lambda_bar = np.array([0.005, 0.005]) 
# params = [alpha,delta,beta,sigma,lambda_bar] 
# m = len(alpha)

# num_paths = 100
# num_timesteps = 1000
# tau = 1

# Lambda, dJ = simulate_jump_lambda_paths(m,num_paths,num_timesteps,tau,params)

# test_dimensions(Lambda,dJ)
# plot_paths(Lambda,dJ,num_paths,num_timesteps)
# --------------------------

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