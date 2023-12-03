import numpy as np
import matplotlib.pyplot as plt
from generate_random import generate_randomNegExp
from generate_random import plot_neg_exp_density

# functions to simulate the asset dynamics
def construct_J_profile(dJ):
    """
    Construct the profile of J (cumulative sum of previous dJ).

    Parameters:
    - dJ: Matrix containing the increments dJ for m assets.

    Returns:
    - J: Matrix representing the cumulative sum of dJ.
    """
    m = len(dJ)

    # Initialize J matrix with zeros
    J = np.array([np.zeros_like(dJ[i]) for i in range(m)])

    # Compute cumulative sum along the time axis
    for i in range(m):
        J[i][:, 1:] = np.array(np.cumsum(dJ[i][:, 1:], axis=1))

    return J

def simulate_jump_lambda_paths(m, num_paths, num_timesteps, tau, jump_params,distribution,distr_params):
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

    [alpha,delta,lambda_bar] = jump_params

    if distribution=='Gaussian':
        [beta, sigma] = distr_params
        Epsilon = [np.random.normal(beta[i],sigma[i],(num_paths,num_timesteps)) for i in range(m)] # or gen_negative_exponential()

    elif distribution=='Exponential':
        [gamma] = distr_params
        Epsilon = [generate_randomNegExp(gamma[i],num_paths,num_timesteps) for i in range(m)]
        # Epsilon = [np.random.exponential(gamma[i],(num_paths,num_timesteps)) for i in range(m)]

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
            # Generate row vector of Poisson draws with intensity equal to corresponding Lambda[t]
            k = np.random.poisson(Lambda[i][:,t-1]*dt)

            # Compute dJ 
            dJ[i][:,t] =  Epsilon[i][:,t-1]*k

            # Get intermediate sum of delta*dJ called 'SumDelta'
            deltai_vec = np.array([delta[i,j] for j in range(m)])

            # Perform element-wise multiplication
            for p in range(num_paths):
                dJi_vec = np.array([dJ[i][p,t-1] for i in range(m)])
                SumDelta[i][p] = np.dot(deltai_vec,dJi_vec)
        
            # Compute Lambda timestep  
            Lambda[i][:,t] = np.maximum(0, Lambda[i][:, t-1] + alpha[i]*(lambda_bar[i]-Lambda[i][:, t-1])*dt + SumDelta[i])

    LT_mean = [(alpha[i]*lambda_bar[i])/(alpha[i] - sum([delta[i,j]*beta[j] for j in range(m)]))for i in range(m)]

    J = construct_J_profile(dJ)

    return Lambda, dJ, J, LT_mean

def simulate_heston_paths(num_paths, num_timesteps, tau, r, q, S0, heston_params, discretisation_scheme,dJ):

    [kappa, eta, theta, rho, sigma0]  = heston_params

    dt = tau/num_timesteps

    S = np.zeros((num_paths, num_timesteps+1))
    V = np.zeros((num_paths, num_timesteps+1))
    S[:, 0] = S0
    V[:, 0] = sigma0

    # generate correlated random numbers
    eps = np.random.normal(0, 1, size=(num_paths, num_timesteps))
    epsS = np.random.normal(0, 1, size=(num_paths, num_timesteps))
    eps1 = eps
    eps2 = rho * eps + np.sqrt(1 - rho**2) * epsS

    if discretisation_scheme == 0:  # Euler
        for j in range(1, num_timesteps + 1):
            S[:, j] = S[:, j - 1] * (1 + (r - q) * dt + np.sqrt(V[:, j - 1]*dt) * eps1[:, j - 1] + dJ[:,j-1])
            V[:, j] = np.abs(V[:, j - 1] + (kappa * (eta - V[:, j - 1])) * dt + theta * np.sqrt(V[:, j - 1]*dt) * eps2[:, j - 1])

    elif discretisation_scheme == 1: # Milstein
        for j in range(1, num_timesteps + 1):
            S[:, j] = S[:, j - 1] * (1 + (r - q) * dt + np.sqrt(V[:, j - 1]) * eps1[:, j - 1]*dt + dJ[:,j-1])
            V[:, j] = np.abs(V[:, j - 1] + (kappa * (eta - V[:, j - 1]) - theta**2 / 4) * dt +
                            theta * np.sqrt(V[:, j - 1]*dt) * eps2[:, j - 1] +
                            theta**2 * dt * (eps2[:, j - 1]**2) / 4)

    return S, V

def simulate_all_paths(m,num_paths, num_timesteps, tau, r, q ,S0, jump_params, distribution, distr_params,heston_params, discretisation_scheme):
    
    # calculate Lambda and dJ
    Lambda, dJ, J, LT_mean = simulate_jump_lambda_paths(m, num_paths, num_timesteps, tau, jump_params,distribution,distr_params)

    # calculate S and V for both assets
    S = []
    V = []
    for i in range(m):
        S_i, V_i = simulate_heston_paths(num_paths, num_timesteps, tau, r, q, S0, heston_params, discretisation_scheme,dJ[i])
        S.append(S_i)
        V.append(V_i)

    return Lambda, dJ, J, S, V

# functions to plot results 
def plot_heston_paths(S, V, num_paths, num_timesteps, LT_variance):
    # Plot Price Paths
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    for i in range(num_paths):
        plt.plot(S[i, :], label=f'Path {i + 1}')    
    plt.title('Price Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    # plt.legend()
    plt.grid(True)

    # Plot Volatility Paths
    plt.subplot(2, 1, 2)
    for i in range(num_paths):
        plt.plot(V[i, :], label=f'Path {i + 1}')
    plt.axhline(y=np.sqrt(LT_variance), color='red', linestyle='--', label=f'LT vol')
    plt.title('Volatility Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')
    # plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # plt.show()

def plot_lambda_dJ_paths(Lambda, dJ, num_paths, num_timesteps,LT_mean=None):
    m = len(Lambda)

    plt.figure(figsize=(18, 8))

    # Plot Lambda paths
    for i in range(m):
        plt.subplot(2, m, i + 1)
        for path in range(num_paths):
            plt.plot(Lambda[i][path, 1:], label=f'Path {path + 1}')
        plt.axhline(y=LT_mean[i], color='red', linestyle='--', label=f'LT mean = {LT_mean[i]}')
        plt.title(f"Lambda Paths for asset {i+1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Lambda")
        # plt.legend()
        plt.grid(True)

    # Plot dJ paths
    for i in range(m):
        plt.subplot(2, m, m + i + 1)
        for path in range(num_paths):
            plt.plot(dJ[i][path, 1:], label=f'Path {path + 1}')
        plt.title(f"dJ Paths for asset {i+1}")
        plt.xlabel("Time Steps")
        plt.ylabel("dJ")
        # plt.legend()
        plt.grid(True)

    plt.tight_layout()

    plt.show()

    # save_choice = input("Do you want to save the plot? (y/n): ").lower()
    
    # if save_choice == 'y':
    #     save_path = input("Enter the file name to save the plot (excluding extension, e.g., 'plot.png'): ")
    #     save_path = "figures/"+save_path+".png"
    #     plt.savefig(save_path)
    #     print(f"Plot saved at {save_path}")
    
    return

def plot_lambda_dJ_paths_single(Lambda, dJ, num_paths, num_timesteps):

    plt.figure(figsize=(18, 8))

    # Plot Lambda paths
    plt.subplot(2, 1, 1)
    for path in range(num_paths):
        plt.plot(Lambda[path, 1:], label=f'Path {path + 1}')
    plt.title("Lambda Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Lambda")
    # plt.legend()
    plt.grid(True)

    # Plot dJ paths
    plt.subplot(2, 1, 2)
    for path in range(num_paths):
        plt.plot(dJ[path, 1:], label=f'Path {path + 1}')
    plt.title("dJ Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("dJ")
    # plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # plt.show()

def plot_lambda_dJ_paths_combined(Lambda, dJ, num_paths, num_timesteps):

    plt.figure(figsize=(18, 8))

    # Plot Lambda and dJ paths on the same plot
    for path in range(num_paths):
        plt.plot(Lambda[path, 1:], label=f'Lambda {path + 1}', color='blue')
        plt.plot(dJ[path, 1:], label=f'dJ {path + 1}', color='orange', linestyle='dashed')

    plt.title("Lambda and dJ Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

def plot_lambda_price_vol_paths(Lambda, S, V,num_paths, num_timesteps):
    # Plot Price Paths
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    for j in range(num_paths):
        plt.plot(S[j, :], label=f'Path {j + 1}')
    plt.title('Price Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    # plt.legend()
    plt.grid(True)

    # Plot Volatility Paths
    plt.subplot(3, 1, 2)
    for j in range(num_paths):
        plt.plot(V[j, :], label=f'Path {j + 1}')
    plt.title('Volatility Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')
    # plt.legend()
    plt.grid(True)

    # Plot Volatility Paths
    plt.subplot(3, 1, 3)
    for j in range(num_paths):
        plt.plot(Lambda[j, :], label=f'Path {j + 1}')
    plt.title('Lambda Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Lambda')
    # plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # plt.show()

def plot_lambda_profiles_and_difference(Lambda, num_paths,  LT_mean, index1, index2):

    """
    Plot the profiles and the absolute difference between two assets in Lambda over different timesteps.

    Parameters:
    - Lambda: List containing matrices representing Lambda paths for different assets.
    - asset1: Index of the first asset.
    - asset2: Index of the second asset.
    """

    # Extract Lambda matrices for the selected assets
    asset1_lambda = Lambda[index1]
    asset2_lambda = Lambda[index2]

    # Compute the absolute difference between the two assets
    difference = np.abs(asset1_lambda[:,:] - asset2_lambda[:,:])
    relative_difference = asset2_lambda[0,:-1]/asset1_lambda[0,1:]

    # Plot the profiles and the difference in separate subplots
    plt.figure(1,figsize=(12, 8))

    # Plot Asset 1 Lambda Profile
    plt.subplot(2, 1, 1)
    for path in range(num_paths):
        plt.plot(Lambda[index1][path, 1:], label=f'Path {path + 1}')
    # plt.title(f"Asset {index1+1} Lambda Profile")
    plt.axhline(y=LT_mean[index1], color='red', linestyle='--', label=f'LT mean = {LT_mean[index1]}')
    plt.xlabel("Time Steps")
    plt.ylabel(f"Lambda asset {index1+1}")
    plt.grid(True)

    # Plot Asset 2 Lambda Profile
    plt.subplot(2, 1, 2)
    for path in range(num_paths):
        plt.plot(Lambda[index2][path, 1:], label=f'Path {path + 1}')
    plt.plot(asset2_lambda[:, 1:])
    plt.axhline(y=LT_mean[index2], color='red', linestyle='--', label=f'LT mean = {LT_mean[index2]}')
    # plt.title(f"Asset {index2+1} Lambda Profile")
    plt.xlabel("Time Steps")
    plt.ylabel(f"Lambda asset {index2+1}")
    plt.grid(True)

    # Plot difference between lambda and its long term mean
    plt.figure(2,figsize=(7,4))
    for path in range(num_paths):
        plt.plot(Lambda[index1][path, 1:]-LT_mean[index1])
    plt.title(f"Difference between lambda and long-term mean")
    plt.xlabel("Time steps")
    plt.ylabel("Absolute difference")
    plt.grid(True)

    # plt.figure(3,figsize=(7,4))
    # for path in range(num_paths):
    #     plt.plot(Lambda[index2][path, 1:]-LT_mean[index2])
    # plt.title(f"Difference between lambda and long-term mean of asset {index2+1}")
    # plt.xlabel("Time Steps")
    # plt.ylabel("Absolute Difference")
    # plt.grid(True)



    # plt.figure(3,figsize=(7,4))
    # plt.plot(relative_difference)
    # plt.title(f"relative difference between lambda of asset {index1+1} and {index2+1}")
    # plt.xlabel("Time Steps")
    # plt.ylabel("Relative  Difference")
    # plt.grid(True)

    plt.tight_layout()

# -----------------------------------------------#
m = 2
num_paths = 10
num_timesteps = 1000
tau = 1

# JUMP parameters
#  - GAUSSIAN 
alpha = np.array([36.6, 36.6])  # [36.6, 36.6]
lambda_bar = np.array([0.56, 0.56]) 
delta = np.array([[13.1, 1.6], [1.6, 13.1]]) # [13.1, 1.1], [1.6, 13.1]]
jump_params = [alpha,delta,lambda_bar]

distribution = 'Gaussian'
sigma = np.array([0.12 , 0.12]) 
beta = np.array([0.14 , 0.14]) 
distr_params = [beta,sigma]

LT_mean = [(alpha[i]*lambda_bar[i])/(alpha[i] - sum([delta[i,j]*beta[j] for j in range(m)]))for i in range(m)]


# - EXPONENTIAL 
# alpha = np.array([20.8, 20.8]) 
# lambda_bar = np.array([1.99, 1.99]) 
# delta = np.array([[-1, -18], [-18, -1]]) 
# jump_params = [alpha,delta,lambda_bar]

# distribution='Exponential'
# gamma = np.array([12.4, 12.4])
# distr_params= [gamma]
# plot_neg_exp_density(gamma[0])

# HESTON parameters
r = 0.05
q = 0.02
S0 = 100.0
kappa = 8.7 # mean reversion speed
eta = 0.015 # LT vol  0.015
theta = 0.013 # vol of vol 
rho = -0.49 #correlation 
v0 = 0.014
heston_params = [kappa, np.sqrt(eta), theta, rho, np.sqrt(v0)]
discretisation_scheme = 0

# -----------------Code to run ------------------ #
# Lambda, dJ, J, S, V = simulate_all_paths(m,num_paths, num_timesteps, tau, r, q ,S0, jump_params, distribution, distr_params,heston_params, discretisation_scheme)
# for i in range(m):
#   plot_lambda_price_vol_paths(Lambda[i],S[i],V[i],num_paths,num_timesteps)
# plot_heston_paths(S[0],V[0],num_paths,num_timesteps,eta)
# plot_lambda_dJ_paths(Lambda,dJ,num_paths,num_timesteps,LT_mean)
# plot_lambda_profiles_and_difference(Lambda,num_paths,LT_mean,0,1)

# plt.show()
