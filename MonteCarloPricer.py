import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.Heston_FFT import Heston_FFT
from src.sse import sse
from src.load_data import load_data
from src.get_maturities import get_maturities


# Import options data from Yahoo Finance
ticker = 'SEDG'
save_dir = 'data_10_27_2023'


# 1. Import the data needed for calculations
date_of_retrieval = '2023-10-27'
exp_date = '2023-11-03'
option_type = 0  # call (0) or put (1)
data = load_data(save_dir, exp_date, option_type)  # Replace with your data import function

r = 0.053
q = 0
S0 = 83.2
method = 0
integration_rule = 1
T = get_maturities([exp_date], date_of_retrieval)[0]  # Replace with your function for getting maturities
K = data['strike']

midquote = (data['ask'] + data['bid']) / 2

# 2. Calibrate the Heston model
options = {
    'maxfun': 10000,
    'maxiter': 5000,
    'ftol': 1e-9,
}
#Boundry fot theta as its devided on later
lb = [0, 0, 0.1, -1, 0]
ub = [1, 1, 1, 0, 10]
#bounds = [(0,1), (0,1), (0,1), (-1,0), (0,10)]
x0 = [0.5, 0.05, 0.2, -0.75, 0.22]
A = []
b = []
Aeq = []
beq = []

def fun(params):
    return sse(params[0], params[1], params[2], params[3], params[4], K, T, S0, r, q, option_type, integration_rule, midquote)

result_opt = minimize(fun, x0, bounds=list(zip(lb, ub)), options=options, tol=1e-9)


kappa, eta, theta, rho, v0 = result_opt.x

# 3. Initialize matrices
m = 10000
n = int(round(365 * T))
dt = T / n

S = np.zeros((m, n + 1))
v = np.zeros((m, n + 1))
S[:, 0] = S0
v[:, 0] = v0

eps = np.random.normal(0, 1, size=(m, n))
epsS = np.random.normal(0, 1, size=(m, n))
eps1 = eps
eps2 = rho * eps + np.sqrt(1 - rho**2) * epsS

# 4. Simulate price paths
if method == 1:  # Euler
    for j in range(1, n + 1):
        S[:, j] = S[:, j - 1] * (1 + (r - q) * dt + np.sqrt(v[:, j - 1]) * np.sqrt(dt) * eps1[:, j - 1])
        v[:, j] = np.abs(v[:, j - 1] + (kappa * (eta - v[:, j - 1])) * dt + theta * np.sqrt(v[:, j - 1]) * np.sqrt(dt) * eps2[:, j - 1])
else:  # Milstein
    for j in range(1, n + 1):
        S[:, j] = S[:, j - 1] * (1 + (r - q) * dt + np.sqrt(v[:, j - 1]) * np.sqrt(dt) * eps1[:, j - 1])
        v[:, j] = np.abs(v[:, j - 1] + (kappa * (eta - v[:, j - 1]) - theta**2 / 4) * dt + theta * np.sqrt(v[:, j - 1]) * np.sqrt(dt) * eps2[:, j - 1] + theta**2 * dt * (eps2[:, j - 1]**2) / 4)

# 5. Plotting price paths
plt.figure()
for i in range(m):
    plt.plot(range(n + 1), S[i, :])
plt.xlabel('Timestep')
plt.ylabel('St')
plt.title('Price paths')
plt.savefig('/generated_plots/Price_paths_MC_1205.png')
plt.show()

# 5. Check: compare prices of European options
EC_MC = np.zeros(len(K))
EC_FFT = np.zeros(len(K))

for i in range(len(K)):
    EC_MC[i] = np.exp(-r * T) * np.mean(np.maximum(S[:, -1] - K[i], 0))
    EC_FFT[i] = Heston_FFT(kappa, eta, theta, rho, np.sqrt(v0), K[i], T, S0, r, q, option_type, integration_rule)

plt.figure()
plt.plot(K, EC_MC, 'r*', linewidth=1.1)
plt.plot(K, EC_FFT, 'g+', linewidth=1.1)
plt.plot(K, midquote, 'o', linewidth=1.1)
plt.xlabel('K')
plt.ylabel('price')
plt.title('Comparison MC and FFT - Midquote')
plt.legend(['MC', 'FFT', 'Market'])
plt.savefig('../generated_plots/ComparisonMCFFT_T1205.png')
plt.show()

# 6. Compute price of CBC for different B(=C) (continued)
# - Zero strike call
ZSC_MC = np.exp(-r * T) * np.mean(S[:, -1])

start = 60
fin = 100
stepsize = 5
steps = int((fin - start) / stepsize) + 1
B = np.linspace(start, fin, steps)
H = 77  # barrier value
C = B

DOBP = np.zeros(len(B))
EC_MC = np.zeros(len(C))
CBC = np.zeros(len(C))

for i in range(len(B)):
    # Down and out barrier option with strike B
    DOBP_dp = np.exp(-r * T) * np.maximum((np.min(S, axis=1) - H) / np.abs(np.min(S, axis=1) - H), 0) * np.maximum(B[i] - S[:, -1], 0)
    DOBP[i] = np.mean(DOBP_dp)

    # European call with strike C
    EC_MC[i] = np.exp(-r * T) * np.mean(np.maximum(S[:, -1] - C[i], 0))

    # Total price of CBC
    CBC[i] = ZSC_MC + DOBP[i] - EC_MC[i]

plt.figure()
plt.plot(B, CBC, 'mo', linewidth=1.1)
plt.plot(B, DOBP, 'b+', linewidth=1.1)
plt.plot(B, EC_MC, 'r+', linewidth=1.1)
plt.xlabel('B')
plt.ylabel('price')
plt.title('CBC price decomposition for different B')
plt.legend(['CBC', 'DOBP', 'EC(MC)'])
plt.savefig('../generated_plots/CBC_decomposition_H200_T1205_Bvar2.png')
plt.show()

# 7. Compute price of CBC for different H
# - Zero strike call
ZSC_MC = np.exp(-r * T) * np.mean(S[:, -1])

start = 60
fin = 100
stepsize = 5
steps = int((fin - start) / stepsize) + 1
B = 110
H = np.linspace(start, fin, steps)  # barrier value
C = B

DOBP = np.zeros(len(H))
CBC = np.zeros(len(H))
EC_MC = np.zeros(len(H))

for i in range(len(H)):
    # Down and out barrier option with strike B
    DOBP_dp = np.exp(-r * T) * np.maximum((np.min(S, axis=1) - H[i]) / np.abs(np.min(S, axis=1) - H[i]), 0) * np.maximum(B - S[:, -1], 0)
    DOBP[i] = np.mean(DOBP_dp)

    # European call with strike C
    EC_MC[i] = np.exp(-r * T) * np.mean(np.maximum(S[:, -1] - C, 0))

    # Total price of CBC
    CBC[i] = ZSC_MC + DOBP[i] - EC_MC[i]

plt.figure()
plt.plot(H, CBC, 'mo', linewidth=1.1)
plt.plot(H, DOBP, 'b+', linewidth=1.1)
plt.plot(H, EC_MC, 'r+', linewidth=1.1)
plt.xlabel('H')
plt.ylabel('price')
plt.title('CBC price decomposition for different H')
plt.legend(['CBC', 'DOBP', 'EC(MC)'])
plt.savefig('../generated_plots/CBC_decomposition_B300_T2605_Hvar.png')
plt.show()