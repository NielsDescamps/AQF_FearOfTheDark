import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import get_expiry_dates
import get_maturities
import load_data
import Heston_FFT
import APE

# Import options data from Yahoo Finance
ticker = 'SEDG'
save_dir = 'data_03_05_2023'

# Transform expiration dates into maturities T
date_of_retrieval = '2023-05-03'
expiry_dates = get_expiry_dates()  # Replace with your function to get expiration dates
get_maturities(expiry_dates, date_of_retrieval)  # Replace with your function to calculate maturities

# 1. Import the data needed for calculations
exp_date = '2023-05-12'
exp_date_name = '2023_05_12'
option_type = 0  # call or put
data = load_data(save_dir, exp_date, option_type)  # Replace with your data import function

# 2. Assign input variables
r = 0.053
q = 0
S0 = 264.55
integration_rule = 1
T = get_maturities(exp_date, date_of_retrieval)  # returns the maturity in terms of years
K = data['strike']

last_price = data['lastPrice']
midquote = (data['ask'] + data['bid']) / 2

# Write function with variables to optimize: kappa, eta, theta, rho, sigma0
heston_price1 = np.zeros(len(K))
heston_price2 = np.zeros(len(K))

options = {
    'maxfun': 10000,
    'maxiter': 5000,
    'xtol': 1e-9,
    'ftol': 1e-9,
}
lb = [0, 0, 0, -1, 0]
ub = [1, 1, 1, 0, 10]
x0 = [0.5, 0.05, 0.21, -0.75, 0.22]
A = []
b = []
Aeq = []
beq = []

# [results, heston_price] = calibrate_Heston_single(K, T, S0, r, q, type, integration_rule, x0, lb, ub, options, midquote)

# # Calibrate the Heston model for all K, T separately
# x01 = x0
# x02 = x0
# for i in range(len(K)):
#     fun1 = lambda params: sqrd_diff(params[0], params[1], params[2], params[3], params[4], K[i], T, S0, r, q, option_type, integration_rule, last_price[i])
#     fun2 = lambda params: sqrd_diff(params[0], params[1], params[2], params[3], params[4], K[i], T, S0, r, q, option_type, integration_rule, midquote[i])
#
#     result1 = minimize(fun1, x01, bounds=list(zip(lb, ub)), options=options)
#     result2 = minimize(fun2, x02, bounds=list(zip(lb, ub)), options=options)
#
#     heston_price1[i] = Heston_FFT(result1.x[0], result1.x[1], result1.x[2], result1.x[3], result1.x[4], K[i], T, S0, r, q, option_type, integration_rule)
#     heston_price2[i] = Heston_FFT(result2.x[0], result2.x[1], result2.x[2], result2.x[3], result2.x[4], K[i], T, S0, r, q, option_type, integration_rule)
#
#     # x0 for the next iteration: use parameters of the previous iteration to speed up calculations
#     x01 = result1.x
#     x02 = result2.x

# Calibrate over all strikes for each maturity T
fun3 = lambda params: APE(params[0], params[1], params[2], params[3], params[4], K, T, S0, r, q, option_type, integration_rule, last_price)
fun4 = lambda params: APE(params[0], params[1], params[2], params[3], params[4], K, T, S0, r, q, option_type, integration_rule, midquote)
result_opt3 = minimize(fun3, x0, bounds=list(zip(lb, ub)), options=options)
result_opt4 = minimize(fun4, x0, bounds=list(zip(lb, ub)), options=options)

cali_heston_price3 = np.zeros(len(K))
cali_heston_price4 = np.zeros(len(K))
for i in range(len(K)):
    cali_heston_price3[i] = Heston_FFT(result_opt3.x[0], result_opt3.x[1], result_opt3.x[2], result_opt3.x[3], result_opt3.x[4], K[i], T, S0, r, q, option_type, integration_rule)
    cali_heston_price4[i] = Heston_FFT(result_opt4.x[0], result_opt4.x[1], result_opt4.x[2], result_opt4.x[3], result_opt4.x[4], K[i], T, S0, r, q, option_type, integration_rule)

plt.figure()
# plt.plot(K, heston_price1, 'r*', linewidth=1.1)
# plt.plot(K, cali_heston_price3, 'ro', linewidth=1.1)
plt.plot(K, last_price, 'b+', linewidth=1.1)
plt.xlabel('K')
plt.ylabel('price')
plt.title('Calibration: Heston - Last price')
plt.legend(['Heston', 'Market'])
file_name = f'plots/cali_LP_APE_{exp_date_name}'
plt.savefig(file_name + '.png')

plt.figure()
# plt.plot(K, heston_price2, 'r*', linewidth=1.1)
# plt.plot(K, cali_heston_price4, 'ro', linewidth=1.1)
plt.plot(K, midquote, 'b+', linewidth=1.1)
plt.xlabel('K')
plt.ylabel('price')
plt.title('Calibration: Heston - Midquote')
plt.legend(['Heston', 'Market'])
file_name = f'plots/cali_MQ_APE_{exp_date_name}'
plt.savefig(file_name + '.png')