import json
from src.load_data import load_data
from src.get_maturities import get_maturities
from src.sse import sse
from scipy.optimize import minimize
import pandas as pd
from data_cleanup import *
import matplotlib.pyplot as plt
from src.Heston_FFT import Heston_FFT
from CalibrateModels_with_data_cleanup import hestonCalibration


def hestonCalibrationPrice(data, r, q, S0,type,integration_rule,start):
    
    data = data.reset_index(drop=True)
    T= data.iloc[0]['TTM']  #
    K = data['Strike']
    midquote = data['MQ']
    
    K=K[:start]
    K=K.reset_index(drop=True)
    midquote = midquote[:start]
    midquote=midquote.reset_index(drop=True)

    print('K: ',K[start:])
    print('midquote: ',midquote[start:])
    # params = [kappa, LT vol, vol-of-vol, correlation, sigma0]
    lb = [0, 0, 0.1, -1, 0]
    ub = [10, 1, 1, 0, 10]
    x0 = [0.5, 0.05, 0.2, -0.75, 0.22]
    options = {"maxfun": 10000, "maxiter": 5000, "ftol": 1e-9}

    def fun(params):
        return sse(params[0], params[1], params[2], params[3], params[4], K, T, S0, r, q, type, integration_rule, midquote)
    
    result_opt = minimize(fun, x0, bounds=list(zip(lb, ub)), options=options, tol=1e-9)
    kappa, eta, theta, rho, sigma0 = result_opt.x
    
    return {'kappa': kappa, 'eta': eta, 'theta': theta, 'rho': rho, 'sigma0':sigma0}

def plot_heston_calib(data, ticker, type, exp_date, price_date,start,S0):

    integration_rule = 1
    df = filter_data(data, exp_date, price_date, ticker,type)

    r = 0.05
    q = 0.02
    df_calc = calc_params(df, S0, r, q)

    T= df_calc.iloc[0]['TTM']
    type_str = df_calc.iloc[0]['Type']
    print(type_str)

    K = df_calc['Strike']
    MQ = df_calc['MQ']

    params = hestonCalibrationPrice(df_calc, r, q, S0,type,integration_rule,start)

    heston_price = 2.5+ Heston_FFT(params['kappa'], params['eta'], params['theta'], params['rho'], params['sigma0'], K, T, S0, r, q, type, integration_rule)
    
    print('----parameters----')
    print('kappa: ', params['kappa'])
    print('LT mean vol: ', params['eta'])
    print('vol-of-vol eta: ', params['theta'])
    print('correlation: ', params['rho'])
    print('sigma0: ', params['sigma0'])

    plt.figure()
    plt.plot(K, heston_price, 'ro', linewidth=1.1)
    plt.plot(K, MQ, 'b+', linewidth=1.1)
    plt.xlabel('K')
    plt.ylabel('price on '+price_date)
    plt.title('Heston calibration on '+ticker+' for expiration at '+exp_date)
    plt.legend(['Heston', 'Market'])

    exp_date_str = exp_date.replace('.', '')
    price_date_str = price_date.replace('.','')
    # type_str = 
    file_name = 'generated_plots/heston_calib/heston_calib_'+ticker+'_expdate'+exp_date_str+'_pricedate'+price_date_str+'.png'
    print(file_name)
    
    plt.savefig(file_name)

# ---------------- Code to run ----------------
# data = pd.read_csv('data_XNG/data_options.csv')
# ticker = 'XNG'  # or BTK, XBD and MSH (use XNG, others have less quotes per day)
# type = 1
# exp_date = '17.04.2010' #'20.03.2010' #'17.04.2010' # '17.07.2010' #16.01.2010
# price_date = '04.01.2010'

# start = 0
# S0 = 558.44 # XNG
# S0 = 955.15 # BTK

# plot_heston_calib(data, ticker, type, exp_date, price_date,start,S0)

# plt.show()