import json
from src.load_data import load_data
from src.get_maturities import get_maturities
from src.sse import sse
from scipy.optimize import minimize
import pandas as pd
from data_cleanup import *
def hestonCalibration(data, r, q, S0):
    #Only calibrate on the call prices
    ##TODO This gives very strange calibration values but dont find the error
    data.drop(data[data['Type'] != 'C'].index, inplace=True)
    df_calc.dropna()
    data = data.reset_index(drop=True)
    T = data['TTM'][0]  # Replace with your function for getting maturities
    K = data['Strike']
    midquote = data['MQ']


    # Boundry fot theta as its devided on later
    lb = [0, 0, 0.1, -1, 0]
    ub = [1, 1, 1, 0, 10]
    # bounds = [(0,1), (0,1), (0,1), (-1,0), (0,10)]
    x0 = [0.5, 0.05, 0.2, -0.75, 0.22]
    options = {"maxfun": 10000, "maxiter": 5000, "ftol": 1e-9}
    def fun(params):
        return sse(params[0], params[1], params[2], params[3], params[4], K, T, S0, r
                   , q, 0, 1, midquote)

    result_opt = minimize(fun, x0, bounds=list(zip(lb, ub)), options=options, tol=1e-9)

    kappa, eta, theta, rho, v0 = result_opt.x

    calib_file = "./models_config_calibration/heston_calib.json"
    with open(calib_file, "w") as outfile:
        json.dump({'kappa': kappa, 'eta': eta, 'theta': theta, 'rho': rho, 'v0':v0}, outfile)
    #TODO save the parameters to a json file as ex, (then no need to always recalibrate)
    return {'kappa': kappa, 'eta': eta, 'theta': theta, 'rho': rho, 'v0':v0}

def jumpProcessCalibration(params):
    #Calib procedure
    # {"kappa": 0.5036229171392832, "eta": 0.06019028366975756, "theta": 0.1, "rho": 0.0, "v0": 0.5232862836528391}
    return 0

def initialize_calib():
    data = pd.read_csv('data_XNG/data_options.csv')

    exp_date = '16.01.2010'  # or  '17.04.2010'
    price_date = '04.01.2010'
    # On pricing day:
    Open = 539.61
    High = 558.48
    Low = 539.61
    Close = 558.44
    S0 = (Open + Close) / 2

    ticker = 'XNG'  # or BTK, XBD and MSH (use XNG, others have less quotes per day)
    df = filter_data(data, exp_date, price_date, ticker)

    r = 0.05
    q = 0.02
    df_calc = calc_params(df, S0, r, q)
    hestonCalibration(df_calc, r, q, S0)

    return 0

##TODO:

initialize_calib()
#Write parameters to file:

