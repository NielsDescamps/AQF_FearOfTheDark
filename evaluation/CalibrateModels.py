import json
from src.load_data import load_data
from src.get_maturities import get_maturities
from src.sse import sse
from scipy.optimize import minimize
def hestonCalibration(data, conf):
    T = get_maturities([conf['exp_date']], conf['date_of_retrieval'])[0]  # Replace with your function for getting maturities
    K = data['strike']
    midquote = (data['ask'] + data['bid']) / 2

    # Boundry fot theta as its devided on later
    lb = [0, 0, 0.1, -1, 0]
    ub = [1, 1, 1, 0, 10]
    # bounds = [(0,1), (0,1), (0,1), (-1,0), (0,10)]
    x0 = [0.5, 0.05, 0.2, -0.75, 0.22]
    def fun(params):
        return sse(params[0], params[1], params[2], params[3], params[4], K, T, conf['last_close'], conf['r']
                   , conf['q'], conf['option_type'], conf['integration_rule'], midquote)

    result_opt = minimize(fun, x0, bounds=list(zip(lb, ub)), options=conf['options'], tol=1e-9)

    kappa, eta, theta, rho, v0 = result_opt.x

    calib_file = "./models_config_calibration/heston_calib.json"
    with open(calib_file, "w") as outfile:
        json.dump({'kappa': kappa, 'eta': eta, 'theta': theta, 'rho': rho, 'v0':v0}, outfile)
    #TODO save the parameters to a json file as ex, (then no need to always recalibrate)
    return {'kappa': kappa, 'eta': eta, 'theta': theta, 'rho': rho, 'v0':v0}
def jumpProcessCalibration(params):
    #Calib procedure
    return 0
def initialize_calib():
    # Import config_file stating: Data to use, input parameters
    file = "./models_config_calibration/heston_config.json"
    conf = json.load(open(file))
    # import calibration data

    #TODO More general load function, so we can import all 4 indices
    data = load_data(conf['save_dir'], conf['exp_date'], conf['option_type'])

    calibrated_params = hestonCalibration(data, conf)

    return 0

##TODO:

initialize_calib()
#Write parameters to file:

