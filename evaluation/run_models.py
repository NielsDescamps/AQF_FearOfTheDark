import json
from src.Heston_FFT import Heston_FFT
from src.load_data import load_data
from src.get_maturities import get_maturities
import numpy as np
import matplotlib.pyplot as plt

def import_model_data():
    #Import calibrated parameters
    heston_file = "./models_config_calibration/heston_calib.json"
    heston_params = json.load(open(heston_file))

    jump_process_file = "./models_config_calibration/jump_process_calib.json"
    jump_process_params = json.load(open(jump_process_file))

    # Import calibration data
    heston_conf_file = "./models_config_calibration/heston_config.json"
    heston_config = json.load(open(heston_conf_file))

    jump_process_conf_file = "./models_config_calibration/heston_config.json"
    jump_process_config = json.load(open(jump_process_conf_file))

    # TODO More general load function, so we can import all 4 indices
    data = load_data(heston_config['save_dir'], heston_config['exp_date'], heston_config['option_type'])

    return heston_params, heston_config, jump_process_params, jump_process_config, data

def plot_prices(option_data, heston_prices):
    strikes = option_data['strike']
    midquote = (option_data['ask'] + option_data['bid']) / 2

    plt.figure()
    plt.plot(strikes, heston_prices, 'g+', linewidth=1.1)
    plt.plot(strikes, midquote, 'o', linewidth=1.1)
    plt.xlabel('K')
    plt.ylabel('price')
    plt.title('Comparison FFT - Midquote')
    plt.legend(['FFT', 'Market'])
    plt.show()
def run_models():
    heston_params, heston_config, jump_process_params, jump_process_config, option_data = import_model_data()

    maturities = get_maturities([heston_config['exp_date']], heston_config['date_of_retrieval'])[0]
    strikes = option_data['strike']
    Heston_FFT_price = [Heston_FFT(heston_params['kappa'],
                                   heston_params['eta'],
                                   heston_params['theta'],
                                   heston_params['rho'],
                                   np.sqrt(heston_params['v0']),
                                   strike,
                                   maturities,
                                   heston_config['last_close'],
                                   heston_config['r'],
                                   heston_config['q'],
                                   heston_config['option_type'],
                                   heston_config['integration_rule']) for strike in strikes]
    plot_prices(option_data, Heston_FFT_price)

run_models()

