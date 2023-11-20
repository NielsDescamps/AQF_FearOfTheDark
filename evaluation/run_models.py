import json
from src.Heston_FFT import Heston_FFT
from src.load_data import load_data
from src.get_maturities import get_maturities
import numpy as np
import matplotlib.pyplot as plt
from jump_characteristic_function import mutualjump_characteristic_function

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
    #plot_prices(option_data, Heston_FFT_price)

    # # Example parameters
    jump_distribution = "Exponential"

    if jump_distribution == "Exponential":
        params = [np.array(jump_process_params['alpha']),
                   np.array(jump_process_params['delta']),
                   np.array(jump_process_params['gamma']),
                   np.array(jump_process_params['lambda_bar'])
                   ]
    elif jump_distribution == "Gaussian":
        params = [np.array(jump_process_params['alpha']),
                  np.array(jump_process_params['delta']),
                  np.array(jump_process_params['beta']),
                  np.array(jump_process_params['sigma']),
                  np.array(jump_process_params['lambda_bar'])
                  ]
    lambda_zero = np.array([0.1, 0.3])
    t_values = np.linspace(0, 1, 100)  # Adjust the time range as needed
    T = 1
    t = 0
    h = 0.1
    u_values = np.linspace(-10, 10, 200)
    # # u_values = [1.5]
    index = 0  # Assuming you want to assess asset with index 0 (first asset)

    # # Example usage of mutualjump_characteristic_function
    PHI_values = [mutualjump_characteristic_function(params, lambda_zero, t, T, h, u, index, jump_distribution) for u in u_values]

    # # Plotting
    plt.plot(u_values, np.real(PHI_values), label='Real part of PHI')
    plt.plot(u_values, np.imag(PHI_values), label='Imaginary part of PHI')
    plt.xlabel('Time')
    plt.ylabel('PHI values')
    plt.legend()
    plt.title('PHI values for different u')
    plt.show()

run_models()

