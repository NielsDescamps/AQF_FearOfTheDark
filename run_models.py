import json
from src.Heston_FFT import Heston_FFT
from src.load_data import load_data
from src.get_maturities import get_maturities
import numpy as np
import matplotlib.pyplot as plt
from src.jump_characteristic_function import mutualjump_characteristic_function
from src.jump_characteristic_function import joint_characteristic_function
from src.BlackScholes import calc_implied_vol
from src.heston_characteristic import heston_characteristic
from src.BlackScholes import BS_price
from src.Carr_Madan_joint_option_pricer import Carr_Madan_joint_option_pricer
import pandas as pd
from data_cleanup import calc_params, filter_data

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


def plot_gaussian_pdf():
    # Set the jump distribution
    jump_distribution = "Gaussian"
    # Select index to model
    index = 1
    #Eval time
    t = 0
    # Maturity
    T = 1
    # time step for the evaluation of the set of ODEs
    h = 0.1
    # Import data
    heston_params, heston_config, jump_process_params, jump_process_config, option_data = import_model_data()
    kappa, eta, theta, rho, sigma0, r, q, S0 = heston_params['kappa'], heston_params['eta'], heston_params['theta'], \
    heston_params['rho'], heston_params['v0'], heston_config['r'], heston_config['q'], heston_config['last_close']

    if jump_distribution == "Exponential":
        jump_params = [np.array(jump_process_params['alpha']),
                       -np.array(jump_process_params['delta']),
                       np.array(jump_process_params['gamma']),
                       np.array(jump_process_params['lambda_bar'])
                       ]
    elif jump_distribution == "Gaussian":
        jump_params = [np.array(jump_process_params['alpha']),
                       np.array(jump_process_params['delta']),
                       np.array(jump_process_params['beta']),
                       np.array(jump_process_params['sigma']),
                       np.array(jump_process_params['lambda_bar'])
                       ]

        N = 4096
        alpha = 1.5
        eta_grid = 0.25
        lambda_val = 2 * np.pi / (N * eta_grid)
        b = lambda_val * N / 2
        k = np.arange(-b, b, lambda_val)
        v = np.arange(-N * eta_grid / 2, N * eta_grid / 2, eta_grid)
        u = np.array(v)
        lambda_zero = jump_params[3]

        PHI_joint = [joint_characteristic_function(val, t, T, heston_params['kappa'], heston_params['eta'],
                                                   heston_params['theta'],
                                                   heston_params['rho'], heston_params['v0'], heston_config['r'],
                                                   heston_config['q'],
                                                   heston_config['last_close'], jump_params, lambda_zero, h, index,
                                                   jump_distribution) for val in u]
        PHI_heston = [heston_characteristic(kappa, eta, theta, rho, sigma0, r, q, S0, T, val) for val in u]

        jump_params[1] = np.array([[1, 1.1], [1.1, 1]])

        PHI_joint_low = [joint_characteristic_function(val, t, T, heston_params['kappa'], heston_params['eta'],
                                                   heston_params['theta'],
                                                   heston_params['rho'], heston_params['v0'], heston_config['r'],
                                                   heston_config['q'],
                                                   heston_config['last_close'], jump_params, lambda_zero, h, index,
                                                   jump_distribution) for val in u]

        jump_params[1] = np.array([[20, 10], [10, 20]])
        PHI_joint_high = [joint_characteristic_function(val, t, T, heston_params['kappa'], heston_params['eta'],
                                                   heston_params['theta'],
                                                   heston_params['rho'], heston_params['v0'], heston_config['r'],
                                                   heston_config['q'],
                                                   heston_config['last_close'], jump_params, lambda_zero, h, index,
                                                   jump_distribution) for val in u]

        pdf_heston = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_heston)))
        pdf_joint = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_joint)))
        pdf_joint_low = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_joint_low)))
        pdf_joint_high = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_joint_high)))

        # Normalisation
        pdf_heston = pdf_heston / lambda_val
        pdf_joint = pdf_joint / lambda_val
        pdf_joint_high = pdf_joint_high / lambda_val
        pdf_joint_low = pdf_joint_low / lambda_val
        #Translation
        #TODO this is sloppy dont see where the need for translation comes from
        E_h = np.sum(k * pdf_heston * lambda_val)
        E_j = np.sum(k * pdf_joint * lambda_val)
        k_heston = k - E_h
        k_joint = k - E_h

        plt.figure()
        plt.plot(k_heston, pdf_heston, label='Heston Model')
        plt.plot(k_joint, pdf_joint, label='Cross excitation medium')
        plt.plot(k_joint, pdf_joint_low, label='Cross excitation low')
        plt.plot(k_joint, pdf_joint_high, label='Cross excitation high')
        plt.xlabel('z [log-returns]')
        plt.ylabel('probability density')
        plt.title('Different Levels of Cross Excitiations Gaussian Jump')
        plt.legend()
        plt.show()

def plot_exponential_pdf():
    # Set the jump distribution
    jump_distribution = "Exponential"
    # Select index to model
    index = 1
    #Eval time
    t = 0
    # Maturity
    T = 1
    # time step for the evaluation of the set of ODEs
    h = 0.1
    # Import data
    heston_calib_XNG_exponential_file = "./models_config_calibration/heston_calib_XNG_jump_exponential.json"
    heston_params_XNG_E = json.load(open(heston_calib_XNG_exponential_file))
    jump_params_XNG_exponential_file = "./models_config_calibration/jump_calib_XNG_jump_exponential.json"
    jump_params_XNG_E = json.load(open(jump_params_XNG_exponential_file))

    heston_params, heston_config, jump_process_params, jump_process_config, option_data = import_model_data()
    kappa, eta, theta, rho, sigma0, r, q, S0 = heston_params['kappa'], heston_params['eta'], heston_params['theta'], \
    heston_params['rho'], heston_params['v0'], heston_config['r'], heston_config['q'], heston_config['last_close']

    jump_params = [np.array(jump_process_params['alpha']),
                       np.array(jump_process_params['delta']),
                       np.array(jump_process_params['gamma']),
                       np.array(jump_process_params['lambda_bar'])]

    N = 4096
    alpha = 1.5
    eta_grid = 0.25
    lambda_val = 2 * np.pi / (N * eta_grid)
    b = lambda_val * N / 2
    k = np.arange(-b, b, lambda_val)
    v = np.arange(-N * eta_grid / 2, N * eta_grid / 2, eta_grid)
    u = np.array(v)
    lambda_zero = jump_params[3]

    PHI_joint = [joint_characteristic_function(val, t, T, heston_params['kappa'], heston_params['eta'],
                                                   heston_params['theta'],
                                                   heston_params['rho'], heston_params['v0'], heston_config['r'],
                                                   heston_config['q'],
                                                   heston_config['last_close'], jump_params, lambda_zero, h, index,
                                                   jump_distribution) for val in u]
    PHI_heston = [heston_characteristic(kappa, eta, theta, rho, sigma0, r, q, S0, T, val) for val in u]

    jump_params[1] = np.array([[-10, -1.1], [-1.1, -10]])

    PHI_joint_low = [joint_characteristic_function(val, t, T, heston_params['kappa'], heston_params['eta'],
                                                   heston_params['theta'],
                                                   heston_params['rho'], heston_params['v0'], heston_config['r'],
                                                   heston_config['q'],
                                                   heston_config['last_close'], jump_params, lambda_zero, h, index,
                                                   jump_distribution) for val in u]

    jump_params[1] = np.array([[-250, -30], [-30, -250]])
    PHI_joint_high = [joint_characteristic_function(val, t, T, heston_params['kappa'], heston_params['eta'],
                                                   heston_params['theta'],
                                                   heston_params['rho'], heston_params['v0'], heston_config['r'],
                                                   heston_config['q'],
                                                   heston_config['last_close'], jump_params, lambda_zero, h, index,
                                                   jump_distribution) for val in u]

    pdf_heston = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_heston)))
    pdf_joint = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_joint)))
    pdf_joint_low = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_joint_low)))
    pdf_joint_high = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_joint_high)))

    # Normalisation
    pdf_heston = pdf_heston / lambda_val
    pdf_joint = pdf_joint / lambda_val
    pdf_joint_high = pdf_joint_high / lambda_val
    pdf_joint_low = pdf_joint_low / lambda_val
    #Translation
    #TODO this is sloppy dont see where the need for translation comes from
    E_h = np.sum(k * pdf_heston * lambda_val)
    E_j = np.sum(k * pdf_joint * lambda_val)
    k_heston = k - E_h
    k_joint = k - E_h

    plt.figure()
    plt.plot(k_heston, pdf_heston, label='Heston Model')
    plt.plot(k_joint, pdf_joint, label='Cross excitation medium')
    plt.plot(k_joint, pdf_joint_low, label='Cross excitation low')
    plt.plot(k_joint, pdf_joint_high, label='Cross excitation high')
    plt.xlabel('z [log-returns]')
    plt.ylabel('probability density')
    plt.title('Different Levels of Cross Excitiations Exponential Jump')
    plt.legend()
    plt.show()

def plot_XNG():

    #Import the necessary data from data_cleanup.py
    data = pd.read_csv('data_XNG/data_options.csv')

    heston_calib_XNG_gaussian_file = "./models_config_calibration/heston_calib_XNG_jump_gaussian.json"
    heston_params_XNG_G = json.load(open(heston_calib_XNG_gaussian_file))

    heston_calib_XNG_exponential_file = "./models_config_calibration/heston_calib_XNG_jump_exponential.json"
    heston_params_XNG_E = json.load(open(heston_calib_XNG_exponential_file))
    jump_params_XNG_exponential_file = "./models_config_calibration/jump_calib_XNG_jump_exponential.json"
    jump_params_XNG_E = json.load(open(jump_params_XNG_exponential_file))

    exp_date = '16.01.2010'  # or  '17.04.2010'
    price_date = '04.01.2010'
    # On pricing day:
    Open = 539.61
    High = 558.48
    Low = 539.61
    Close = 558.44
    S0 = (Open+Close)/2

    ticker = 'XNG'  # or BTK, XBD and MSH (use XNG, others have less quotes per day)
    df = filter_data(data, exp_date, price_date, ticker)
    t = 0
    r = 0.05
    q = 0.02
    df_calc = calc_params(df, S0, r, q)
    print(df_calc)
    print('output shape: ', df_calc.shape)

    df_calc["moneyness"] = df_calc["Strike"]/df_calc["MQ"]
    df_calc.drop(df_calc[df_calc['Type'] != "C"].index, inplace=True)
    #df_calc.drop(df_calc[np.abs(df_calc['moneyness']-1) >= 2].index, inplace=True)



    # Set the jump distribution
    jump_distribution = "Gaussian"
    # Select index to model
    indice_index = 1
    # Eval time
    t = 0
    # Maturity
    T = 1
    # time step for the evaluation of the set of ODEs
    h = 0.1
    heston_params, heston_config, jump_process_params, jump_process_config, option_data = import_model_data()

    maturities = get_maturities([heston_config['exp_date']], heston_config['date_of_retrieval'])[0]
    strikes = option_data['strike']

    # # Example parameters
    jump_distribution = "Gaussian"

    if jump_distribution == "Exponential":
        jump_params = [np.array(jump_process_params['alpha']),
                       np.array(jump_process_params['delta']),
                       np.array(jump_process_params['gamma']),
                       np.array(jump_process_params['lambda_bar'])
                       ]
    elif jump_distribution == "Gaussian":
        jump_params = [np.array(jump_process_params['alpha']),
                       np.array(jump_process_params['delta']),
                       np.array(jump_process_params['beta']),
                       np.array(jump_process_params['sigma']),
                       np.array(jump_process_params['lambda_bar'])
                       ]
    lambda_zero_G = jump_params[3]
    kappa, eta, theta, rho, v0, r, q, S0 = heston_params['kappa'], heston_params['eta'], heston_params['theta'], \
    heston_params['rho'], heston_params['v0'], heston_config['r'], heston_config['q'], heston_config['last_close']

    kappa_XNG_G, eta_XNG_G, theta_XNG_G, rho_XNG_G, sigma0_XNG_G = heston_params_XNG_G['kappa'], heston_params_XNG_G['eta'], heston_params_XNG_G['theta'], \
        heston_params_XNG_G['rho'], heston_params_XNG_G['v0']

    kappa_XNG_E, eta_XNG_E, theta_XNG_E, rho_XNG_E, sigma0_XNG_E = heston_params_XNG_E['kappa'], heston_params_XNG_E[
        'eta'], heston_params_XNG_E['theta'], \
        heston_params_XNG_E['rho'], heston_params_XNG_E['v0']
    jump_params_E = [np.array(jump_params_XNG_E['alpha']),
                   np.array(jump_params_XNG_E['delta']),
                   np.array(jump_params_XNG_E['gamma']),
                   np.array(jump_params_XNG_E['lambda_bar'])]

    lambda_zero_E = np.array(jump_params_XNG_E['lambda_bar'])

    BS_pricing = []
    heston_pricing = []
    jump_pricing_G = []
    jump_pricing_E = []
    heston_impliedVol = []
    jump_impliedVol_G = []
    jump_impliedVol_E = []
    market_impliedVol = []
    BS_impliedVol = []
    for index, row in df_calc.iterrows():
        type = row["Type"]
        if type == 'C':
            type_nbr = 0
        elif type == 'P':
            type_nbr = 1
        r = row["R"]
        q = row["Q"]
        S0 = row["S0"]
        T = row["TTM"]
        MQ = row["MQ"]
        #T = 1
        K = row["Strike"]
        integration_rule = 1
        heston_call_price = Heston_FFT(kappa, eta, theta, rho, np.sqrt(v0), K, T, S0, r, q, type_nbr, integration_rule)
        heston_pricing.append(heston_call_price)

        BS_call_price = BS_price(np.sqrt(eta), S0, K, r, q, T, type_nbr)
        BS_pricing.append(BS_call_price)

        jump_price_G = Carr_Madan_joint_option_pricer(t, T, K, type, kappa_XNG_G, eta_XNG_G, theta_XNG_G, rho_XNG_G, np.sqrt(sigma0_XNG_G), r, q, S0, jump_params, lambda_zero_G, h, indice_index, jump_distribution)
        jump_pricing_G.append(jump_price_G)

        jump_price_E = Carr_Madan_joint_option_pricer(t, T, K, type, kappa_XNG_E, eta_XNG_E, theta_XNG_E, rho_XNG_E,
                                                      np.sqrt(sigma0_XNG_E), r, q, S0, jump_params_E, lambda_zero_E, h,
                                                      indice_index, "Exponential")
        jump_pricing_E.append(jump_price_E)


        # implied vol doesnt work on all values so have to sort None vals
        heston_impliedVol.append(calc_implied_vol(heston_call_price, S0, K, r, q, T, type_nbr))
        jump_impliedVol_G.append(calc_implied_vol(jump_price_G, S0, K, r, q, T, type_nbr))
        market_impliedVol.append(calc_implied_vol(MQ, S0, K, r, q, T, type_nbr))
        jump_impliedVol_E.append(calc_implied_vol(jump_price_E, S0, K, r, q, T, type_nbr))
        #BS_impliedVol.append(calc_implied_vol(BS_call_price, S0, K, r, q, T, type_nbr))


    df_calc["Heston Call Prices"] = heston_pricing
    df_calc["Black Scholes Call Prices"] = BS_pricing
    df_calc["Jump model Call Price G"] = jump_pricing_G
    df_calc["Jump model Call Price E"] = jump_pricing_E
    df_calc["Heston ImpliedVol"] = heston_impliedVol
    df_calc["Jump ImpliedVol G"] = jump_impliedVol_G
    df_calc["Jump ImpliedVol E"] = jump_impliedVol_E
    df_calc["Market ImpliedVol"] = market_impliedVol
    #df_calc["BS ImpliedVol"] = BS_impliedVol

    df_calc.dropna()

    #Plottting strike - call price
    plt.figure()
    plt.plot(df_calc["Strike"], df_calc["MQ"], 'bo', label="Market Midquotes", linewidth=1.1)
    plt.plot(df_calc["Strike"], df_calc["Heston Call Prices"], 'r+', label="Heston call price", linewidth=1.1)
    plt.plot(df_calc["Strike"], df_calc["Black Scholes Call Prices"], '+', label="Black Scholes call price", linewidth=1.1)
    plt.plot(df_calc["Strike"], df_calc["Jump model Call Price G"], 'v', label="Jump model Call price", linewidth=1.1)
    plt.plot(df_calc["Strike"], df_calc["Jump model Call Price E"], 'g2', label="Jump model Call price", linewidth=1.1)
    plt.xlabel('Strikes')
    plt.ylabel('Call Price')
    plt.title('Pricing model')
    plt.legend()
    plt.show()

    #Plotting implied vol
    plt.figure()
    plt.plot(df_calc["moneyness"], df_calc["ImpliedVol"], 'bo', label="Market implied vol", linewidth=1.1)
    plt.plot(df_calc["moneyness"], df_calc["Heston ImpliedVol"], 'r+', label="Heston implied vol", linewidth=1.1)
    plt.plot(df_calc["moneyness"], df_calc["Jump ImpliedVol G"], 'v', label="Jump model implied vol", linewidth=1.1)
    plt.plot(df_calc["moneyness"], df_calc["Jump ImpliedVol E"], 'g2', label="Jump model implied vol", linewidth=1.1)
    plt.xlabel('Moneyness K/S0')
    plt.ylabel('Implied volatility')
    plt.title('Pricing models')
    plt.legend()
    plt.show()

plot_exponential_pdf()
plot_gaussian_pdf()






