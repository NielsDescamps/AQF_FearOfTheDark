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

def run_models():
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
    kappa, eta, theta, rho, sigma0, r, q, S0 = heston_params['kappa'], heston_params['eta'], heston_params['theta'], heston_params['rho'], heston_params['v0'], heston_config['r'], heston_config['q'], heston_config['last_close']

    lambda_zero = jump_params[-1]
    t_values = np.linspace(0, 1, 100)  # Adjust the time range as needed
    T = 1
    t = 0
    h = 0.1
    u_values = np.linspace(-10, 10, 200)
    # # u_values = [1.5]
    index = 0  # Assuming you want to assess asset with index 0 (first asset)

    #params = [heston_conf : heston_params : jump_conf : jump_params]
    # # Example usage of mutualjump_characteristic_function

    def norm_char(u_values):
        return np.exp(1j*u_values -1/2*u_values**2*0.5232862836528391)

    def char_func(u):
        return np.exp(u * 1j - 1 / 2 * u ** 2)

    N = 4096
    alpha = 1.5
    eta_grid = 0.25
    lambda_val = 2 * np.pi / (N * eta_grid)
    b = lambda_val * N / 2

    # define grid of log-strikesAPE has size 4095
    # k = np.arange(-b, b - lambda_val, lambda_val)

    k = np.arange(-b, b, lambda_val)
    # compute rho
    v = np.arange(-N * eta_grid / 2, N * eta_grid / 2, eta_grid)
    u = np.array(v)

    #lewis_pricing_formula(t, T, kappa, eta, theta, rho, sigma0, r, q, S0, jump_params, lambda_zero, h, index, jump_distribution)


    PHI_joint = [joint_characteristic_function(val, t, T, heston_params['kappa'], heston_params['eta'], heston_params['theta'],
                                      heston_params['rho'], heston_params['v0'], heston_config['r'], heston_config['q'],
                                      heston_config['last_close'], jump_params, lambda_zero, h, index,
                                      jump_distribution) for val in u]

    PHI_heston = [heston_characteristic(kappa, eta, theta, rho, sigma0, r, q, S0, T, val) for val in u]
    PHI_jump = [mutualjump_characteristic_function(jump_params,lambda_zero, t, T, h, val, index, jump_distribution) for val in u]
    # Inverse Fourier transform to obtain the probability density function
    pdf = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(norm_char(u))))
    pdf_heston = np.fft.ifftshift(np.fft.ifft(PHI_heston))
    pdf_joint = np.fft.ifftshift(np.fft.ifft(PHI_joint))
    #pdf_joint = np.fft.ifftshift(np.fft.ifft(PHI_joint))

    k_heston, norm_heston = characteristic_to_pdf(u,np.abs(pdf_heston))
    k_joint, norm_joint = characteristic_to_pdf(u, np.abs(pdf_joint))

    plt.figure()
    #plt.plot(u_values, np.real(characteristic_function), label='standard normal pdf')
    #plt.plot(u_values, np.imag(characteristic_function), label = 'standard imaginary')
    plt.plot(k_heston, norm_heston, label = 'pdf joint')
    plt.plot(k_joint, norm_joint, label='pdf heston')
    #plt.plot(u, np.real(PHI_jump, label = 'jump abs')
    #plt.plot(k, np.real(pdf_joint), label='real norm')
    # plt.plot(u_values, np.imag(pdf), label = 'pdf imag')
    plt.xlabel('u')
    plt.ylabel('pdf')
    plt.title('Probability density function')
    plt.legend()
    plt.show()

def plot_pdf():
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

        pdf_heston = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_heston)))
        pdf_joint = np.abs(np.fft.ifftshift(np.fft.ifft(PHI_joint)))

        # Normalisation
        pdf_heston = pdf_heston / lambda_val
        pdf_joint = pdf_joint / lambda_val
        #Translation
        #TODO this is sloppy dont see where the need for translation comes from
        E_h = np.sum(k * pdf_heston * lambda_val)
        E_j = np.sum(k * pdf_joint * lambda_val)
        k_heston = k - E_h
        k_joint = k - E_h

        plt.figure()
        plt.plot(k_heston, pdf_heston, label='pdf Hawkes Model')
        plt.plot(k_joint, pdf_joint, label='pdf Heston Model')
        plt.xlabel('z [log-returns]')
        plt.ylabel('probability density')
        plt.title('Probability density functions')
        plt.legend()
        plt.show()

def plot_option_smiles():

    #Import the necessary data from data_cleanup.py
    data = pd.read_csv('data_XNG/data_options.csv')

    heston_calib_XNG_gaussian_file = "./models_config_calibration/heston_calib_XNG_jump_gaussian.json"
    heston_params_XNG_G = json.load(open(heston_calib_XNG_gaussian_file))

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
    lambda_zero = jump_params[3]
    kappa, eta, theta, rho, v0, r, q, S0 = heston_params['kappa'], heston_params['eta'], heston_params['theta'], \
    heston_params['rho'], heston_params['v0'], heston_config['r'], heston_config['q'], heston_config['last_close']

    kappa_XNG_G, eta_XNG_G, theta_XNG_G, rho_XNG_G, sigma0_XNG_G = heston_params_XNG_G['kappa'], heston_params_XNG_G['eta'], heston_params_XNG_G['theta'], \
        heston_params_XNG_G['rho'], heston_params_XNG_G['v0']

    BS_pricing = []
    heston_pricing = []
    jump_pricing = []
    heston_impliedVol = []
    jump_impliedVol = []
    market_impliedVol = []
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

        jump_price = Carr_Madan_joint_option_pricer(t, T, K, type, kappa_XNG_G, eta_XNG_G, theta_XNG_G, rho_XNG_G, np.sqrt(sigma0_XNG_G), r, q, S0, jump_params, lambda_zero, h, indice_index, jump_distribution)
        jump_pricing.append(jump_price)


        ##TODO the imped vol doesnt work for some reason
        heston_impliedVol.append(calc_implied_vol(heston_call_price, S0, K, r, q, T, type_nbr))
        jump_impliedVol.append(calc_implied_vol(jump_price, S0, K, r, q, T, type_nbr))
        market_impliedVol.append(calc_implied_vol(MQ, S0, K, r, q, T, type_nbr))


    df_calc["Heston Call Prices"] = heston_pricing
    df_calc["Black Scholes Call Prices"] = BS_pricing
    df_calc["Jump model Call Price"] = jump_pricing
    df_calc["Heston ImpliedVol"] = heston_impliedVol
    df_calc["Jump ImpliedVol"] = jump_impliedVol
    df_calc["Market ImpliedVol"] = market_impliedVol

    df_calc.dropna()

    #Plottting strike - call price
    plt.figure()
    plt.plot(df_calc["Strike"], df_calc["MQ"], 'bo', label="Market Midquotes", linewidth=1.1)
    plt.plot(df_calc["Strike"], df_calc["Heston Call Prices"], 'r+', label="Heston call price", linewidth=1.1)
    plt.plot(df_calc["Strike"], df_calc["Black Scholes Call Prices"], '+', label="Black Scholes call price", linewidth=1.1)
    plt.plot(df_calc["Strike"], df_calc["Jump model Call Price"], 'v', label="Jump model Call price", linewidth=1.1)
    plt.xlabel('Strikes')
    plt.ylabel('Call Price')
    plt.title('Pricing model')
    plt.legend()
    plt.show()

    #Plotting implied vol
    plt.figure()
    plt.plot(df_calc["moneyness"], df_calc["ImpliedVol"], 'bo', label="Market implied vol", linewidth=1.1)
    plt.plot(df_calc["moneyness"], df_calc["Heston ImpliedVol"], 'r+', label="Heston implied vol", linewidth=1.1)
    plt.plot(df_calc["moneyness"], df_calc["Jump ImpliedVol"], 'v', label="Jump model implied vol", linewidth=1.1)
    plt.xlabel('Moneyness K/S0')
    plt.ylabel('Implied volatility')
    plt.title('Pricing models')
    plt.legend()
    plt.show()




plot_option_smiles()



