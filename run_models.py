import json
from src.Heston_FFT import Heston_FFT
from src.load_data import load_data
from src.get_maturities import get_maturities
import numpy as np
import matplotlib.pyplot as plt
from src.jump_characteristic_function import mutualjump_characteristic_function
from src.jump_characteristic_function import joint_characteristic_function
from src.heston_characteristic import heston_characteristic
from src.characteristic_to_pdf import characteristic_to_pdf
from src.jump_characteristic_function import lewis_pricing_formula
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
from scipy.stats import norm

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
    """
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
    """
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

    lambda_zero = np.array([0.1, 0.3])
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




