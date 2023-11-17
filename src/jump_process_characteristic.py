import numpy as np
from scipy import integrate

#Exponential jump process
def lambda_t(t):
    return 0
def theta_i(F_i):
    return 0
# Pdf for exponential
def dF_i(z,gamma_i):
    if z > 0:
        return
    else:
        return gamma_i*np.exp(gamma_i*z)

def xi_i(gamma_i):
    return integrate.quad(lambda y: (np.exp(y)-1)*dF_i(y, gamma_i), -np.inf, 0)
def jump_process_characteristic(a, lamb, delta, gamma):

    #Number of indices
    N = 4
    xi = np.ones(N)
    zero = np.zeros(N)
    Zero = np.zeros((N,N))
    K_0 = np.concatenate((np.multiply(a,lamb),zero))
    K_1 = np.concatenate((np.concatenate((np.diag(-a), Zero), axis=1), np.concatenate((np.diag(-xi),Zero), axis=1)))
    Lambda_0 = 0
    Lambda_1 = []
    zeta = []
    for i in range(N):
        # Lambda_1_i
        e = np.zeros(N)
        e[i] = 1
        Lambda_1.append(np.concatenate((e,zero)))

        # Zeta_i
        zeta.append(np.diag(np.concatenate((delta[:,i], e))))

    return 0


delta = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
jump_process_characteristic(np.ones(4), np.ones(4), delta, np.ones(4))



