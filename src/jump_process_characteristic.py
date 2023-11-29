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

def exp_ODE(a, b, alpha, xi, delta, lamb, gamma):
    m = len(xi)
    db_dt = alpha*b[:m] + xi * b[m:] - gamma/(np.matmul(delta.T, b[:m]) + b[m:] + gamma)+1
    db_dt = np.concatenate((db_dt, np.zeros(m)), axis=0)
    da_dt = -np.dot(alpha * lamb, b[:m])
    return db_dt, da_dt

# Xi for exponential distribution
def exp_xi_i(gamma_i):
    return gamma_i/(gamma_i+1)-1
def jump_process_characteristic(alpha, lamb, delta, gamma):

    #Number of indices
    M = 2
    xi = np.ones(M)
    zero = np.zeros(M)
    Zero = np.zeros((M,M))
    #K_0 = np.concatenate((np.multiply(alpha, lamb), zero))
    #K_1 = np.concatenate((np.concatenate((np.diag(-alpha), Zero), axis=1), np.concatenate((np.diag(-xi), Zero), axis=1)))
    Lambda_0 = 0
    Lambda_1 = []
    zeta = []
    for i in range(M):
        # Lambda_1_i
        e = np.zeros(M)
        e[i] = 1
        Lambda_1.append(np.concatenate((e,zero)))

        # Zeta_i
        zeta.append(np.diag(np.concatenate((delta[:,i], e))))
    a = 1

    b = [4,3,2,1]
    da, db = exp_ODE(a, b, alpha, xi, delta, lamb, gamma)

    return 0


delta = np.array([[1,2],[3 , 4]])
jump_process_characteristic(np.ones(2), np.ones(2), delta, np.ones(2))



