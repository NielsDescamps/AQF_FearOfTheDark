import numpy as np
import matplotlib.pyplot as plt


def ode_system_Exponential(t,y,params):
    """
        Ths function defines the system of ODEs that need solving for Exponential jump size
        Input params:
        - alpha = vector of size m with alpha values
        - gamma = vector of size m with gamma values
        - delta = matrix of size mXm

        The input vector y consists of  :
        - b1, b2, b3, b4 (2*m)
        - a (1)
        ==> in total y will have dimension of (2*m + 1)

        This function should return a numpy array representing the derivatives of y with respect to t.
        """
    alpha, delta, gamma, lamb = params
    m = len(alpha)
    b = y[:2 * m]
    xi = gamma/(gamma+1)-1
    db_dt = alpha * b[:m] + xi * b[m:] - gamma / (np.matmul(delta.T, b[:m]) + b[m:] + gamma) + 1
    db_dt = np.concatenate((db_dt, np.zeros(m)), axis=0)
    da_dt = -np.dot(alpha * lamb, b[:m])
    dy_dt = np.concatenate((db_dt,[da_dt]), axis=0)
    return dy_dt

def ode_system_Gaussian(t,y,params):
    """
        Ths function defines the system of ODEs that need solving for Gaussian jump size
        Input params:
        - alpha = vector of size m with alpha values
        - beta = vector of size m with beta values
        - sigma = vector of size m with beta values
        - delta = matrix of size mXm

        The input vector y consists of  :
        - b1, b2, b3, b4 (2*m)
        - a (1)
        ==> in total y will have dimension of (2*m + 1)

        This function should return a numpy array representing the derivatives of y with respect to t.
        """
    alpha, delta, beta, sigma, lambda_bar = params
    m = len(alpha)
    b = y[:2 * m]
    xi = np.exp(beta+1/2*(np.square(sigma)))-1
    db_dt = alpha * b[:m] + xi * b[m:] - np.exp(np.matmul(delta.T, b[:m]) + b[m:] * beta + 1/2*b[m:]**2 * sigma**2)+1
    db_dt = np.concatenate((db_dt, np.zeros(m)), axis=0)
    da_dt = -np.dot(alpha * lambda_bar, b[:m])
    dy_dt = np.concatenate((db_dt,[da_dt]), axis=0)
    return dy_dt

def runge_kutta_4th_order_finalbc(ode_func, final_boundary_condition, t_span, h, params):
    """
    Implement the RK4 method to solve a system of ODEs given a set of final conditions

    Parameters:
    - ode_func: The function defining the system of ODEs.
    - initial_conditions: Numpy array with initial values for the variables.
    - t_span: A tuple (t_start, t_end) representing the time span for integration.
    - h: Step size for the integration.

    Returns:
    - t_values: Array of time values.
    - y_values: Array of variable values corresponding to each time point.
        - row 1..(2m-1) contains b(t)
        - row 2m contains a(t) 
    """
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + h, h)
    num_steps = len(t_values)

    y_values = np.zeros((num_steps, len(final_boundary_condition)), dtype=np.complex128)

    # Set the final boundary condition
    y_values[-1] = final_boundary_condition

    for i in range(num_steps - 1, 0, -1): # reverse the order in which the algorithms runs over time
        t = t_values[i]
        y = y_values[i]

        k1 = h * ode_func(t, y, params)
        k2 = h * ode_func(t - 0.5 * h, y - 0.5 * k1, params)
        k3 = h * ode_func(t - 0.5 * h, y - 0.5 * k2, params)
        k4 = h * ode_func(t - h, y - k3, params)
        y_values[i - 1] = y - (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values
