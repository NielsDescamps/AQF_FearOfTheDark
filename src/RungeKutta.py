import numpy as np
import matplotlib.pyplot as plt

def ode_system(t, y,params):
    """
    Define your system of ODEs here.
    This function should return a numpy array representing the derivatives of y with respect to t.
    """
    # Example: dy/dt = [y[1], -y[0]]
    dydt = np.array([y[1], -y[0]])
    return dydt

#ODE system for Gaussian jumps
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
    [alpha,delta,beta,sigma,lambda_bar] = params

    m = len(alpha)
    b=y[:2*m]
    a=y[2*m]

    dydt = np.zeros(2*m+1)
    
    ## 1. db/dt for 1..m
    for j in range(m):
        # calculate exponent
        exponent_j = b[m+j]*beta[j] + 1/2*(b[j+m])**2 * sigma[j]**2 + 1
        for i in range(m):
            exponent_j += delta[i,j]*b[i]
         
        #calculate xi
        xi_j = np.exp(beta[j]+1/2*sigma[j]**2)

        #calculate dydt
        dydt[j] = alpha[j]*b[j] + xi_j*b[j+m] - np.exp(exponent_j)


    ## 2. db/dt for m+1..2*m
    for j in range(m, 2*m):
        dydt[j] = 0 

    ## 3. da/dt
    dydt[2*m] = -np.sum(alpha * lambda_bar * b[:m])

    return dydt


def runge_kutta_4th_order_initbc(ode_func, initial_conditions, t_span, h):
    """
    Implement the RK4 method to solve a system of ODEs given a set of initial conditions.

    Parameters:
    - ode_func: The function defining the system of ODEs.
    - initial_conditions: Numpy array with initial values for the variables.
    - t_span: A tuple (t_start, t_end) representing the time span for integration.
    - h: Step size for the integration.

    Returns:
    - t_values: Array of time values.
    - y_values: Array of variable values corresponding to each time point.
    """
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + h, h)
    num_steps = len(t_values)

    y_values = np.zeros((num_steps, len(initial_conditions)))
    y_values[0] = initial_conditions

    for i in range(1, num_steps):
        t = t_values[i - 1]
        y = y_values[i - 1]

        k1 = h * ode_func(t, y)
        k2 = h * ode_func(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * ode_func(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * ode_func(t + h, y + k3)

        y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values


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
    """
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + h, h)
    num_steps = len(t_values)

    y_values = np.zeros((num_steps, len(final_boundary_condition)))

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

# #Example usage of ode_systemGaussian 
# alpha = np.array([0.1, 0.2]) 
# beta = np.array([0.5 , 0.6]) 
# sigma = np.array([0 , 0]) 
# delta = np.array([[0, -0.1], [-0.1, 0]]) 
# lambda_bar = np.array([0.001, 0.001]) 
# params = [alpha,delta,beta,sigma,lambda_bar] 
# m = len(alpha)

# v=np.array([1,1,1,1])
# final_boundary_conditions = np.array([1,-1,2,2,0])

# t_span = (0,10)
# h=1

# t_values, y_values = runge_kutta_4th_order_finalbc(ode_system_Gaussian, final_boundary_conditions, t_span, h,params)

# for i in range(len(y_values[1])):
#     plt.plot(t_values, y_values[:, i], label=f'y[{i}]')
# plt.xlabel('Time')
# plt.ylabel('Variable Values')
# plt.legend()
# plt.show()


# # Example usage of ode_system
# final_boundary_conditions = np.array([1.0, 0.0])  # Example initial conditions
# t_span = (0, 10)  # Example time span
# h = 0.01  # Example step size

# t_values, y_values = runge_kutta_4th_order_finalbc(ode_system, final_boundary_conditions, t_span, h,False)

# print(t_values)
# print(y_values[:,1])

# # Plot the results
# for i in range(len(y_values[1])):
#     plt.plot(t_values, y_values[:, i], label=f'y[i]')
# plt.xlabel('Time')
# plt.ylabel('Variable Values')
# plt.legend()
# plt.show()
