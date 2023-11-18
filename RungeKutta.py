import numpy as np
import matplotlib.pyplot as plt

def ode_system(t, y):
    """
    Define your system of ODEs here.
    This function should return a numpy array representing the derivatives of y with respect to t.
    """
    # Example: dy/dt = [y[1], -y[0]]
    dydt = np.array([y[1], -y[0]])
    return dydt

#ODe system for Gaussian jumps
def ode_system_Gausian(t,a,b,params):
    """
    Ths function defines the system of ODEs that need solving for Gaussian jump size
    The vector y consists of (for m=2):
    - b1, b2, b3, b4
    - a
    This function should return a numpy array representing the derivatives of y with respect to t.
    """
    

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


def runge_kutta_4th_order_finalbc(ode_func, final_boundary_condition, t_span, h):
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

        k1 = h * ode_func(t, y)
        k2 = h * ode_func(t - 0.5 * h, y - 0.5 * k1)
        k3 = h * ode_func(t - 0.5 * h, y - 0.5 * k2)
        k4 = h * ode_func(t - h, y - k3)

        y_values[i - 1] = y - (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values

# Example usage:
final_boundary_conditions = np.array([1.0, 0.0])  # Example initial conditions
t_span = (0, 10)  # Example time span
h = 0.01  # Example step size

t_values, y_values = runge_kutta_4th_order_finalbc(ode_system, final_boundary_conditions, t_span, h)

# Plot the results
plt.plot(t_values, y_values[:, 0], label='y[0]')
plt.plot(t_values, y_values[:, 1], label='y[1]')
plt.xlabel('Time')
plt.ylabel('Variable Values')
plt.legend()
plt.show()
