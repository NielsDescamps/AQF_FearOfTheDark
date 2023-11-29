import numpy as np
import matplotlib.pyplot as plt

## 0. generate random samples 
# negative exponential
def pdfNegExp(gamma,x):
    return gamma*np.exp(gamma*x)

def inversecdfNegExp(gamma,z):
    inversecdf = np.log(z)/gamma
    return inversecdf

def generate_randomNegExp(gamma,rows,columns):
    
    uniform_samples = np.random.rand(rows,columns)

    return inversecdfNegExp(gamma,uniform_samples)

def plot_neg_exp_density(gamma, num_samples=1000, num_bins=50):
    # Generate random samples
    samples = generate_randomNegExp(gamma, num_samples, 1)

    # Plot the true PDF for comparison
    x = np.linspace(0, -1, 1000)
    plt.plot(x, pdfNegExp(gamma, x), 'r', label='True PDF')

    # Create a density plot (histogram)
    plt.hist(samples, bins=num_bins, density=True, alpha=0.6, color='g', label='Histogram')

    plt.title('Negative Exponential Density Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


