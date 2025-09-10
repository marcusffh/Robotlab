import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

#Gaussian pdf
def Gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

#Target distribution p(x)
def p(x):
    return (0.3 * Gaussian_pdf(x, 2, 1) + 0.4 * Gaussian_pdf(x, 5, 2) + 0.3 * Gaussian_pdf(x, 9, 1))

#Sampling-Importance-Resampling (SIR) algorithm
def SIR(k, sample_q, q):
    #Sampling from proposal distribution q(x)
    x =  sample_q(k)
    
    #Compute importance weights               
    w_unnormalized = p(x) / q(x)
    w = w_unnormalized / np.sum(w_unnormalized)
    
    x = x[w > 0]
    w = w[w > 0]
    #Resampling
    resampled = []
    #Compute the cumulative distribution
    H = np.cumsum(w)
    
    #Sample a random number 𝑧 ∈ [0,1] from a uniform distribution
    for _ in range(k):
        z = np.random.rand()
        i = np.searchsorted(H, z)
        resampled.append(x[i])
    return resampled

def sample_uniform(k):
    return np.random.uniform(0, 15, k)

#always 1/15 but need to be able to call as a function
def uniform_pdf(x):
    return uniform.pdf(x, 0, 15)

def sample_normal(k):
    return np.random.normal(5, 2, size=k)

def normal_pdf(x):
    return Gaussian_pdf(x, 5, 2) 

k_values = [20, 100, 1000]

for k in k_values:
    resampled = SIR(k, sample_uniform, uniform_pdf)
    plt.figure()
    plt.hist(resampled, bins=50, density=True, alpha=0.6, label='Resampled particles')
    x_vals = np.linspace(0, 15, 500)
    plt.plot(x_vals, p(x_vals), 'r-', lw=2, label='Target p(x)')
    plt.title(f'SIR with {k} particles (Uniform proposal distribution)')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

for k in k_values:
    resampled = SIR(k, sample_normal, normal_pdf)
    plt.figure()
    plt.hist(resampled, bins=50, density=True, alpha=0.6, label='Resampled particles')
    x_vals = np.linspace(0, 15, 500)
    plt.plot(x_vals, p(x_vals), 'r-', lw=2, label='Target p(x)')
    plt.title(f'SIR with {k} particles (Normal proposal distribution)')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
        
    
    