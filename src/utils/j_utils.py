import os 
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from tqdm import tqdm
from scipy.special import comb
from scipy.integrate import quad
try:
    from math_utils import legendre, S, Fourier_Expansion, sample_primes
    from r_utils import compute_J1
except:
    from utils.math_utils import legendre, S, Fourier_Expansion, sample_primes
    from utils.r_utils import compute_J1
import time
import seaborn as sns

import numpy as np
from scipy.integrate import quad
import math

def compute_integral(s, k, H):
    """Numerically integrate sin(Hx)/tan(x/2) from 2pi*s/k to pi."""
    result, _ = quad(lambda x: np.sin(H * x) / np.tan(x / 2), 2 * np.pi * s / k, np.pi)
    return result

def J(k, H=None):
    if H is None:
        H = math.floor(math.log(k)**2)
    """Perform the vectorized calculation based on the given instructions."""
    # Precompute the Legendre symbol values
    s_values = np.arange(1, (k + 1) // 2)
    leg_vals = np.array([legendre(s, k) for s in s_values])
    leg_vals = np.concatenate([leg_vals, leg_vals[::-1]]) # leg(1, k) to leg(k-1, k)
    
    # Compute the integral values
    int_vals = np.array([compute_integral(s, k, H) for s in s_values])
    int_vals = np.concatenate([int_vals, -int_vals[::-1]]) #1 to k-1

    vec = []
    for N in range(1, (k + 1) // 2):
        if N > 1:
            n_int_vals = np.concatenate([int_vals[1-N:], [0],int_vals[:k - N-1]])
        else:
            n_int_vals = np.concatenate([[0], int_vals[:k-2]])
        vec.append(np.sum(np.array(leg_vals) * np.array(n_int_vals)))

    return -1 / (2 * np.pi) * np.array(vec)

def J_vec(k, H=None):
    if H is None:
        H = math.floor(math.log(k)**2)
    
    # Precompute the Legendre symbol values
    s_values = np.arange(1, (k + 1) // 2)
    leg_vals = np.array([legendre(s, k) for s in s_values])
    leg_vals_full = np.concatenate([leg_vals, leg_vals[::-1]]) # Full array for 1 to k-1
    
    # Compute the integral values
    int_vals = np.array([compute_integral(s, k, H) for s in s_values])
    int_vals_full = np.concatenate([int_vals, -int_vals[::-1]]) # Full array for 1 to k-1
    
    # Matrix of shifted integral values
    int_matrix = np.zeros((len(s_values), k - 1))
    for i in range(len(s_values)):
        if i > 0:
            int_matrix[i, :k - 1 - i] = int_vals_full[i:2*i]
            int_matrix[i, k - 2 - i:] = -int_vals_full[:i]
        else:
            int_matrix[i, 1:] = int_vals_full[:k - 2]
    
    result_vector = -1 / (2 * np.pi) * leg_vals_full @ int_matrix.T

    return result_vector

if __name__ == '__main__':
    prime = 1009
    print(J(1009,5) - J_vec(1009, 5))