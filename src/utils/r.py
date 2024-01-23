import os 
import csv
from sage.all import kronecker, Primes, sin, cos, floor, pi, ln, prime_pi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from tqdm import tqdm
from scipy.special import comb

import time
import seaborn as sns

from math_utils import sample_primes

PLOT_PATH = os.path.join("/Users", "jcheigh", "Thesis", "plots")

large1 = sample_primes(1, 1000000, 2000000, mod = 1)[0]
large3 = sample_primes(1, 1000000, 2000000, mod = 3)[0]
med1   = sample_primes(1, 100000, 200000, mod = 1)[0]
med3   = sample_primes(1, 100000, 200000, mod = 3)[0]
small1 = sample_primes(1, 1000, 2000, mod = 1)[0]
small3 = sample_primes(1, 1000, 2000, mod = 3)[0]

primes = [large1, large3, med1, med3, small1, small3]
H_vals = [1, 5, 20, 100]
print(primes)
def R(x, H=30):
    # x is a vector 
    y = x % (2 * np.pi)
    y[y == np.pi] = 0  # Handle the special case where y equals pi

    # Calculating R for all y values
    n = np.arange(1, H + 1)
    sin_term = np.sin(np.outer(y, n)) / n
    return (np.pi - y) / 2 - np.sum(sin_term, axis=1)

def create_1i(primes, H_vals):
    for k in tqdm(primes):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        title = f's vs. R(2πs/k) for k = {k}'
        fig.suptitle(title)

        # Compute s values and corresponding R values for each H
        s_values = np.arange(1, k)
        x_values = 2 * np.pi * s_values / k

        for i, H in enumerate(H_vals):
            R_values = R(x_values, H)
            error_values = k / (2 * H * s_values)

            # Plotting
            ax = axs[i // 2, i % 2]
            ax.scatter(s_values, R_values, label=f'H = {H}')
            #ax.plot(s_values, error_values, 'r--', label='Error +')
            #ax.plot(s_values, -error_values, 'r--', label='Error -')
            ax.set_title(f'H = {H}')
            ax.set_xlabel('s')
            ax.set_ylabel(f'R(2πs/{k})')
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        os.makedirs(f'{PLOT_PATH}/r plots', exist_ok=True)
        plt.savefig(f'{PLOT_PATH}/r plots/{title}')

        plt.close(fig)
create_1i(primes, H_vals)

