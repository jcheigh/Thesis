import os 
import csv
from sage.all import kronecker, Primes, sin, cos, floor, pi, ln, prime_pi
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm
from scipy.special import comb
from scipy.integrate import quad
import time
import seaborn as sns

def legendre(a: int, p: int) -> int:
    """
    Computes legendre symbol a on p using sage's built in kronecker function

    The kronecker symbol is a generalization of the legendre symbol.
    The first line is almost certainly built in:
        (b on p) == (a on p) if a == b (mod p)
    """
    b = a % p 
    return kronecker(b, p)

def primes(k: int) -> int:
    """
    Returns kth prime, where p_0 = 2
    """
    P = Primes()
    return P.unrank(k)

def get_range(p: int) -> int:
    """
    Returns k iff p is the kth prime

    Uses fact that p_k => pi(p_k) = k + 1
    """
    return prime_pi(p) - 1
  
def is_prime(n: int) -> bool:
    """
    Returns True if n is a prime number, else False, using the primes function
    """
    if n < 2:
        return False
    limit = int(n**0.5)
    
    k = 0
    while True:
        prime_k = primes(k)
        if prime_k > limit:
            break
        if n % prime_k == 0:
            return False
        k += 1
        
    return True
    
def sample_primes(n: int, p_min: int = 100000, p_max: int = 200000, mod=None):
    """
    Returns list of n primes sampled from primes s.t. p_min <= p <= p_max

    Defaults:
        p_min: 100000
        p_max: 200000
    """
    k_min = get_range(p_min)
    k_max = get_range(p_max)

    if mod is not None:
        smpl = random.sample(list(range(k_min, k_max+1)), min(3*n, k_max-k_min + 1))
    else:
        smpl = random.sample(list(range(k_min, k_max+1)), min(n, k_max-k_min + 1))

    prime_list = [primes(k) for k in smpl]

    if mod is not None:
        prime_list = [prime for prime in prime_list if prime % 4 == mod]

    try:
        return random.sample(prime_list, n)
    except ValueError:
        print(f"You can't sample more than {len(prime_list)}")


def S(p: int, half=False) -> list:
    """
    Returns [S_p(x_min), ..., S_p(x_max)], where S_p(x) is character sum of legendre symbol

    Defaults:
        x_min = 0
        x_max = p - 1
        
    S_p(x) := sum(legendre(a, p) for a in range(1, x + 1)) 

    DP Approach:
        Base Case:           S_p(0) = 0
        Recurrence Relation: S_p(x) = S_p(x-1) + legendre(x, p)  
    """
    x_min = 0

    if half:
        x_max = (p-1)//2
    else:
        x_max = p-1

    first_val = sum([legendre(a, p) for a in range(1, x_min + 1)]) 
    result = [first_val] 
    
    for x in range(x_min + 1, x_max + 1):
        # recurrence relation
        val = result[-1] + legendre(x, p)
        result.append(val)
    
    return np.array(result)

def Fourier_Expansion(p: int, H = None, half=False, start=1) -> list:
    """
    Returns [S_p(x_min),...,S_p(x_max)], where S_p(x) is main term of Polya's 
    Fourier Expansion with H = H

    Defaults:
        H     = floor(ln(p)^2)

    Uses Leo's idea of pairing up sums (S_p(-x) + S_p(x))
    Derivation in Justin Cheigh's .tex documentation 

    Also uses np vectorization, which speeds everything dramatically 
        before p = 196993 probably took almost an hour. Now it's ~.5 seconds 

    The idea is to precompute:
        reciprocals = [1/1, 1/2, 1/3, ..., 1/H] 
        leg_symbols = [legendre(1, p), ..., legendre(H, p)]
        T is a p x H matrix where the kth row is [T(2pi k/p), ..., T(2pi Hk/p)]
            Here T(x) = sin(x) if p == 1 mod 4 else 1 - cos(x)

        Then sqrt(p)/pi * reciprocals * leg_symbols can just be matrix multiplied with
        T transposed to get the desired result.
    """
    if H is None:
        H = floor((ln(p)) ** 2)

    if half:
        xvals = np.arange(0, (p-1)//2 + 1)
    else:
        xvals = np.arange(0, p)

    nvals = np.arange(start, H + 1)
    leg_symbols = np.array([legendre(n, p) for n in nvals])
    reciprocals = 1.0 / nvals 

    # T(x)
    if p % 4 == 1: 
        T_values = np.sin(2 * np.pi * np.outer(xvals, nvals) / p)
    else:
        T_values = 1 - np.cos(2 * np.pi * np.outer(xvals, nvals) / p)

    Fp = np.sqrt(p) / np.pi * np.sum(leg_symbols * reciprocals * T_values, axis=1)
    
    del T_values
    del leg_symbols
    del reciprocals
    del nvals
    del xvals
    
    return Fp

def scatter(x, y, xlabel, ylabel, title=None):
    # both numeric
    plt.figure(figsize=(10, 8))
    sns.set(style='ticks')
    sns.set_context("poster")

    if title is None:
        title = f"{ylabel} by {xlabel}"
    
    p = sns.regplot(x=x, y=y)
        
    p.axes.set_title(title, fontsize=25)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.show()
    return p

def histplot(x, bins=30, xlabel=None, title=None):
    # distribution of numerical
    plt.figure(figsize = (10,8))
    p = sns.histplot(x, bins=bins, kde=True, fill=True,
                    edgecolor="black", linewidth=3
                    )

    p.axes.lines[0].set_color("orange")
    
    if xlabel is None:
        xlabel = str(x)
        
    if title is None:
        title = f"{xlabel} Distribution"
    
    plt.ylabel("Count", fontsize = 20)
    plt.xlabel(xlabel, fontsize = 20)
    plt.title(title, fontsize = 25)
    plt.show()

if __name__ == "__main__":
    p = 1009
    H_range = list(range(1, p, 5))
    char_sum = S(p)
    for H in H_range:
        error = char_sum - Fourier_Expansion(p, H)
        print(f'H: {H}, {error[105]}')