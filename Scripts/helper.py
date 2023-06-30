from sage.all import *
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os 

def legendre(a, p):
    """
    Computes (a / p) i.e. a on p using built in kronecker function, which 
    is a generalization of the legendre symbol. 
    """
    b = a % p # this optimization is prob built into sage but whatever 
    return kronecker(b, p)

def primes(k):
    """
    Returns kth prime (k = 0 => 2)
    """
    P = Primes()
    return P.unrank(k)

def get_range(p):
    """
    Returns index for prime
    Can be optimized with binary search
    but not a big deal
    """
    n = 0 
    while primes(n) < p:
        n += 1 
        
    return n

def get_prime_range(p_min, p_max):
    k_min = get_range(p_min)
    k_max = get_range(p_max)
    
    return [primes(k) for k in range(k_min, k_max + 1)] 

def sample_primes(n, p_min = None, p_max = None, restraints = None):
    """
    Returns list of n primes sampled from primes s.t. p_min <= p <= p_max
    
    With default n <= 8393

    """
    if p_min is None:
        p_min = 100000
        
    if p_max is None:
        p_max = 200000
        
    prime_list = get_prime_range(p_min, p_max)

    if restraints is not None:
        prime_list = list(set(prime_list) - set(restraints))
          
    return random.sample(prime_list, n)

def S(p, x_min = None, x_max = None):
    """
    Constraints: 0 < x_min < x_max <= p, p odd prime
    
    Defaults: x_min = 1, x_max = p

    Returns: [S_p(x_min), ..., S_p(x_max)] (S_p(x) = Char Sum of Legendre Symbol)
    
    Uses fact that S_p(0) = 0 and S_p(x) = S_p(x-1) + legendre(x, p) recurrence
    """
    if x_min is None:
        x_min = 1 # default value
        
    if x_max is None:
        x_max = p # default value

    
    first_val = sum([legendre(a, p) for a in range(1, x_min + 1)]) # S_p(x_min)
    result = [first_val] # to return
    
    for x in range(x_min + 1, x_max + 1):
        val = result[-1] + legendre(x, p) # recurrence relation
        result.append(val)
    
    return result

def T(x, p):
    """
    T(x) in Leo's Desmos 
    """
    return sin(x) if p % 4 == 1 else 1 - cos(x) # if p % 4 == 3    
    if p % 4 == 1:
        return sin(x)
    
def Fourier_Expansion(p, x_min = None, x_max = None, H = None):
    """
    Constraints: 0 < x_min < x_max <= p, p odd prime, H int
    
    Defaults: x_min = 1, x_max = p, H = floor((ln(p))^2)

    Returns: [S_p(x_min), ..., S_p(x_max)] (S_p(x) = Polya Fourier Expansion)
    
    Uses: Leo's idea of pairing up sums (S_p(-x) + S_p(x))
    """
    if x_min is None:
        x_min = 1 
        
    if x_max is None:
        x_max = p
    
    if H is None:
        H = floor((ln(p)) ** 2)
        
    result = [] 
    C = sqrt(p) / pi
    
    for x in range(x_min, x_max + 1):
        exp = 2 * pi * x / p
        main_term = C * sum(legendre(n, p) * T(n * exp, p) / n for n in range(1, H + 1))
        result.append(main_term.n())             
        
    return result       

def add_data(p):
    x = list(range(1, p + 1)) # x values 
    actual = S(p)
    fourier = Fourier_Expansion(p)

    diff = list(np.array(actual) - np.array(fourier))
    
    min_error, max_error = diff[0], diff[0]
    min_indices, max_indices = [], []
    
    for i, val in enumerate(diff):
        if val == max_error: #update indices list
            max_indices.append(i) 
        
        if val == min_error:
            min_indices.append(i)
            
        if val > max_error: # if larger max error
            max_error = val # set error 
            max_indices = [i] # reset indices list 
        
        if val < min_error: # same with min error 
            min_error = val 
            min_indices = [i]
        
    data = {"prime" : p, "max_error" : max_error, "max_neg_error" : min_error,
           "x_max_error" : max_indices, "x_max_neg_error" : min_indices}
    
    path = f"/Users/jcheigh/Thesis/Data/Prime = {p} Data.txt"
    
    with open(path, "w") as file:
        file.write(str(data))
        
    print(f"Data: {data}")
    
    plt.figure(figsize = (20,10))
    plt.plot(x, diff, label = "S_p(x) - Fourier Exp(x)")
    
    error_term = floor(ln(p))
    plt.axhline(y = error_term, color='r', linestyle='--', label= "Log Error")
    plt.axhline(y = -1 * error_term, color='r', linestyle='--')
  
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel("x")
    plt.ylabel('y')
    plt.title(f"Prime = {p}")
    
    path = f"/Users/jcheigh/Thesis/Plots/Prime = {p} Difference Plot.png"
    plt.savefig(path)
    plt.show()          
    print(f"\nData Collection for Prime = {p} Complete\n")
