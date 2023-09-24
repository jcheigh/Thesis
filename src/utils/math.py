import os 
import csv
from sage.all import *
import numpy as np
import matplotlib.pyplot as plt 
import random
from tqdm import tqdm

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

    This can be optimized using binary search but doesn't really matter
    """
    n = 0 
    while primes(n) < p:
        n += 1 
        
    return n

def get_range(p: int) -> int:
    """
    Returns k iff p is the kth prime
    """
    # Initial bounds for binary search
    low = 0
    high = 1
    
    # Increase the upper bound until the prime at that position is greater than p
    while primes(high) < p:
        high *= 2

    while low <= high:
        mid = (low + high) // 2
        prime_at_mid = primes(mid)
        
        if prime_at_mid < p:
            low = mid + 1
        elif prime_at_mid > p:
            high = mid - 1
        else:
            return mid
    
    raise ValueError(f"{p} is not a prime as per the given primes function.")


def get_prime_range(p_min: int, p_max: int) -> list:
    """
    Returns list of primes p s.t. p_min <= p <= p_max
    """
    k_min = get_range(p_min)
    k_max = get_range(p_max)
    
    return [primes(k) for k in range(k_min, k_max + 1)] 

def sample_primes(n: int, p_min: int = 100000, p_max: int = 200000):
    """
    Returns list of n primes sampled from primes s.t. p_min <= p <= p_max

    Defaults:
        p_min: 100000
        p_max: 200000
    """
    prime_list = get_prime_range(p_min, p_max)

    try:
        return random.sample(prime_list, n)
    except ValueError:
        print(f"You can't sample more than {len(prime_list)}")


def S(p: int, x_min: int = 1, x_max: int = None) -> list:
    """
    Returns [S_p(x_min), ..., S_p(x_max)], where S_p(x) is character sum of legendre symbol

    Defaults:
        x_min = 1
        x_max = p
        
    S_p(x) := sum(legendre(a, p) for a in range(1, x + 1)) 

    DP Approach:
        Base Case:           S_p(0) = 0
        Recurrence Relation: S_p(x) = S_p(x-1) + legendre(x, p)  
    """
    if x_max is None:
        x_max = p 

    first_val = sum([legendre(a, p) for a in range(1, x_min + 1)]) 
    result = [first_val] 
    
    for x in range(x_min + 1, x_max + 1):
        # recurrence relation
        val = result[-1] + legendre(x, p)
        result.append(val)
    
    return result

def T(x: int, p: int) -> float: 
    """
    Returns T_p(x) := sin(x) iff p % 4 == 1 else 1 - cos(x)
    T(x) in Leo's Desmos 

    Used in Fourier_Expansion function (see below)
    """
    return sin(x) if p % 4 == 1 else 1 - cos(x) # if p % 4 == 3
    
def Fourier_Expansion(p: int, x_min: int = 1, x_max = None, H = None) -> list:
    """
    Returns [S_p(x_min),...,S_p(x_max)], where S_p(x) is main term of Polya's 
    Fourier Expansion with H = H

    Defaults:
        x_min = 1
        x_max = p
        H     = floor(ln(p)^2)

    Uses Leo's idea of pairing up sums (S_p(-x) + S_p(x))
    Derivation in Justin Cheigh's .tex documentation 
    """
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


def get_data(p: int, diff: np.array):
    ### writing data to csv file
    max_error = round(np.max(diff), 2)
    max_index = np.argmax(diff)
    min_error = round(np.min(diff), 2)
    min_index = np.argmin(diff)

    new_row = [[p, max_error, min_error, max_index, min_index]]
    
    path = os.path.join("/Users", "jcheigh", 'Thesis')
    csv_name = f"{path}/data.csv"

    with open(csv_name, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(new_row)
    
    ### writing list to Thesis/list
    path = os.path.join(path, "data")
    txt_name = f"{path}/Prime = {p} Diff List.txt"

    with open(txt_name, "w") as file:
        for elem in diff:
            file.write(str(round(elem, 2)) + "\n")

    ### writing plot to Thesis/plots
    plt.figure(figsize = (20,10))
    x = list(range(1, p + 1)) # x values 
    plt.plot(x, diff, label = "S_p(x) - Fourier Exp(x)")
    
    error_term = floor(ln(p))
    plt.axhline(y = error_term, color='r', linestyle='--', label= "Log Error")
    plt.axhline(y = -1 * error_term, color='r', linestyle='--')
  
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel("x")
    plt.ylabel('y')
    plt.title(f"Prime = {p}")
    
    # plot saved
    plot_path = f"/Users/jcheigh/Thesis/Plots/Prime = {p} Difference Plot.png"
    plt.savefig(plot_path)
    plt.close()         
    print(f"\nData Collection for Prime = {p} Complete\n")

def update_lst(p: int):
    """
    Creates D(p) := S(p) - Fourier_Expansion(p) for large p 

    Basically just updates a .txt file as to allow you to do this
    over multiple runs
    """
    for _ in tqdm(range(10)):
        data_path = os.path.join("/Users", "jcheigh", 'Thesis', "data")
        txt_name = f"{data_path}/Prime = {p} Diff List.txt"

        if not os.path.exists(txt_name):
            actual  = S(p, x_min = 1, x_max = 150000)
            fourier = Fourier_Expansion(p, x_min = 1, x_max = 150000)
            diff = np.array(actual) - np.array(fourier)

            with open(txt_name, "w") as file:
                for elem in diff:
                    file.write(str(round(elem, 2)) + "\n")
        else:
            with open(txt_name, "r") as txt_file:
                elems = txt_file.read().splitlines()
                x_min = len(elems) + 1
                elems_left = p - x_min
                if elems_left <= 150000:
                    actual = S(p, x_min = x_min, x_max = p)
                    fourier = Fourier_Expansion(p, x_min = x_min, x_max = p)
                else:
                    x_max = x_min + 150000
                    actual = S(p, x_min = x_min, x_max = x_max)
                    fourier = Fourier_Expansion(p, x_min = x_min, x_max = x_max)

                diff = np.array(actual) - np.array(fourier)
            
            with open(txt_name, "w") as txt_file:
                for elem in elems:
                    txt_file.write(elem + "\n")

                for elem in diff:
                    txt_file.write(str(round(elem, 2)) + "\n")

def add_data(p: int):
    """
    Collects relevant data on prime p

    Define D(p) := S(p) - Fourier_Expansion(p)
    Then this function does the following:
        Collects data:
            max, argmax, min, argmin of D(p)
            write this data to data.csv 
        Plots D(p):
            Write plot to .png file in Thesis/plots
        Write D(p) to Thesis/data
    
    Takes a long time, maybe around 30 minutes for a prime p ~ 150k. This is due to the Fourier 
    function, as the S(p) one is more optimized with dynamic programming.

    To optimize this, I break into cases where p > 200k or not. 
        If p > 200k:
            You need to call update_lst(p) a few times first.
            This create D(p) by breaking into components (thus can be done over multiple runs)
        Else:
            I just add data all in one run
    """
    plot_path = f"/Users/jcheigh/Thesis/Plots/Prime = {p} Difference Plot.png"

    if os.path.exists(plot_path):
        print(f"Already Analyzed Prime = {p}")
        return None 

    if p <= 200000:
        actual = S(p)
        fourier = Fourier_Expansion(p)
        diff = np.array(actual) - np.array(fourier)
        get_data(p, diff)
    else:
        data_path = os.path.join("/Users", "jcheigh", 'Thesis', "data")
        txt_name = f"{data_path}/Prime = {p} Diff List.txt"

        if not os.path.exists(txt_name):
            raise Exception(f"For primes larger than 200k please call create_lst({p}) first")

        with open(txt_name, "r") as txt_file:
            elems = txt_file.read().splitlines()
        
        diff = np.array([float(elem) for elem in elems])

        if len(diff) < p:
            raise Exception(f"For primes larger than 200k please call create_lst({p}) first")

        get_data(p, diff)

if __name__ == "__main__":
    large_prime = 1000003
    add_data(large_prime)