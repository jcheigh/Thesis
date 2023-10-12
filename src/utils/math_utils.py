import os 
import csv
from sage.all import kronecker, Primes, sin, cos, floor, pi, ln
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time

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

def get_prime_range(p_min: int, p_max: int) -> list:
    """
    Returns list of primes p s.t. p_min <= p <= p_max
    """
    k_min = get_range(p_min)
    k_max = get_range(p_max)
    
    return [primes(k) for k in range(k_min, k_max + 1)] 

def sample_primes(n: int, p_min: int = 100000, p_max: int = 200000, mod= None):
    """
    Returns list of n primes sampled from primes s.t. p_min <= p <= p_max

    Defaults:
        p_min: 100000
        p_max: 200000
    """
    prime_list = get_prime_range(p_min, p_max)

    if mod is not None:
        prime_list = [prime for prime in prime_list if prime % 4 == mod]

    try:
        return random.sample(prime_list, n)
    except ValueError:
        print(f"You can't sample more than {len(prime_list)}")


def S(p: int, x_min: int = 0, x_max: int = None) -> list:
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
    if x_max is None:
        x_max = p - 1

    first_val = sum([legendre(a, p) for a in range(1, x_min + 1)]) 
    result = [first_val] 
    
    for x in range(x_min + 1, x_max + 1):
        # recurrence relation
        val = result[-1] + legendre(x, p)
        result.append(val)
    
    return np.array(result)

def Fourier_Expansion(p: int, H = None) -> list:
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

    xvals = np.arange(0, p)
    nvals = np.arange(1, H + 1)
    leg_symbols = np.array([legendre(n, p) for n in nvals])
    reciprocals = 1.0 / nvals 

    # T(x)
    if p % 4 == 1: 
        T_values = np.sin(2 * np.pi * np.outer(xvals, nvals) / p)
    else:
        T_values = 1 - np.cos(2 * np.pi * np.outer(xvals, nvals) / p)

    Fp = np.sqrt(p) / np.pi * np.sum(leg_symbols * reciprocals * T_values, axis=1)
    
    return Fp

"""
outdated because np vectorization is crazy crazy fast

def Fourier_Expansion(p: int, x_min: int = 0, x_max = None, H = None) -> list:
    Returns [S_p(x_min),...,S_p(x_max)], where S_p(x) is main term of Polya's 
    Fourier Expansion with H = H

    Defaults:
        x_min = 0
        x_max = p - 1
        H     = floor(ln(p)^2)

    Uses Leo's idea of pairing up sums (S_p(-x) + S_p(x))
    Derivation in Justin Cheigh's .tex documentation 
    if x_max is None:
        x_max = p - 1
    
    if H is None:
        H = floor((ln(p)) ** 2)
        
    result = [] 
    C = np.sqrt(p) / pi
    
    for x in range(x_min, x_max + 1):
        exp = 2 * pi * x / p
        main_term = C * sum(legendre(n, p) * T(n * exp, p) / n for n in range(1, H + 1))
        result.append(main_term.n())             
        
    return result       
"""

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
    start = time.time()
    lst = Fourier_Expansion(7231013)   
    end = time.time()
    print(end-start)
    char = S(7231013)
    data = lst - char 
    prime = 7231013
    fig, ax = plt.subplots(figsize=(20,10))
    x = list(range(0, prime))
    ax.plot(x, data, label="S_p(x) - Fourier Exp(x)")

    mean, sd = np.mean(data), np.std(data)
    largest, smallest = max(data), min(data)
    
    ax.axhline(y=mean, color='black', linestyle='--', label=f"Mean(E_p(x)) = {round(mean, 3)}")
    ax.axhline(y=mean + sd, color='r', linestyle='--', label=f"+1 SD = {round(mean,3)} + {round(sd, 3)}")
    ax.axhline(y=mean - sd, color='r', linestyle='--', label=f"-1 SD = {round(mean,3)} - {round(sd, 3)}")
    ax.axhline(y=largest, color='blue', linestyle='--', label=f"Max(E_p(x)) = {round(largest, 3)}")
    ax.axhline(y=smallest, color='blue', linestyle='--', label=f"Min(E_p(x)) = {round(smallest,3)}")

    error_term = floor(ln(prime))
    ax.axhline(y=error_term, color='green', linestyle='--', label="Log Error")
    ax.axhline(y=-1 * error_term, color='green', linestyle='--')

    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel('error')
    ax.set_title(f"p = {prime} Error Plot (p = {prime % 4} mod 4)")
    MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
    PLOT_PATH = os.path.join(MAIN_PATH, "plots")
    path = f"{PLOT_PATH}/error plots/p = {prime} error plot.jpg"

    fig.savefig(path)