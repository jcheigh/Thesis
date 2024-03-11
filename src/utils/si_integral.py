from scipy.special import sici
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
try:
    from math_utils import sample_primes
    from r_utils import compute_R_sum
except:
    from utils.math_utils import sample_primes
    from utils.r_utils import compute_R_sum
from tqdm import tqdm

memoized_si = {}

def compute_si(x):
    """Compute Si(x) using memoization to avoid redundant calculations."""
    if x not in memoized_si:
        si_val, _ = sici(x)
        memoized_si[x] = si_val
    return memoized_si[x]

def compute_sums(k, H):
    si_constant = compute_si(np.pi/2 * (2*H+1))
    sum1 = 0
    sum2 = 0
    pi_half = np.pi / 2

    for s in range(int(k/2)):
        si_val = compute_si(np.pi * s / k * (2*H+1))
        sum1 += abs(si_constant - si_val)
        sum2 += abs(si_constant - pi_half)

    return sum1, sum2, sum2 +  k/(np.pi * (2*H+1))



def plot_sums_vs_h(k, max_H, error_line=lambda k, H: k/H, start=0):
    sum1_vals = []
    sum2_vals = []
    sum3_vals = []
    error_line_vals = []
    h_vals = list(range(1, max_H+1))
    r_vals = compute_R_sum(k, max_H)
    # Compute the sums for each H
    for H in tqdm(h_vals):
        sum1, sum2, sum3 = compute_sums(k, H)
        sum1_vals.append(sum1)
        sum2_vals.append(sum2)
        sum3_vals.append(sum3)
        error_line_vals.append(error_line(k, H))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(h_vals[start:], sum1_vals[start:], label="(3): |Si(pi/2 * (2H+1)) - Si(pi s/k* (2H+1))|", marker='o')
    plt.plot(h_vals[start:], sum2_vals[start:], label="(5): |Si(pi/2 * (2H+1)) - pi/2|", marker='x')
    plt.plot(h_vals[start:], sum3_vals[start:], label="(5) |Si(pi/2 * (2H+1)) - pi/2| + Extra", marker='x')
    plt.plot(h_vals[start:], r_vals[start:], label="(2) |R(2pi s/k)|", marker='x')
    plt.plot(h_vals[start:], error_line_vals[start:], 'r--', label="(1) Error: (k/H)", linewidth=2, markersize=12, linestyle='dotted')
    
    # Labeling the plot
    plt.xlabel("H values")
    plt.ylabel("Computed Sums and Error Line")
    plt.title(f"k = {k} Sums")
    plt.legend()
    plt.grid(True)
    plt.show()

def test_si(k, H, start=0,error_line=lambda x: 1/x):
    def error_line(x):
        return min(np.pi/2, 1/x)
    s_vals = np.arange(1, (k+1)//2)
    x_vals = np.pi * s_vals/ k * (2*H+1)
    si_vals = np.array([abs(np.pi/2 - sici(x)[0]) for x in x_vals])

    error_line_vals = np.array([error_line(x) for x in x_vals])
    print(si_vals[:20])
    print('---')
    print(error_line_vals[:20])
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(s_vals[start:], si_vals[start:], label="|Si(x)|", marker='None', linestyle='-')
    plt.plot(s_vals[start:], 10/H * np.ones_like(np.array(s_vals)[start:]), label="pi/4H", marker='None', linestyle='-')
    plt.plot(s_vals[start:], error_line_vals[start:], 'r--', label="Error Line (1/x)", linewidth=2, linestyle='dotted')
    
    # Labeling the plot
    plt.xlabel("s")
    plt.ylabel("|si(x)| and Error Line")
    plt.title(f"k = {k}, H = {H}: |si(x)| vs. x with Error Line")
    plt.legend()
    plt.grid(True)
    plt.show()


memoized_integrals = {}

def integral_memoizer(func, x):
    """Memoizes and computes the integral based on func and x."""
    if (func.__name__, x) not in memoized_integrals:
        result, _ = quad(func, 0, np.inf)
        memoized_integrals[(func.__name__, x)] = result
    return memoized_integrals[(func.__name__, x)]

def compute_2ints(k, H):
    int1_values = []
    int2_values = []
    
    for s in range(1, (k+1)//2):
        x = np.pi * s / k * (2*H + 1)
        
        func1 = lambda t: np.sin(t) / (t + x)
        func2 = lambda t: np.cos(t) / (t + x)
        
        int1 = np.cos(x) * integral_memoizer(func1, x)
        int2 = np.sin(x) * integral_memoizer(func2, x)
        
        int1_values.append(int1)
        int2_values.append(int2)
    
    return int1_values, int2_values

def plot_2ints(k, H, error_line=lambda k, H: k/H):
    int1_values, int2_values = compute_2ints(k, H)
    s_values = range(1, (k+1)//2)
    error_line_values = [error_line(k, H) for _ in s_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, int1_values, label="Integral 1 (sin(t)/(t+x)", marker='o')
    plt.plot(s_values, int2_values, label="Integral 2 (cos(t)/(t+x)", marker='x')
    plt.plot(s_values, np.array(int1_values) + np.array(int2_values), label="Sum of Int", marker='x')
    #plt.plot(s_values, error_line_values, 'r--', label="Error Line (k/H)", linewidth=2, linestyle='dotted')
    
    plt.xlabel("s")
    plt.ylabel("Integral Values")
    plt.title(f"k = {k}, H = {H}: Integral Values vs. s")   
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_2ints_abs(k, H, error_line=lambda k, H: k/H):
    int1_values, int2_values = compute_2ints(k, H)
    int1_values, int2_values = np.array(int1_values), np.array(int2_values)
    s_values = range(1, (k+1)//2)
    error_line_values = [error_line(k, H) for _ in s_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, abs(int1_values), label="Integral 1 |(sin(t)/(t+x)|", marker='o')
    #plt.plot(s_values, abs(int2_values), label="Integral 2 |(cos(t)/(t+x)|", marker='x')
    plt.plot(s_values, abs((int1_values)) + abs(np.array(int2_values)), label="Sum of |Int| (tri)", marker='x')
    #plt.plot(s_values, abs((int1_values) + np.array(int2_values)), label="|Sum of Int|", marker='x')
    #plt.plot(s_values, error_line_values, 'r--', label="Error Line (k/H)", linewidth=2, linestyle='dotted')
    
    plt.xlabel("s")
    plt.ylabel("Integral Values")
    plt.title(f"k = {k}, H = {H}: Abs Integral Values vs. s")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    primes = sample_primes(2, 10000, 20000)
    
    for k in primes:
        H_vals = [1, int(np.log(k)), int(np.sqrt(k)), k, k**2]
        for H in H_vals:
            test_si(k, H, start=100)
            #plot_2ints(k, H)
            #plot_2ints_abs(k, H)
    
