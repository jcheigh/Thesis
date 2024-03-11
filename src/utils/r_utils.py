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
    from math_utils import legendre, S, Fourier_Expansion
except:
    from utils.math_utils import legendre, S, Fourier_Expansion 
import time
import seaborn as sns


def compute_R(x, H=30):
    """
    computes [R_H(x_i) for x_i in x]
    """
    y = x % (2 * np.pi)
    y[y == np.pi] = 0  # Handle the special case where y equals pi

    # Calculating R for all y values
    n = np.arange(1, H + 1)
    sin_term = np.sin(np.outer(y, n)) / n
    return (np.pi - y) / 2 - np.sum(sin_term, axis=1)

def compute_R_sum(k, max_H):
    """
    computes [sum_{s <= (k-1)/2} |R_H(x)| for H in range(1, max_H +1)


    Computes the sum of R(2pi * s / k) and the sum of the absolute value of R(2pi * s / k)
    for different H values in a vectorized manner.

    :param k: The prime number.
    :param H_vals: A list of H values.
    :return: Two arrays - sum of R values and sum of absolute R values for each H.
    """
    H_vals = list(range(1, max_H + 1))
    s_values = np.arange(1, int((k+1)/2))
    x_values = 2 * np.pi * s_values / k

    sum_abs_R = []
    for H in tqdm(H_vals):
        R_values = compute_R(x_values, H)
        sum_abs_R.append(np.sum(np.abs(R_values)))

    return np.array(sum_abs_R)

def compute_R_char_sum(k, max_H):
    """

    Computes the sum of R(2pi * s / k) and the sum of the absolute value of R(2pi * s / k)
    for different H values in a vectorized manner.

    :param k: The prime number.
    :param H_vals: A list of H values.
    :return: Two arrays - sum of R values and sum of absolute R values for each H.
    """
    H_vals = list(range(1, max_H + 1))
    s_values = np.arange(1, int((k+1)/2))
    x_values = 2 * np.pi * s_values / k
    R_char_sum = []
    for H in tqdm(H_vals):
        R_values = compute_R(x_values, H)
        char_sum = S(k, half=True)[1:]
        R_char_sum.append(np.sum(R_values * char_sum))

    return np.array(R_char_sum)


def compute_R_diff_sum(k, H):
    """
    Computes the sum of |R(2πs/k) - R(2π/k * (s-N))| for s in range(1, k+1)
    in a vectorized manner for different values of N.

    :param k: The prime number.
    :param H: The value of H.
    :param N: Array of N values.
    :return: Array of sums for each N.
    """
    N = np.arange(1, k+1)
    s_values = np.arange(1, k + 1)
    x_values = 2 * np.pi * s_values / k

    R_values = compute_R(x_values, H)

    sum_diff = []
    for n in tqdm(N):
        shifted_s_values = np.arange(1, k + 1) - n
        shifted_x_values = 2 * np.pi * shifted_s_values / k
        shifted_R_values = compute_R(shifted_x_values, H)
        sum_diff.append(np.sum(np.abs(R_values - shifted_R_values)))

    return np.array(sum_diff)

def leg_r_sum(k, N, H=30):
    s_vals = np.arange(1, k+1)
    leg = np.array([legendre(s,k) for s in s_vals])
    r_vals = compute_R(2*np.pi * s_vals / k, H)
    r_n_vals = compute_R(2*np.pi * (s_vals - N) / k, H)
    return np.sum(leg * (r_vals - r_n_vals))

def get_diff(k, H=None):
    s_vals = np.arange(1, (k+1)//2)
    r_vals = compute_R(2*np.pi*s_vals/k, H)
    R_total = np.zeros((k-1)//2)
    
    for N in tqdm(range(1, (k+1)//2)):
        # Using rotation properties for back and forward computation for each N
        back = np.concatenate((r_vals[-N:], r_vals[:-N]))
        forward = np.concatenate((r_vals[N:], r_vals[:N]))
        
        # Compute R for current N using the precomputed Legendre symbols and the R_H differences
        R = np.sum(abs((back - forward)))
        R_total[N-1] = abs(R)
        del R
        del back
        del forward
    
    del s_vals
    return R_total

def polya_error(k, H=None):
    """
    Computes the Polya error vector R for all N in range(1, k), for a given prime k.
    If H is not provided, it defaults to floor((ln(k)) ** 2).
    """
    if H is None:
        H = int(np.floor(np.log(k) ** 2))
    
    # Precompute Legendre symbols for s in range(1, (k+1)//2)
    s_vals = np.arange(1, (k+1)//2)

    legendre_vals = np.array([legendre(s, k) for s in s_vals])
    # Precompute R_H(2pi*s/k) for s in s_vals
    r_vals = compute_R(2*np.pi*s_vals/k, H)
    
    # Initialize R vector for all N
    R_total = np.zeros((k-1)//2)
    
    for N in tqdm(range(1, (k+1)//2)):
        # Using rotation properties for back and forward computation for each N
        back = np.concatenate((r_vals[-N:], r_vals[:-N]))
        forward = np.concatenate((r_vals[N:], r_vals[:N]))
        
        # Compute R for current N using the precomputed Legendre symbols and the R_H differences
        R = np.sum(legendre_vals * (back - forward))
        R_total[N-1] = R
        del R
        del back
        del forward
    
    del s_vals
    del legendre_vals
    return R_total

def test1():
    for k in [1009]:
        vals = np.arange(100, 500)
        values = []
        others = []
        for H in tqdm(vals):
            error = polya_error(k, H)
            #other = get_diff(k, H)
            max_error = max(max(error), -1 * min(error))
            #max_other = max(max(other), -1 * min(other))
            values.append(max_error)
            #others.append(max_other)
        
        plt.figure(figsize=(10,6))
        plt.scatter(vals, (np.sqrt(k)/np.log(vals)), label='sqrt(k)log k/H error')
        plt.scatter(vals, values, label='values')
        plt.legend()
        plt.show()




def test():
    k = 1009
    H = 30
    vals = list(range(1, k))
    s_vals = np.arange(1, (k+1)//2)
    legendre_vals = np.array([legendre(s, k) for s in s_vals])
    for N in vals:
        back = s_vals - N 
        forwards = s_vals + N
        br_vals = compute_R(2*np.pi*back/k, H)
        fr_vals = compute_R(2*np.pi*forwards/k, H)
        plt.figure(figsize=(10,6))
        plt.scatter(s_vals[:10], (fr_vals - br_vals)[:10])
        plt.scatter(s_vals[:10], legendre_vals[:10], color='r')
        plt.title(f'{N}')
        plt.show()

def compute_integral(H, k, s, N):
    lower_limit = 2 * np.pi / k * (s - N)
    upper_limit = np.pi
    result, _ = quad(lambda x: np.sin(H*x) / np.tan(x/2), lower_limit, upper_limit)
    return result

def compute_J(H, k):
    ### same int bounds
    s_vals = np.arange(1, k)
    leg_vals = np.array([legendre(s, k) for s in s_vals])

    J = []
    for N in range(1, k):
        integral_vals = np.array([compute_integral(H, k, s, N) for s in s_vals])
        J_N = np.sum(leg_vals * integral_vals)
        J.append(J_N)
    
    return -1 / (2*np.pi) * np.array(J)

def compute_integral1(H, k, s, N):
    # Define the lower limit based on s < N or N <= s <= k
    if s < N:
        lower_limit = 2 * np.pi + 2 * np.pi / k * (s - N)
    elif s == N:
        return 0
    else:  # N <= s <= k
        lower_limit = 2 * np.pi / k * (s - N)
    upper_limit = np.pi
    
    # Compute the integral using the defined limits
    result, _ = quad(lambda x: np.sin(H*x) / np.tan(x/2), lower_limit, upper_limit)
    return result

def compute_J1(H, k):
    s_vals = np.arange(1, k)
    leg_vals = np.array([legendre(s, k) for s in s_vals])

    J = []
    for N in range(1, k):
        # Vectorizing the integral computation might be challenging due to the dependency of the limits on s and N
        # Therefore, a loop is used here for clarity and simplicity
        integral_vals = np.array([compute_integral1(H, k, s, N) for s in s_vals])
        J_N = np.sum(leg_vals * integral_vals)
        J.append(J_N)
    
    return -1 / (2*np.pi) * np.array(J)

if __name__ == "__main__":
    H = 10 
    k = 1009
    for N in range(1, 500):
        for s in range(1, k, 10):
            result, _ = quad(lambda x: np.sin(H*x) / np.tan(x/2), 2*np.pi/k * (s-N), np.pi)
            result1, _ = quad(lambda x: np.sin(H*x) / np.tan(x/2), 2*np.pi - 2*np.pi/k * (s+N), np.pi)
            print(result)
            print(result1)
            print(f'N:{N},s:{s}, {result + result1}')
            print('=' * 20)
    k = 257
    H = 5
    x = np.arange(1, k)
    char = S(k)
    char = char[1:]
    leg_vals = [legendre(s, k) for s in x]
    J_term = compute_J(H, k)
    J1_term = compute_J1(H, k)
    plt.figure(figsize=(10,6))
    plt.scatter(x, -J_term, label='J')
    plt.scatter(x, char, label='char')
    plt.legend()
    plt.show()
"""
        plt.figure(figsize=(10,6))
        plt.scatter(x, J_term  + char + leg_vals, label='diff')
        plt.scatter(x, J1_term, label='J')
        plt.legend()
        plt.show()
"""
"""
    k = 10477
    H = int(np.floor(np.log(k) ** 2))
    error = S(k) - Fourier_Expansion(k)
    error = error[1: (k+1)//2]
    s_vals = np.arange(1, k+1)
    legendre_vals = np.array([legendre(s, k) for s in s_vals])
    r_vals = compute_R(2*np.pi*s_vals/k, H)

    def xi(s, N, H, k, diff):
        if diff:
            if s < N:
                a = 2 * np.pi + (2 * np.pi / k) * (s - N)
            else:  # N < s <= k
                a = (2 * np.pi / k) * (s - N)
        else:
            a = (2 * np.pi / k) * (s - N)
        upper_bound = np.pi
        
        # Compute the integral from a to pi
        integral_value, _ = quad(lambda x: np.sin(H*x) / np.tan(x/2), a, upper_bound)
        return integral_value

    # Define the J function
    def J(H, N, k, diff=False):
        sum_result = 0
        for s in range(1, k + 1):
            legendre_value = legendre(s, k)
            xi_value = xi(s, N, H, k, diff)
            sum_result += legendre_value * xi_value
        return -1 / (2* np.pi) * sum_result

    def O(H, N, k):
    # Initialize the total sum
        total_sum = 0
        
        # Sum for s < N
        for s in range(1, N):
            a = 2 * np.pi + (2 * np.pi / k) * (s - N)
            upper_bound = np.pi
            integral_value, _ = quad(lambda x: np.sin(H*x) / np.tan(x/2), a, upper_bound)
            total_sum += legendre(s, k) * integral_value
        
        # Sum for N < s <= k
        for s in range(N+1, k + 1):
            a = (2 * np.pi / k) * (s - N)
            upper_bound = np.pi
            integral_value, _ = quad(lambda x: np.sin(H*x) / np.tan(x/2), a, upper_bound)
            total_sum += legendre(s, k) * integral_value
        
        return -1 / (2 * np.pi) * total_sum

    def B(H, N, k, diff=True):
        total_sum = 0

        # Define the integrand function for the B function
        def integrand(t, H):
            if not diff:
                return np.sin((H + 0.5) * t) / np.sin(t / 2)
            else:
                return (np.cos(H*t) + np.sin(H*t)/np.tan(t/2))

        # Sum for s < N
        for s in range(1, N):
            a = 2 * np.pi + (2 * np.pi / k) * (s - N)
            upper_bound = np.pi  # Assuming the upper bound is 2*pi for the integral
            integral_value, _ = quad(lambda t: integrand(t, H), a, upper_bound)
            total_sum += legendre(s, k) * integral_value

        # Sum for N < s <= k
        for s in range(N + 1, k + 1):
            a = (2 * np.pi / k) * (s - N)
            upper_bound = np.pi  # Assuming the upper bound is 2*pi for the integral
            integral_value, _ = quad(lambda t: integrand(t, H), a, upper_bound)
            total_sum += legendre(s, k) * integral_value

        return -1 / (2 * np.pi) * total_sum


    for N in range(1, k, 20):
        shifted_r = compute_R(2*np.pi/k * (s_vals - N), H)
        #polya_error = 1/np.pi * sum(legendre_vals * (r_vals - shifted_r)) + (legendre(N, k))
        #sym_equation = -1/np.pi * sum(legendre_vals * shifted_r) + (legendre(N, k))
        #final = -legendre(H, k) * np.sqrt(k) * np.sin(2*np.pi*H*N/k)/(2*np.pi*H) + J(H, N, k) + legendre(N, k)
        #other1 = 1/(2*np.pi*H) * sum(legendre_vals * np.sin(2*np.pi * H/k * (s_vals - N))) + O(H, N, k)
        #print(abs(B(H, N, k) - error[N-1]))
        #print(abs(B(H, N, k) - error[N-1]))
        #print(abs(final - error[N-1]))
        #assert abs(other1- error[N-1]) < 1e-13


        final1 = -legendre(H, k) * np.sqrt(k) * np.sin(2*np.pi*H*N/k)/(2*np.pi*H) + legendre(N, k)
        final2 = J(H, N, k) + S(k)[N-1]
        print(f"{abs(error[N-1] - final1 - final2)}")
        #print(f'Error(N): {error[N-1]}')
        #print(f"N: {N}, T1: {final1}, T2: {final2}")

""" 