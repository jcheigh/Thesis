import sys
sys.path.append('../')
from typing import List, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import random
import time
import re 
import numpy as np
from multiprocessing import Pool

from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import Fourier_Expansion, is_prime, sample_primes

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")
PLOT_PATH = os.path.join(MAIN_PATH, "plots")
PATH      = f'{DATA_PATH}/polya fourier stats.csv'

import numpy as np
import csv
from multiprocessing import Pool

def compute_stats(p, half):
    print(f'Analyzing Prime = {p}')
    fourier_lst = Fourier_Expansion(p, half=half)
    
    data = {
        "prime"   : p, 
        "max"     : np.max(fourier_lst),
        "min"     : np.min(fourier_lst),
        "mean"    : np.mean(fourier_lst),
        "median"  : np.median(fourier_lst),
        "std dev" : np.std(fourier_lst),
        "x_max"   : np.argmax(fourier_lst),
        "x_min"   : np.argmin(fourier_lst)
    }
    
    return data

def compute_stats_wrapper(args):
    return compute_stats(args[0], args[1])

def add_primes(primes, multithreading=True, half=False):
    
    if multithreading:
        with Pool(processes=6) as pool:
            results = pool.map(compute_stats_wrapper,[(p, half) for p in primes])
    else:
        results = [compute_stats(p, half) for p in primes]
    
    # Append results to the CSV
    with open(PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        for row in results:
            writer.writerow(row)

def get_analyzed_primes():
    """Fetches the primes that have already been analyzed from the CSV."""
    analyzed_primes = set()
    
    try:
        with open(PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                analyzed_primes.add(int(row["prime"]))
    except FileNotFoundError:
        pass  # File doesn't exist yet, so no primes have been analyzed

    return analyzed_primes

def get_primes(n, p_min, p_max):
    analyzed_primes = get_analyzed_primes()
    unique_primes = set()

    while len(unique_primes) < n:
        sampled_primes = set(sample_primes(n, p_min, p_max))
        new_primes = sampled_primes - analyzed_primes  # Remove already analyzed primes
        unique_primes.update(new_primes)
        # In case we're running out of primes to sample, ensure we don't enter an infinite loop
        if not new_primes:
            break

    return list(unique_primes)[:n]


if __name__ == "__main__":
    primes = get_primes(200, 1000000, 2000000)
    add_primes(primes)
