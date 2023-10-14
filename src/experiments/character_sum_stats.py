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
from utils.math_utils import S, is_prime, sample_primes

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")
PLOT_PATH = os.path.join(MAIN_PATH, "plots")
PATH      = f'{DATA_PATH}/char sum stats.csv'

import numpy as np
import csv
from multiprocessing import Pool

def compute_stats(p):
    print(f'Analyzing Prime = {p}')
    char_sum_lst = S(p)
    
    data = {
        "prime"   : p, 
        "max"     : np.max(char_sum_lst),
        "min"     : np.min(char_sum_lst),
        "mean"    : np.mean(char_sum_lst),
        "median"  : np.median(char_sum_lst),
        "std dev" : np.std(char_sum_lst),
        "x_max"   : np.argmax(char_sum_lst),
        "x_min"   : np.argmin(char_sum_lst)
    }
    
    return data

def add_primes(primes, multithreading=True, half=False):
    
    if multithreading:
        with Pool(processes=6) as pool:
            results = pool.map(compute_stats, primes)
    else:
        results = [compute_stats(p) for p in primes]
    
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
    primes = get_primes(20, 100000, 200000)
    add_primes(primes)
