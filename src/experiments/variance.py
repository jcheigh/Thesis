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
from utils.math_utils import Fourier_Expansion, is_prime, sample_primes,S

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")
PLOT_PATH = os.path.join(MAIN_PATH, "plots")
PATH      = f'{DATA_PATH}/variance.csv'

import numpy as np
import csv
from multiprocessing import Pool

def compute_stats(args):
    p, half = args
    print(f'Analyzing Prime = {p}')
    char_lst = S(p, half=half) 
    four_lst = Fourier_Expansion(p, half=half)
    error_lst = char_lst - four_lst
    mean = np.mean(error_lst)

    char_squared = char_lst ** 2
    four_squared = four_lst ** 2
    char_four    = char_lst * four_lst

    data = {
        "prime"             : p, 
        "variance"          : np.var(error_lst),
        "Sp^2"              : np.sum(char_squared) / p,
        "Fp^2"              : np.sum(four_squared) / p,
        "Mp^2"              : (mean ** 2),
        "-2SF"              : -2 * np.sum(char_four) / p,
        "2MF"               : 2 * mean * np.sum(four_lst) / p,
        "-2MS"              : -2 * mean * np.sum(char_lst) / p,
        'half'              : half,
    }
    del error_lst
    del char_lst
    del char_four 
    del four_lst
    del char_squared
    del four_squared

    with open(PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writerow(data)

    del data
    print(f"Finished Analyzing Prime = {p}")

def add_primes(primes, multithreading=True, half=False, num_processes=5):
    
    if multithreading:
        with Pool(processes=num_processes) as pool:
            pool.map(compute_stats, [(p, half) for p in primes])
    else:
        for p in primes:
            compute_stats((p, half))
    

def get_analyzed_primes():
    """Fetches the primes that have already been analyzed from the CSV."""
    analyzed_primes = set()
    
    try:
        with open(PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                analyzed_primes.add(int(float(row["prime"])))
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
    primes = get_primes(1000,7, 100)
    add_primes(primes,multithreading=True, half=False, num_processes=5)
