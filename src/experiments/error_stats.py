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
PATH      = f'{DATA_PATH}/error stats.csv'

import numpy as np
import csv
from multiprocessing import Pool

def compute_stats(args):
    p, half = args
    print(f'Analyzing Prime = {p}')
    char_lst = S(p, half=half) 
    four_lst = Fourier_Expansion(p, half=half)
    error_lst = char_lst - four_lst
    
    data = {
        "prime"             : p, 
        "max"               : np.max(error_lst),
        "min"               : np.min(error_lst),
        "mean"              : np.mean(error_lst),
        "median"            : np.median(error_lst),
        "std dev"           : np.std(error_lst),
        "x_max"             : np.argmax(error_lst),
        "x_min"             : np.argmin(error_lst),
        'half'              : half,
        'mae'               : sum(abs(error_lst - np.mean(error_lst)))/ len(error_lst)
    }
    indices = np.where(char_lst[1:] == char_lst[:-1] - 1)[0]
    if len(indices) > 0:
        pt = indices[0] + 1
    else:
        pt = "N/A" # couldn't find any

    data["least quad nonres"] = pt

    del error_lst

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
    primes = get_primes(500, 500000, 750000)
    add_primes(primes,multithreading=True, half=False, num_processes=5)
