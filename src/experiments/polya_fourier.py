import sys
sys.path.append('../')
from typing import List, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import random
import time
import re 

from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import Fourier_Expansion, is_prime, sample_primes

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")
PLOT_PATH = os.path.join(MAIN_PATH, "plots")
class PolyaFourier(Run):
    path = f"{DATA_PATH}/polya fourier lists/"
    name = lambda self, **kwargs: f"p = {kwargs['p']} fourier exp list"
    required = ['p']
    
    def main(self, p: int, x_min: int = 0, x_max: int = None, H: int = None) -> list:
        return Fourier_Expansion(p, x_min, x_max, H)
        

    def validate_inputs(self, p: int, x_min: int = 0, x_max: int = None, H: int = None):
        if not is_prime(p):
            print(f"p must be prime")
            return False 
        if x_max is not None:
            if not isinstance(x_min, int) or not isinstance(x_max, int):
                print(f"x_min and x_max must be ints")
                return False 
            if x_min > x_max:
                print(f"x_min must be <= x_max")
                return False 
        if H is not None:
            if not isinstance(H, int):
                print(f"H must be an integer")
                return False 
        
        return True 

def get_primes_from_filename(path):
    # Regular expression to match the required filename format and extract the prime
    pattern = re.compile(r'p = (\d+) character sum list.txt')
    
    primes = []
    for filename in os.listdir(path):
        match = pattern.match(filename)
        if match:
            prime = int(match.group(1))
            primes.append(prime)
    return primes

def filter_primes():
    char_sum_path = f"{DATA_PATH}/char sum lists/"
    error_plot_path = f"{PLOT_PATH}/error plots/"
    fourier_path   = f"{DATA_PATH}/polya fourier lists/"

    primes = get_primes_from_filename(char_sum_path)
    
    for prime in primes.copy():  # iterate over a copy of the list so we can modify the original list
        filename_to_check = f"p = {prime} error plot.jpg"
        if filename_to_check in os.listdir(error_plot_path):
            primes.remove(prime)
        filename_to_check = f"p = {prime} fourier exp list.txt"
        if filename_to_check in os.listdir(fourier_path):
            primes.remove(prime)
    return primes

def get_random_primes(n):
    filtered_primes = filter_primes()
    return random.sample(filtered_primes, n)

if __name__ == "__main__":
    primes = get_random_primes(8)
    config = Config(
            run_instance    = PolyaFourier(),
            inputs          = [{"p" : p} for p in primes],
            time_experiment = True
            )
    experiment = Experiment(config)
    experiment.run(multithread=False)
