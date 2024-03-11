import sys
sys.path.append('../')
from typing import List, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import time
import numpy as np
from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import is_prime, sample_primes
from utils.r_utils import compute_R

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")

class R(Run):
    path = f"{DATA_PATH}/r lists/"
    name = lambda self, **kwargs: f"k = {kwargs['k']}, H = {kwargs['H']} R(x)"
    required = ['H', 'k']
    
    def main(self, H: int, k: int):
        s = np.arange(k)
        x = 2*np.pi * s/k
        r_vals = compute_R(x, H)
        return r_vals

    def validate_inputs(self, H, k):
        return True 

if __name__ == "__main__":
    primes = sample_primes(6, 1000000, 2000000)
    H_vals = [1, 5, 20, 100]
    config = Config(
            run_instance    = R(),
            inputs          = [{"H" : H, 'k' : k} for k in primes for H in H_vals],
            time_experiment = True
            )
    experiment = Experiment(config)
    experiment.run()
