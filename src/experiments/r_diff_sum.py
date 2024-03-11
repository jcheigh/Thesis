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
from utils.r_utils import compute_R_sum, compute_R_char_sum, compute_R_diff_sum

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")

class RDiffSum(Run):
    path = f"{DATA_PATH}/r diff sum lists/"
    name = lambda self, **kwargs: f"k = {kwargs['k']}, H= {kwargs['H']} R diff sum"
    required = ['H', 'k']
    
    def main(self, H: int, k: int):
        return compute_R_diff_sum(k, H)

    def validate_inputs(self, H, k):
        return True 

if __name__ == "__main__":
    primes = sample_primes(2, 2000, 5000)
    H = 500
    config = Config(
            run_instance    = RDiffSum(),
            inputs          = [{"H" : H, 'k' : k} for k in primes],
            time_experiment = True
            )
    experiment = Experiment(config)
    experiment.run()



