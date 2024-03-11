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
from utils.r_utils import compute_R_sum, compute_R_char_sum, compute_J1

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")

class J1(Run):
    ### differnet integral bounds
    path = f"{DATA_PATH}/j1 lists/"
    name = lambda self, **kwargs: f"k = {kwargs['k']}, H = {kwargs['H']} J1 list"
    required = ['H', 'k']
    
    def main(self, H: int, k: int):
        return compute_J1(H, k)

    def validate_inputs(self, H, k):
        return True 

if __name__ == "__main__":
    primes = sample_primes(6, 500, 750, mod=1)
    H = 5
    config = Config(
            run_instance    = J1(),
            inputs          = [{"H" : H, 'k' : k} for k in primes],
            time_experiment = True
            )
    experiment = Experiment(config)
    experiment.run()



