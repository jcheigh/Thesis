import sys
sys.path.append('../')
from typing import List, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import time
import numpy as np
import math
from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import is_prime, sample_primes
from utils.j_utils import J

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")

class JRun(Run):
    path = f"{DATA_PATH}/j lists/"
    name = lambda self, **kwargs: f"k = {kwargs['k']}, H = {kwargs['H']} J list"
    required = ['H', 'k']
    
    def main(self, H, k: int):
        return J(k, H)

    def validate_inputs(self, H, k):
        return True 

if __name__ == "__main__":
    k = sample_primes(1, 1000000, 1200000, mod=1)[0]
    H_vals = [math.floor(math.log(k) ** i) for i in range(1, 6)] + [k]
        
    config = Config(
            run_instance    = JRun(),
            inputs          = [{"H" : H, 'k' : k} for H in H_vals],
            time_experiment = True
            )
    experiment = Experiment(config)
    experiment.run()



