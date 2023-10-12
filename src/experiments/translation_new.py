import sys
sys.path.append('../')
from typing import List, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import random
import time
import re 
from tqdm import tqdm

import numpy as np

from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import Fourier_Expansion, is_prime, sample_primes, S

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")
PLOT_PATH = os.path.join(MAIN_PATH, "plots")

class Translation(Run):
    path = f"{DATA_PATH}/translation"
    name = lambda self, **kwargs: f"translation means"
    required = []

    def main(self, n: int=750, p_min: int=10000, p_max: int=25000):
        primes = sample_primes(n, p_min, p_max, mod=3)
        result = []
        for prime in tqdm(primes):
            error = S(prime) - Fourier_Expansion(prime)
            result.append(np.mean(error))

        return result

    def validate_inputs(self, n: int=500, p_min: int=100000, p_max: int=200000): 
        return True 

if __name__ == "__main__":
    config = Config(
            run_instance    = Translation(),
            inputs          = [{}],
            time_experiment = True
            )
    experiment = Experiment(config)
    experiment.run(multithread=True)
