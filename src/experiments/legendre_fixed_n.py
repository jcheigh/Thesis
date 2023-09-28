import sys
sys.path.append('../')
from typing import List, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import time
import random

from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import get_prime_range, legendre

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")

class LegendreFixedN(Run):
    path = f"{DATA_PATH}/legendre fixed n/"
    name = lambda self, **kwargs: f"n = {kwargs['n']} legendre symbol list"
    required = ['n']

    def main(self, n: int, p_min: int = 100000, p_max: int = 200000) -> list:
        primes = get_prime_range(p_min, p_max)
        return [legendre(n, prime) for prime in primes]

    def validate_inputs(self, n: int, p_min: int = 100000, p_max: int = 200000):
        if not isinstance(n, int):
            print(f"n must be an integer")
            return False 
        
        if not isinstance(p_min, int):
            print(f"p_min must be an integer")

        if not isinstance(p_max, int):
            print(f"p_max must be an integer")

        if n > p_min: 
            print(f"n cannot be larger than the primes")
            return False 

        if p_min > p_max:
            print(f"p_min must be <= p_max")
            return False 

        return True 
     
if __name__ == "__main__":
    vals = random.choices(list(range(75000, 100000)), k=50)
    config = Config(
        run_instance    = LegendreFixedN(),
        inputs          = [{"n" : n} for n in vals],
        time_experiment = True
        )
    experiment = Experiment(config)
    experiment.run()
