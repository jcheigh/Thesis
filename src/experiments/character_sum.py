import sys
sys.path.append('../')
from typing import List, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import time

from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import S, is_prime, sample_primes

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")

class CharacterSum(Run):
    path = f"{DATA_PATH}/char sum lists/"
    name = lambda self, **kwargs: f"p = {kwargs['p']} character sum list"
    required = ['p']
    
    def main(self, p: int, x_min: int = 0, x_max: int = None) -> list:
        return S(p, x_min, x_max)

    def validate_inputs(self, p: int, x_min: int = 0, x_max: int = None):
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
        
        return True 

if __name__ == "__main__":
    primes = sample_primes(5, 10000000, 50000000)
    config = Config(
            run_instance    = CharacterSum(),
            inputs          = [{"p" : p} for p in primes],
            time_experiment = True
            )
    experiment = Experiment(config)
    experiment.run()
