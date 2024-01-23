import sys
sys.path.append('../')
from typing import List, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import time

from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import S, is_prime, sample_primes, Fourier_Expansion
import numpy as np

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")

class VaryingH(Run):
    path = f"{DATA_PATH}/varying_H/"
    name = lambda self, **kwargs: f"p = {int(kwargs['p'])} four exp {int(kwargs['low'])}-{int(kwargs['high'])}"
    required = ['p', 'low', 'high']
    
    def main(self, p: int, low, high) -> list:
        return Fourier_Expansion(p, start=low, H=high)

    def validate_inputs(self, p: int, low: int, high):
        if not is_prime(p):
            print(f"p must be prime")
            return False 
       
        
        return True 

if __name__ == "__main__":
    primes = [1986169] 
    for p in primes:
        lows = [(k) * np.floor(np.log(p)) + 1 for k in range(1 * int(np.ceil(np.sqrt(p)/np.log(p))))]
        highs = [k * np.floor(np.log(p)) for k in range(1, 1 * int(np.ceil(np.sqrt(p)/np.log(p)) + 1))]
        config = Config(
                run_instance    = VaryingH(),
                inputs          = [{"p" : p, "low" : low, 'high' : high} for low, high in zip(lows, highs)],
                time_experiment = True
                )
        experiment = Experiment(config)
        experiment.run()
       