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
from sage.all import pi, floor, sin, ln

from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import legendre, is_prime, sample_primes

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")

class Symmetry(Run):
    path = f"{DATA_PATH}/symmetry/"
    name = lambda self, **kwargs: f"symmetry(p={kwargs['p']},d={kwargs['d']})"
    required = ['p', 'd']

    def main(self, p: int, d: int, sin_approx: bool = False, H: int = None):
        if H is None:
            H = floor(ln(p)**2)

        if not sin_approx:
            cons = (2 * np.sqrt(p)/ pi).n()
            main = [(
                    legendre(n, p),
                    (-1)**n,
                    sin(pi*n/p).n(),
                    sin(2*pi*n*d/p).n(),
                    1/n
                    ) 
                    for n in range(1, H + 1)]
            return [cons] + main
        else:
            cons = (2 / np.sqrt(p)).n()
            main = [(legendre(n,p), (-1)**n, sin(2*pi*n*d/p).n()) for n in range(1, H+1)]
            return [cons] + main    

    def validate_inputs(self, p: int, d: int, sin_approx: bool = False, H: int = None):
        if H is None:
            H = int(floor(ln(p)**2))

        if not is_prime(p):
            print(f"p must be prime")
            return False 

        if d not in list(range(1, (p-1)/2 + 1)):
            print("d must be from 1 to #p")
            return False 
        
        if not isinstance(sin_approx, bool):
            print("sin_approx must be a boolean")
            return False 
        
        if not isinstance(H, int):
            print(f"H must be an integer")
            return False 
        
        return True 


if __name__ == "__main__":
    primes = sample_primes(10, 500000, 750000)
    config = Config(
            run_instance    = Symmetry(),
            inputs          = [{"p" : p, "d" : d} for p in primes for d in [5, int(p/100), int(p/10), int(p/5), (p-1)/2]],
            time_experiment = True
            )
    experiment = Experiment(config)
    experiment.run()

