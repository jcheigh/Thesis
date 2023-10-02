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
from sage.all import cos, ln, floor, pi

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")
PLOT_PATH = os.path.join(MAIN_PATH, "plots")

def test_sum(p, n):
    return sum([cos(2*pi*n*d/p) for d in range(1, (p-1)//2+1)]).n()

def test(p):
    for n in range(1, floor(ln(p)**2) + 1):
        print('=' * 15)
        result = test_sum(p, n)s
        print(f"n = {n}, result = {result}")

if __name__ == "__main__":
    primes = [104729]
    for prime in primes:
        if prime % 4 == 1:
            print(f"Testing Prime = {prime}")ç
            test(prime)
