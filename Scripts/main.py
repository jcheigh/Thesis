
from sage.all import * 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import timeit
from functools import partial
from tqdm import tqdm
import multiprocessing
import random
import os
from helper import add_data, sample_primes

def update_repo(prime_list):
    """
    When you come back just run this again 
    and stop it whenever
    """
    for prime in tqdm(prime_list):
        print(f"P = {prime}: Checking Data Repository...\n")
        

        path = f"/Users/jcheigh/Thesis/Plots/Prime = {prime} Difference Plot.png"
        
        if not os.path.exists(path):
            print(f"P = {prime}: Analyzing Data...\n")
            
            add_data(prime)
    
    print(f"Dataset Update Complete: N = {len(prime_list)} Primes Processed")


def process_primes(primes):
    for prime in tqdm(primes):
        print(f"P = {prime}: Checking Data Repository...\n")
    
        path = f"/Users/jcheigh/Thesis/Plots/Prime = {prime} Difference Plot.png"
        
        if not os.path.exists(path):
            print(f"P = {prime}: Analyzing Data...\n")
            
            add_data(prime)

def update_repo1(prime_list, num_processes = 4):
    """
    When you come back just run this again 
    and stop it whenever
    """
    chunk_size = len(prime_list) // num_processes
    number_chunks = [prime_list[i:i+chunk_size] for i in range(0, len(prime_list), chunk_size)]
    pool = multiprocessing.Pool(processes=num_processes)

    pool.map(process_primes, number_chunks)

if __name__ == "__main__":
    prime_list = sample_primes(10)
    update_repo1(prime_list, num_processes = 8)
