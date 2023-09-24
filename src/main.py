from tqdm import tqdm
import multiprocessing
import os
from helper import add_data, sample_primes

def process_primes(primes: list):
    """
    Helper for main function (see below)

    For all p in primes,
        creates diff plot and writes to Plots folder
        gets prime data and writes to Data folder
    """
    for prime in tqdm(primes):
        print(f"Analyzing Prime = {prime}")
        add_data(prime)

def main(prime_list, multithread = True, num_processes = 6):
    """
    For all p in primes,
        creates diff plot and writes to Plots folder
        gets prime data and writes to Data folder

    If multithread then uses multiprocessing on num_processing cores

    Multiprocessing is faster by approximately factor of num_processes, but
    since the generated plots are so computationally expensive if you do num_processes
    around 6 you get (i.e. I got) memory problems after processing ~ 6 primes from
    100k to 200k

    COMMAND TO RUN: caffeinate -s sage main.py
        caffeinate -s ensures that the laptop doesn't sleep while running
        sage main.py runs this file- sage since we are using Sage Python
    """
    if multithread:
        chunk_size = len(prime_list) // num_processes
        number_chunks = [prime_list[i : i + chunk_size] for i in range(0, len(prime_list), chunk_size)]
        pool = multiprocessing.Pool(processes = num_processes)

        pool.map(process_primes, number_chunks)

    else:
        process_primes(prime_list)

if __name__ == "__main__":
    prime_list = [193, 197, 199]
    main(prime_list, multithread = False)
