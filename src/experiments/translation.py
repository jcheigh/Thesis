import sys
sys.path.append("../")
import os
import random
import time
from tqdm import tqdm 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.experiment_utils import Run, Experiment, Config
from utils.math_utils import Fourier_Expansion, is_prime, sample_primes, S

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")
PLOT_PATH = os.path.join(MAIN_PATH, "plots")

primes = sample_primes(500, 500, 2000, mod=3)
print('Finished sampling primes')
means = []
normalized_means = []
for prime in tqdm(primes):
    E_p = np.array(S(prime)) - np.array(Fourier_Expansion(prime))
    means.append(np.mean(E_p))
    normalized_means.append(np.mean(E_p) / prime)

TXT_PATH = os.path.join(DATA_PATH, "translation")
T_PLOT_PATH = os.path.join(PLOT_PATH, 'translation')

if not os.path.exists(TXT_PATH):
    os.mkdir(TXT_PATH)
if not os.path.exists(T_PLOT_PATH):
    os.mkdir(T_PLOT_PATH)

txt_name = f"{DATA_PATH}/translation/means.txt"

with open(txt_name, "w") as file:
    for elem in means:
        file.write(str(round(elem, 2)) + "\n")

plt.figure(figsize = (10,8))
p = sns.histplot(means, bins=50, kde=True, fill=True,
                edgecolor="black", linewidth=3
                )

p.axes.lines[0].set_color("orange")

plt.ylabel("Count", fontsize = 20)
plt.xlabel("Means", fontsize = 20)
plt.title("Translation Check", fontsize = 25)
plt.show()


plot_path = f"{PLOT_PATH}/translation/means.jpg"
plt.savefig(plot_path)
plt.close()       

plt.figure(figsize = (10,8))
p = sns.histplot(normalized_means, bins=50, kde=True, fill=True,
                edgecolor="black", linewidth=3
                )

p.axes.lines[0].set_color("orange")

plt.ylabel("Count", fontsize = 20)
plt.xlabel("Normalized Means", fontsize = 20)
plt.title("Normalized Translation Check", fontsize = 25)
plt.show()


plot_path = f"{PLOT_PATH}/translation/normalized_means.jpg"
plt.savefig(plot_path)
plt.close()         
