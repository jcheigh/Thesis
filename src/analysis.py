import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os 
from main_helper import get_prime_range

def create_dataset():
    """
    Creates pd.DataFrame based on Thesis repository.

    Input:
    -------
    None

    Output:
    -------
    pd.DataFrame: df with data stored in Thesis/Data

    Column Description:
        prime: prime that data collection was for 
        pos_error: max(S(x) - Fourier(x))
        neg_error: min(S(x) - Fourier(x))
        x_pos_error: List[x] s.t. S(x) - Fourier(x) == pos_error
        x_neg_error: List[x] s.t. S(x) - Fourier(x) == neg_error
        max_error: max(pos_error, -1 * neg_error)
        max_error_ind: x s.t. |S(x) - Fourier(x)| == max_error
        index: k iff p_k == prime
        mag_diff: |pos_error + neg_error| 
    """


    df = pd.DataFrame(columns=["prime", "pos_error", "neg_error", "x_pos_error", "x_neg_error"])
    
    folder_path = os.path.join("/Users", "")
    folder_path = "/Users/jcheigh/Thesis/Data/"
    
    file_list = [file for file in os.listdir(folder_path) if file.endswith(".txt")]
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, "r") as file:
            contents = file.read()
            data = eval(contents)
            df.loc[len(df.index)] = list(data.values()) # set data collected
            
    prime_list = get_prime_range(100000, 200000)       
    
    # engineered features 
    df['max_error'] = df.apply(lambda row: max(row['pos_error'], -1 * row['neg_error']), axis=1)
    df['max_error_ind'] = np.where(df['max_error'] == df['pos_error'], 1, 0)
    df["x_max_error"] = np.where(df["max_error_ind"] == 1, df["x_pos_error"], df["x_neg_error"])
    df['index'] = df['prime'].apply(lambda p: prime_list.index(p))
    df["mag_diff"] = df.apply(lambda row : abs(row["pos_error"] + row["neg_error"]), axis = 1)
    print(f"Of the {len(prime_list)} primes we sampled from, there are currently {len(df)} datapoints")

    return df.sort_values("index")

def plot_dist(df, num_bins, y = None):
    if y is None:
        samp = list(df["prime"])
        title = "Sample Distribution of Primes"
    
    if y == "all primes":
        samp = get_prime_range(100000, 200000)
        title = "Distribution of All Primes"
        
    bins = np.linspace(100000, 200000, num_bins)
    
    plt.hist(samp, bins = bins)
    
    plt.xlabel("Prime")
    plt.ylabel("Frequency")
    plt.title(title)
    
    plt.show()
    
def plot(df, x, y, x_lab, y_lab, title, x_trans = None):
    # change later 
    
    X = df[x]
    Y = df[y]
    
    if x_trans == 'log':
        Y = np.log(X)
        
    if x_trans == 'log squared':
        X = np.square(np.log(X))
        
    plt.plot(X, Y)
    
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)

    plt.show()       
    
def boxplot(df):
    # add group by feature 
    data = list(df["max_error_ind"])
    print(data)
    count_0 = data.count(0)
    count_1 = data.count(1)
    
    counts = [count_0, count_1]
    labels = ["Negative", "Positive"]
    plt.bar(labels, counts)

    plt.xlabel("Error Type")
    plt.ylabel('Count')
    plt.title('Bar Plot of Pos/Neg Error')
    plt.show()


if __name__ == "__main__":
    df = create_dataset()
    plot(df, "prime", "max_error", x_lab = "Prime", y_lab = "Maximum Error", title = "Max Error vs. Prime")#, x_trans = "log")