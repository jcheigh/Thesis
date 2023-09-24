### organize files 

import os 
import re 

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")
PLOT_PATH = os.path.join(MAIN_PATH, "plots")

def rename_plots(path: str) -> None:
    # Regular expressions to match the two possible names and extract the prime
    ## FILL OUT!
    patterns = [
        re.compile(r'Prime = (?P<prime>\d+) Diff List\.txt', re.IGNORECASE)
    ]

    # Loop through all files in the directory
    for filename in os.listdir(path):
        # Only consider .png files
        if filename.endswith('.txt'):
            for pattern in patterns:
                match = pattern.match(filename)
                if match:
                    prime = match.group("prime")
                    new_name = f"p = {prime} error list.txt"
                    os.rename(os.path.join(path, filename), os.path.join(path, new_name))
                    break  # Break out of the pattern loop once a match is found

    return f"Renaming process in {path} completed."

# Assuming a sample path variable (you can replace this with your actual path)
path = os.path.join(DATA_PATH, 'error lists')
rename_plots(path)