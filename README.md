# Thesis

**By**: [Justin Cheigh](https://www.linkedin.com/in/justin-cheigh/) <br>
**Advisor**: [Leo Goldmakher](https://web.williams.edu/Mathematics/lg5)  

---

### Project Description
This repository contains computational work surrounding my undergraduate thesis in analytic number theory.
Specifically, this repo allows me to run computational experiments. In general, I often am working with some
erratic function. This erratic function has some input domain, and it may be conditioned on certain parameters.
Ideally I would be able to generate the output of this function, save it, and then analyze that data. This process 
is steamlined in this repo. Here's the general pipeline:

(1) Experiment:
    - An "experiment" allows you to specify multiple input parameters and a main function and generate the outputs for
    that function for each input. 
    - The source code for the Experiment, Run, and Config classes are in src/utils/experiments_utils.py
    - One should create new experiments in src/experiments (see character_sum.py as an example)
    - The default is to use python multiprocessing to spread across multiple cores 
    - There's auto saving and auto input valiation if you create the subclass correctly 

(2) Analyis:
    - Given this generated experiment data we need to analyze the data. There's a starter notebook src/analysis/analysis.ipynb
    that makes a lot of this easy. It provides functionality to easily fetch_data(experiment) and has lot of plotting functionality. Additionally, there is a Saver class that abstracts away saving new data. 

(3) Results:
    - Once you do this analysis of your data you may want to create a nice looking report to showcase the results. There's a basic starter notebook, but most of this is dependent on the actual analysis.

---
### Repo Layout

├── README.md                              <- Project description/relevant links
├── src/                                   <- Main Codebase
├── plots/                                 <- Saved Plots
├── data/                                  <- Saved Data
