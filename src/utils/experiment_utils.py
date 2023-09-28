# Modifying the Config class to accept an instance of Run (or its subclass)
from typing import List, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool
import os
import time

MAIN_PATH = os.path.join("/Users", "jcheigh", "Thesis")
DATA_PATH = os.path.join(MAIN_PATH, "data")
PLOT_PATH = os.path.join(MAIN_PATH, "plots")
RES_PATH  = os.path.join(MAIN_PATH, "results")

class Run(ABC):
    """
    Abstract base class representing a specific experimental run.

    Subclasses should define the following properties:
        - path (str): The directory path where the results will be saved.
        - name (Callable[..., str]): A function that computes the filename based on the experiment's inputs.
        - required (List[str]): A list of required input variable names for the experiment.

    Subclasses should also implement the main() method, which contains the main logic of the experiment.
    """
    def __init__(self):
        # Ensure that subclasses define the necessary properties
        if not hasattr(self, "path"):
            raise NotImplementedError("Subclasses must define a 'path' property.")
        
        if not hasattr(self, "name"):
            raise NotImplementedError("Subclasses must define a 'name' property.")
        
        if not hasattr(self, "required"):
            raise NotImplementedError("Subclasses must define a 'required' property.")

    @abstractmethod
    def main(self, **kwargs) -> List[Any]:
        """The main logic of the experiment. Must be implemented by all subclasses."""
        raise NotImplementedError

    def write_output(self, **kwargs):
        # Ensure the directory exists
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
        # Get the output from main
        output = self.main(**kwargs)
        
        # Compute the filename using the lambda function
        filename = self.name(**kwargs)
        full_path = os.path.join(self.path, f"{filename}.txt")
        
        # Write the output to the file
        with open(full_path, "w") as f:
            for item in output:
                f.write(str(item) + "\n")

    def validate_inputs(self, **kwargs) -> bool:
        """Optional method to validate the inputs. Returns True by default."""
        return True

    def execute(self, **kwargs):
        # Ensure all required inputs are present
        for var_name in self.required:
            if var_name not in kwargs:
                raise ValueError(f"Missing required input: {var_name}")
        
        # Validate inputs if the subclass provides a validation method
        if not self.validate_inputs(**kwargs):
            raise ValueError("Inputs validation failed.")
        
        self.write_output(**kwargs)

class Config:
    """
    Configuration for each Experiment
        run_instance is a Run object that can run a given experiment 
        inputs is a list of input to call the experiment on 
        time_experiment is whether or not to call
    """
    def __init__(self, run_instance: Run, inputs: List[Dict[str, Any]], time_experiment: bool = False):
        self._validate(run_instance, inputs, time_experiment)
        self.run_instance = run_instance
        self.inputs = inputs
        self.time_experiment = time_experiment
        
    def _validate(self, run_instance: Run, inputs: List[Dict[str, Any]], time_experiment: bool):
        if not isinstance(run_instance, Run):
            raise ValueError("run_instance must be an instance of a subclass of Run.")
        if not isinstance(inputs, list) or not all(isinstance(i, dict) for i in inputs):
            raise ValueError("inputs must be a list of dictionaries.")
        if not isinstance(time_experiment, bool):
            raise ValueError("time_experiment must be a boolean value.")

class Experiment:
    """
    Experiment runs an experiment to take a bunch of inputs and generate data from it 
    Input is a Config object (see above)

    run(self) calls .execute on each input in inputs

    Each of these calls a main function and writes the data to a .txt file, 
    though there's additional abstraction hidden. 
    """
    def __init__(self, config: Config):
        self.config = config
        
    def _run_single_experiment(self, input_kwargs):
        """Runs a single experiment based on input_kwargs."""
        print(f'Running experiment on {input_kwargs}')
        try:
            start_time = time.time() if self.config.time_experiment else None
                
            # Calling the execute method of the run_instance
            self.config.run_instance.execute(**input_kwargs)
                
            if self.config.time_experiment:
                elapsed_time = time.time() - start_time
                return f"Experiment completed in {elapsed_time:.2f} seconds for input {input_kwargs}."
                
        except Exception as e:
            return f"Error running experiment for input {input_kwargs}: {str(e)}"
        
    def run(self):
        # Use multiprocessing Pool to parallelize the experiments
        with Pool(processes=min(6, len(self.config.inputs))) as pool:
            results = pool.map(self._run_single_experiment, self.config.inputs)
            
        # Print results after all experiments are done
        for result in results:
            print(result)