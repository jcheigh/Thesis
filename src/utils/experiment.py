import os
import time
import logging
class Experiment:
    MAIN_PATH = os.path.join("Users", "jcheigh", "Thesis")
    EXPERIMENT_LOG = f"{MAIN_PATH}/experiments.txt"

    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
        self._update_experiment_log()
        self._handle_output_path()

    def _setup_logging(self):
        logging.basicConfig(filename='experiment.log', level=logging.INFO)
        logging.info(f"Initializing experiment: {self.config.name}")

    def _update_experiment_log(self):
        with open(Experiment.EXPERIMENT_LOG, 'a+') as f:
            f.seek(0)  # Start reading from the beginning of the file
            lines = f.readlines()
            if self.config.name not in lines:
                f.write(f"{self.config.name}\n")
                logging.info(f"Added experiment {self.config.name} to the log")
            else:
                logging.info(f"Experiment {self.config.name} already exists in the log")

    def _handle_output_path(self):
        if not os.path.exists(self.config.output_path):
            os.makedirs(self.config.output_path)
            logging.info(f"Created output path: {self.config.output_path}")

    def run(self):
        results = []
        for inp in self.config.inputs:
            if self.config.time_function:
                start_time = time.time()

            if self.config.kwargs:
                result = self.config.run_experiment(inp, **self.config.kwargs)
            else:
                result = self.config.run_experiment(inp)

            if self.config.time_function:
                elapsed_time = time.time() - start_time
                logging.info(f"Experiment {self.config.name} took {elapsed_time:.2f} seconds for input {inp}")

            results.append(result)

        if self.config.save_output:
            self._save_results(results)

        return results

    def _save_results(self, results):
        # Assuming we are saving as a text file for simplicity. 
        # This can be expanded to save in other formats like JSON, CSV, etc.
        result_path = os.path.join(self.config.output_path, f"{self.config.name}_results.txt")
        with open(result_path, 'w') as f:
            for result in results:
                f.write(str(result) + '\n')
        logging.info(f"Saved results to {result_path}")


def my_experiment_function(input_data, param1, param2):
    # Your experiment logic here
    return f"Result for {input_data} with {param1} and {param2}"

config = Config(
    run_experiment=my_experiment_function,
    inputs=[1, 2, 3],
    name="TestExperiment",
    output_path="./experiment_outputs",
    kwargs={"param1": "A", "param2": "B"},
    time_function=True,
    save_output=True
)

experiment = Experiment(config)
results = experiment.run()
print(results)
