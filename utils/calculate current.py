import os
from datetime import date
import numpy as np


TASK = "MNIST" # possible values: "MNIST"
OUTPUT_FEATURES = "statistics" # possible values: "statistics", "trace"

if TASK=="MNIST": 
    DATASET_PATH = "dati/mnist_rate_encoded.npz"
else:
    print("selected unknown task.")
    exit()


TRACE_TAU = 60
NUM_WEIGHT_STEPS = 51  # how many mean_weight values to test

PARAM_NAME = "membrane_threshold" # possible value: "beta", "membrane_threshold", "current_amplitude"


PARAMETER_VALUES = [0.5, 1, 2] # use it when PARM_NAME = "current_amplitude"

today_str = date.today().strftime("%Y_%m_%d")
RESULTS_DIR = f"results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{today_str}"
CSV_NAME = os.path.join(
    RESULTS_DIR,
    f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
)


def load_dataset(filename: str):
    """
    Return (data, labels) from a .npz file.
    """
    npz_data = np.load(filename)
    return npz_data["X"], npz_data["y"]

def compute_mean_I(inputs: np.ndarray):
    """
    Estimate the average input current injected into the network.
    """
    avg_input_current = (
        np.sum(inputs)
        / (inputs.shape[0] * inputs.shape[2])
    )

    return avg_input_current

data, labels = load_dataset(DATASET_PATH)
print(f"Loaded data: {data.shape}, labels: {labels.shape}")
mean_I = compute_mean_I(data)
print(mean_I)
