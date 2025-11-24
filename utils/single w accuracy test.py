import os
import numpy as np

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent  # la cartella sopra utils
sys.path.append(str(ROOT))
from LSM.model import SimulationParams
from functions.simulates import simulate_trace, simulate_statistic_features
from functions.cross_validations import cross_validation_rf


DATASET_PATH = "dati/trajectory_spike_encoded.npz"
CV_NUM_SPLITS = 10

NUM_NEURONS = 2000
MEMBRANE_THRESHOLD = 2
REFRACTORY_PERIOD = 2
NUM_OUTPUT_NEURONS = 100
LEAK_COEFFICIENT = 0.001
CURRENT_AMPLITUDE = MEMBRANE_THRESHOLD
PRESYNAPTIC_DEGREE = 0.1
SMALL_WORLD_GRAPH_P = 0.2
MEMBRANE_RESET = False
RELOAD = True
WEIGHT_VARIANCE = 50
LEAK_CV = 1
MEAN_DISTANCE_CV = 5

TRACE_TAU = 50


def load_dataset(filename: str):
    """
    Return (data, labels) from a .npz file.
    """
    npz_data = np.load(filename)
    return npz_data["data"], npz_data["labels"]


def compute_critical_weight(inputs: np.ndarray):
    """
    Estimate the average input current and the critical synaptic weight.
    """
    # average current per synapse / tick
    avg_input_current = (
        np.sum(inputs)
        / (NUM_NEURONS * inputs.shape[0] * inputs.shape[2])
    )

    critical_weight = (
        MEMBRANE_THRESHOLD
        - 2 * avg_input_current * CURRENT_AMPLITUDE * (REFRACTORY_PERIOD + 1)
    ) / (PRESYNAPTIC_DEGREE * NUM_NEURONS)

    return avg_input_current, critical_weight

def main():
    os.makedirs("dati", exist_ok=True)

    data, labels = load_dataset(DATASET_PATH)
    print(f"Loaded data: {data.shape}, labels: {labels.shape}")

    _, critical_weight = compute_critical_weight(data)
    print(critical_weight)
    critical_weight *= 0.9 

    small_world_graph_k = int(PRESYNAPTIC_DEGREE * NUM_NEURONS * 2)

    sim_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=critical_weight,
        weight_variance= WEIGHT_VARIANCE,
        leak_cv = LEAK_CV,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_GRAPH_P,
        small_world_graph_k=small_world_graph_k,
        mean_distance = MEAN_DISTANCE_CV * critical_weight , 
        input_spike_times=np.zeros(
            (data.shape[1], data.shape[2]),
            dtype=np.uint8,
        ),
    )
    

    trace_df, avg_spike_count = simulate_statistic_features(
        data=data,
        labels=labels,
        parameters=sim_params,
        statistic_set=2,
    )

    print("avg spike count:", avg_spike_count)

    mean_accuracy_rf, std_accuracy_rf,_,_,_,_  = cross_validation_rf(trace_df, CV_NUM_SPLITS)
    print("mean accuracy:", mean_accuracy_rf, "std accuracy", std_accuracy_rf)

if __name__ == "__main__":
    main()
