import os
import numpy as np
import matplotlib.pyplot as plt

from snnpy.snn import SimulationParams

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent  # la cartella sopra utils
sys.path.append(str(ROOT))
from functions.simulates import simulate_trace, simulate_statistic_features
from functions.cross_validations import cross_validation_rf


DATASET_PATH = "dati/trajectory_spike_encoded.npz"
CV_NUM_SPLITS = 10

NUM_NEURONS = 2000
MEMBRANE_THRESHOLD = 2
REFRACTORY_PERIOD = 2
NUM_OUTPUT_NEURONS = 50
LEAK_COEFFICIENT = 0
CURRENT_AMPLITUDE = MEMBRANE_THRESHOLD
PRESYNAPTIC_DEGREE = 0.20
SMALL_WORLD_GRAPH_P = 0.2

TRACE_TAU = 60
NUM_WEIGHT_STEPS = 51  # how many weights to test


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
    # make sure the data directory exists
    os.makedirs("dati", exist_ok=True)

    # load rate-encoded dataset
    data, labels = load_dataset(DATASET_PATH)
    print(f"Loaded data: {data.shape}, labels: {labels.shape}")

    _, critical_weight = compute_critical_weight(data)

    small_world_graph_k = int(PRESYNAPTIC_DEGREE * NUM_NEURONS * 2)

    sim_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=critical_weight,
        weight_variance=critical_weight * 5,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_GRAPH_P,
        small_world_graph_k=small_world_graph_k,
        input_spike_times=np.zeros(
            (data.shape[1], data.shape[2]),
            dtype=np.uint8,
        ),
    )

    # choose the weights to test
    weight_values = np.linspace(
        critical_weight / 100,
        critical_weight * 1.4,
        NUM_WEIGHT_STEPS,
    )

    results = []  # list of (weight, accuracy)
    cnt = 0
    for weight in weight_values:
        cnt+= 1
        print(f"\n--- Testing weight = {weight:.6f} ---test {cnt}/{NUM_WEIGHT_STEPS}")
        sim_params.mean_weight = weight
        sim_params.weight_variance = weight * 5

        trace_dataset, _ = simulate_trace(
            data=data,
            labels=labels,
            parameters=sim_params,
            trace_tau=TRACE_TAU,
        )

        """trace_dataset, _ = simulate_statistic_features(
            data=data,
            labels=labels,
            parameters=sim_params,
            statistic_set=2,
        )"""
        
        mean_accuracy = cross_validation_rf(trace_dataset, CV_NUM_SPLITS)
        print("Mean accuracy:", mean_accuracy)

        results.append((weight, mean_accuracy))

    # plot results in a single figure
    weights_plot = [w for (w, _) in results]
    accuracies_plot = [acc for (_, acc) in results]

    plt.figure()
    plt.plot(weights_plot, accuracies_plot, marker="o", label="Accuracy")
    plt.axvline(x=critical_weight, color="red", linestyle="--", label="Critical weight")
    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean CV accuracy")
    plt.title("Accuracy vs. mean synaptic weight")
    plt.grid(True)
    plt.legend()  # <- questa
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
