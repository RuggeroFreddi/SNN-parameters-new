import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent  # la cartella sopra utils
sys.path.append(str(ROOT))
from LSM.model import SimulationParams
from functions.simulates import simulate_trace, simulate_statistic_features, simulate
from functions.cross_validations import cross_validation_rf, cross_validation_slp, cross_validation_lr


DATASET_PATH = "dati/mnist_rate_encoded.npz" #"dati/trajectory_spike_encoded.npz"
DATASET_PATH = "dati/trajectory_spike_encoded.npz"

CV_NUM_SPLITS = 10

NUM_NEURONS = 2000
MEMBRANE_THRESHOLD = 2
REFRACTORY_PERIOD = 2
NUM_OUTPUT_NEURONS = 100
LEAK_COEFFICIENT = 0.004
CURRENT_AMPLITUDE = MEMBRANE_THRESHOLD
PRESYNAPTIC_DEGREE = 0.20
SMALL_WORLD_GRAPH_P = 0.2

TRACE_TAU = 50
NUM_WEIGHT_STEPS = 21  # how many weights to test
MEMBRANE_RESET = False
RELOAD = True
WEIGHT_VARIANCE = 15


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
        weight_variance= WEIGHT_VARIANCE,
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
        critical_weight * 1.3,
        #critical_weight * 1,
        NUM_WEIGHT_STEPS,
    )

    results = []  # list of (weight, accuracy)
    cnt = 0
    is_first = True
    for weight in weight_values:
        cnt+= 1
        print(f"\n--- Testing weight = {weight:.6f} ---test {cnt}/{NUM_WEIGHT_STEPS}")
        sim_params.mean_weight = weight
        sim_params.weight_variance = WEIGHT_VARIANCE 

        trace_dataset, _, avg_spike = simulate(
            data=data,
            labels=labels,
            parameters=sim_params,
            trace_tau=TRACE_TAU,
            statistic_set=2,
            reload=RELOAD,
            is_first = is_first,
            membrane_reset= MEMBRANE_RESET,
        )
        is_first = False
        """trace_dataset, avg_spike = simulate_statistic_features(
            data=data,
            labels=labels,
            parameters=sim_params,
            statistic_set=2,
            membrane_reset= MEMBRANE_RESET,
        )"""

        mean_accuracy_rf, std_accuracy_rf,_,_,_,_ = cross_validation_rf(trace_dataset, CV_NUM_SPLITS)
        print("Mean accuracy: ", mean_accuracy_rf, "std accuracy: ", std_accuracy_rf, "avg spike: ", avg_spike)

        mean_accuracy_slp, std_accuracy_slp,_,_,_,_ = cross_validation_slp(trace_dataset, CV_NUM_SPLITS)
        print("Mean accuracy: ", mean_accuracy_slp, "std accuracy: ", std_accuracy_slp, "avg spike: ", avg_spike)

        results.append((weight, mean_accuracy_rf, std_accuracy_rf, mean_accuracy_slp, std_accuracy_slp))
        
    weights_plot = []
    acc_rf = []
    std_rf = []
    acc_slp = []
    std_slp = []


    for (w,
         mean_accuracy_rf, std_accuracy_rf,
         mean_accuracy_slp, std_accuracy_slp) in results:
        weights_plot.append(w)
        acc_rf.append(mean_accuracy_rf)
        std_rf.append(std_accuracy_rf)
        acc_slp.append(mean_accuracy_slp)
        std_slp.append(std_accuracy_slp)


    plt.figure()

    # 1) Random Forest
    plt.plot(weights_plot, acc_rf, marker="o", label="RF accuracy")
    lower_rf = [m - s for m, s in zip(acc_rf, std_rf)]
    upper_rf = [m + s for m, s in zip(acc_rf, std_rf)]
    plt.fill_between(weights_plot, lower_rf, upper_rf, alpha=0.15)

    # 2) SLP (o perceptron, dipende da cosa è cross_validation_slp)
    plt.plot(weights_plot, acc_slp, marker="s", label="SLP accuracy")
    lower_slp = [m - s for m, s in zip(acc_slp, std_slp)]
    upper_slp = [m + s for m, s in zip(acc_slp, std_slp)]
    plt.fill_between(weights_plot, lower_slp, upper_slp, alpha=0.15)

    # linea del critical weight
    plt.axvline(x=critical_weight, linestyle="--", label="Critical weight")

    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean CV accuracy")
    plt.title("Accuracy vs. mean synaptic weight")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
