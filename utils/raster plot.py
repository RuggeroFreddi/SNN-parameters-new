import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent  # la cartella sopra utils
sys.path.append(str(ROOT))
from LSM.model import SimulationParams, Reservoir
from pathlib import Path
import sys

from functions.simulates import simulate_trace, simulate_statistic_features, simulate
from functions.cross_validations import cross_validation_rf, cross_validation_slp, cross_validation_lr


DATASET_PATH = "dati/trajectory_spike_encoded.npz"
CV_NUM_SPLITS = 10

NUM_NEURONS = 2000
MEMBRANE_THRESHOLD = 2
REFRACTORY_PERIOD = 2
NUM_OUTPUT_NEURONS = 2000 - 32*32
LEAK_COEFFICIENT = 0.001
CURRENT_AMPLITUDE = MEMBRANE_THRESHOLD
PRESYNAPTIC_DEGREE = 0.20
SMALL_WORLD_GRAPH_P = 0.2
RELOAD = True
STATISTIC_SET = 1
TRACE_TAU = 30
MEMBRANE_RESET = False

def load_dataset(filename: str):
    npz_data = np.load(filename)
    return npz_data["data"], npz_data["labels"]

def main():
    os.makedirs("dati", exist_ok=True)

    data, labels = load_dataset(DATASET_PATH)
    print(f"Loaded data: {data.shape}, labels: {labels.shape}")

    weight = 0.00441576521429
    small_world_graph_k = int(PRESYNAPTIC_DEGREE * NUM_NEURONS * 2)

    sim_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=weight,
        weight_variance= 15,
        leak_cv = 10,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        mean_distance = 25 * weight , 
        small_world_graph_p=SMALL_WORLD_GRAPH_P,
        small_world_graph_k=small_world_graph_k,
        input_spike_times=np.zeros(
            (data.shape[1], data.shape[2]),
            dtype=np.uint8,
        ),
    )
    
    i=1
    sample = data[i]
    print("label: ", labels[i])
    LSM = Reservoir(sim_params)
    LSM.set_input_spike_times(sample)
    trace_tau = 20

    # simulazione
    spike_matrix_output = LSM.simulate(trace_tau=trace_tau, reset_trace=True)
    spike_matrix_output = np.asarray(spike_matrix_output)  # shape attesa: (T, N)

    print("spike_matrix_output shape:", spike_matrix_output.shape)

    # numero di spike per neurone (somma sui time-step)
    # asse 0 = tempo, asse 1 = neurone
    spike_counts = spike_matrix_output.sum(axis=0)  # shape: (num_neurons,)

    min_spikes = spike_counts.min()
    max_spikes = spike_counts.max()
    mean_spikes = spike_counts.mean()

    num_silent_neurons = (spike_counts == 0).sum()
    perc_silent = 100.0 * num_silent_neurons / spike_counts.shape[0]

    print(f"Spike per neurone: min = {min_spikes}, max = {max_spikes}, mean = {mean_spikes:.2f}")
    print(f"Neuroni con zero spike: {num_silent_neurons} ({perc_silent:.2f}%)")

    # raster plot: x = tempo, y = neuroni
    time_idx, neuron_idx = np.where(spike_matrix_output > 0)
    
    W = LSM.synaptic_weights      # csr_matrix

    # valori NON zero
    nz = W.data                   # array 1D con solo i pesi diversi da zero

    min_w = nz.min()
    max_w = nz.max()
    mean_w = nz.mean()

    print(f"peso minimo (≠0): {min_w}")
    print(f"peso massimo (≠0): {max_w}")
    print(f"peso medio  (≠0): {mean_w}")

    is_first = True
    trace_dataset, _, avg_spike = simulate(
                data=data,
                labels=labels,
                parameters=sim_params,
                trace_tau=TRACE_TAU,
                statistic_set=1,
                reload=True,
                is_first = is_first,
                membrane_reset= MEMBRANE_RESET,
            )


    mean_accuracy_rf, std_accuracy_rf,_,_,_,_ = cross_validation_rf(trace_dataset, CV_NUM_SPLITS)
    print("Mean accuracy: ", mean_accuracy_rf, "std accuracy: ", std_accuracy_rf, "avg spike: ", avg_spike)

    mean_accuracy_slp, std_accuracy_slp,_,_,_,_ = cross_validation_slp(trace_dataset, CV_NUM_SPLITS)
    print("Mean accuracy: ", mean_accuracy_slp, "std accuracy: ", std_accuracy_slp, "avg spike: ", avg_spike)
    LSM.synaptic_weights

    plt.figure(figsize=(10, 6))
    plt.scatter(time_idx, neuron_idx, s=1)
    plt.xlabel("time step")
    plt.ylabel("neuron index")
    plt.title("Raster plot")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
