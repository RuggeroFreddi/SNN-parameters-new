from snnpy.snn import SNN
import numpy as np
import pandas as pd


def simulate_trace(data, labels, parameters, trace_tau):
    # configuration constants
    START_INDEX = 700              # first neuron to keep
    EXCLUDED_TAIL_NEURONS = 200    # how many neurons to drop from the very end

    snn = SNN(parameters)
    initial_membrane_potentials = snn.get_membrane_potentials()

    end_index = parameters.num_neurons - EXCLUDED_TAIL_NEURONS
    kept_indices = list(range(START_INDEX, end_index))
    rows = []
    
    avg_spike_count = 0
    for i in range(len(data)):
        if i % 100 == 0:
            print(f"processed {i} of {len(data)} samples")

        sample = data[i]
        label = labels[i]

        snn.set_input_spike_times(sample)
        snn.set_membrane_potentials(initial_membrane_potentials)
        snn.simulate(trace_tau=trace_tau, reset_trace=True)
        avg_spike_count += snn.tot_spikes
        trace = np.asarray(snn.get_trace()).reshape(-1)

        # select only the neurons we want to export
        selected_trace_values = trace[kept_indices].tolist()

        row = selected_trace_values + [label]
        rows.append(row)

    # column names must match kept indices
    trace_columns = [f"neuron_{idx}_trace" for idx in kept_indices]
    columns = trace_columns + ["label"]

    df = pd.DataFrame(rows, columns=columns)
    return df, avg_spike_count / len(data)

def simulate_statistic_features(data, labels, parameters, statistic_set=1):
    """Simulate SNN and build a feature table (4 features per neuron + label)."""
    snn = SNN(parameters)
    initial_membrane_potentials = snn.get_membrane_potentials()

    rows = []
    avg_spike_count = 0
    num_output_neurons = None  

    for i in range(len(data)):
        if i % 100 == 0:
            print(f"processed {i} of {len(data)} samples")

        sample = data[i]
        label = labels[i]

        snn.set_input_spike_times(sample)
        snn.set_membrane_potentials(initial_membrane_potentials)
        snn.simulate(trace_tau=10, reset_trace=True)

        avg_spike_count += snn.tot_spikes
        
        if statistic_set ==1:
            spike_counts = snn.get_spike_counts()
            spike_variances = snn.get_spike_variances()
            first_spike_times = snn.get_first_spike_times()
            mean_spike_times = snn.get_mean_spike_times()

            if num_output_neurons is None:
                num_output_neurons = len(spike_counts)

            sample_features = np.stack(
                [
                    spike_counts,
                    spike_variances,
                    first_spike_times,
                    mean_spike_times,
                ],
                axis=1,
            )  
        elif statistic_set == 2:
            mean_spike_times = snn.get_mean_spike_times()
            first_spike_times = snn.get_first_spike_times()
            last_spike_times = snn.get_last_spike_times()
            mean_isi_per_neuron = snn.get_mean_isi_per_neuron()
            isi_variance_per_neuron = snn.get_isi_variance_per_neuron()

            if num_output_neurons is None:
                num_output_neurons = len(mean_spike_times)

            sample_features = np.stack(
                [
                    mean_spike_times,
                    first_spike_times,
                    last_spike_times,
                    mean_isi_per_neuron,
                    isi_variance_per_neuron,
                ],
                axis=1,
            )  

        row = sample_features.flatten().tolist() + [label]
        rows.append(row)
    if statistic_set == 1:
        metrics = ["spike_count", "spike_variance", "first_spike_time", "mean_spike_time"]
        column_names = [
            f"neuron_{i}_{metric}"
            for i in range(num_output_neurons)
            for metric in metrics
        ]
    elif statistic_set == 2:
        metrics = ["mean_spike_times", "first_spike_times", "last_spike_times", "mean_isi_per_neuron", "isi_variance_per_neuron"]
        column_names = [
            f"neuron_{i}_{metric}"
            for i in range(num_output_neurons)
            for metric in metrics
        ]
    column_names.append("label")

    df = pd.DataFrame(rows, columns=column_names)
    return df, avg_spike_count / len(data)
