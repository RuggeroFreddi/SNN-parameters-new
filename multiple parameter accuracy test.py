"""nota per me: la formula per i MEMBRANE_THRESHOLD 
MEMBRANE_THRESHOLD = (MEMBRANE_THRESHOLD - 2 * I * REFRACTORY_PERIOD) * (beta_rir / beta_new) + 2 * I * REFRACTORY_PERIOD
esempio: se voglio sapere quale valore di MEMBRANE_THRESHOLD usare affinchè il punto critico sia uguale a quello che avevo usando
MEMBRANE_THRESHOLD_rif e beta_new, nel momento in cui beta è beta_rif, allora uso la formual qui sopra"""

import os
import yaml
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from snnpy.snn import SimulationParams
from functions.simulates import simulate_trace, simulate_statistic_features
from functions.cross_validations import cross_validation_rf, cross_validation_slp

# the following global variables will be loaded using function load_config
TASK = None
OUTPUT_FEATURES = None
CV_NUM_SPLITS = None
ACCURACY_THRESHOLD = None
NUM_NEURONS = None
MEMBRANE_THRESHOLD = None
REFRACTORY_PERIOD = None
NUM_OUTPUT_NEURONS = None
LEAK_COEFFICIENT = None
CURRENT_AMPLITUDE = None
PRESYNAPTIC_DEGREE = None
SMALL_WORLD_GRAPH_P = None
SMALL_WORLD_GRAPH_K = None
TRACE_TAU = None
NUM_WEIGHT_STEPS = None
PARAM_NAME = None
PARAMETER_VALUES = None
DATASET_PATH = None
RESULTS_DIR = None
CSV_NAME = None
STATISTIC_SET = None
MEMBRANE_RESET = None

def load_config(path: str = "settings/config.yaml"):
    global TASK, OUTPUT_FEATURES, CV_NUM_SPLITS, ACCURACY_THRESHOLD
    global NUM_NEURONS, MEMBRANE_THRESHOLD, REFRACTORY_PERIOD, NUM_OUTPUT_NEURONS
    global LEAK_COEFFICIENT, CURRENT_AMPLITUDE, PRESYNAPTIC_DEGREE
    global SMALL_WORLD_GRAPH_P, SMALL_WORLD_GRAPH_K, TRACE_TAU
    global NUM_WEIGHT_STEPS, PARAM_NAME, PARAMETER_VALUES
    global DATASET_PATH, CSV_NAME, RESULTS_DIR, STATISTIC_SET
    global MEMBRANE_RESET

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config non trovato: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # calcolo derivato
    presyn = cfg["PRESYNAPTIC_DEGREE"]
    num_neurons = cfg["NUM_NEURONS"]
    cfg["SMALL_WORLD_GRAPH_K"] = int(presyn * num_neurons * 2)

    # assegno alle globali
    TASK = cfg["TASK"]
    OUTPUT_FEATURES = cfg["OUTPUT_FEATURES"]
    CV_NUM_SPLITS = cfg["CV_NUM_SPLITS"]
    ACCURACY_THRESHOLD = cfg["ACCURACY_THRESHOLD"]
    NUM_NEURONS = cfg["NUM_NEURONS"]
    MEMBRANE_THRESHOLD = cfg["MEMBRANE_THRESHOLD"]
    REFRACTORY_PERIOD = cfg["REFRACTORY_PERIOD"]
    NUM_OUTPUT_NEURONS = cfg["NUM_OUTPUT_NEURONS"]
    LEAK_COEFFICIENT = cfg["LEAK_COEFFICIENT"]
    CURRENT_AMPLITUDE = cfg["CURRENT_AMPLITUDE"]
    PRESYNAPTIC_DEGREE = cfg["PRESYNAPTIC_DEGREE"]
    SMALL_WORLD_GRAPH_P = cfg["SMALL_WORLD_GRAPH_P"]
    SMALL_WORLD_GRAPH_K = cfg["SMALL_WORLD_GRAPH_K"]
    TRACE_TAU = cfg["TRACE_TAU"]
    NUM_WEIGHT_STEPS = cfg["NUM_WEIGHT_STEPS"]
    PARAM_NAME = cfg["PARAM_NAME"]
    PARAMETER_VALUES = cfg["PARAMETER_VALUES"]
    MEMBRANE_RESET = cfg ["MEMBRANE_RESET"]
    
    if TASK=="MNIST": 
        DATASET_PATH = "dati/mnist_rate_encoded.npz"
        STATISTIC_SET = 1
    elif TASK  == "TRAJECTORY":
        DATASET_PATH = "dati/trajectory_spike_encoded.npz"
        STATISTIC_SET = 2
    else:
        print("selected unknown task.")
        exit()

    today_str = date.today().strftime("%Y_%m_%d")
    RESULTS_DIR = f"results/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{today_str}"
    CSV_NAME = os.path.join(
        RESULTS_DIR,
        f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
    )

    return cfg

def load_dataset(filename: str):
    """
    Return (data, labels) from a .npz file.
    """
    npz_data = np.load(filename)
    return npz_data["data"], npz_data["labels"]

def _to_builtin(obj):
    # converte ricorsivamente numpy/pandas -> tipi Python
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return { _to_builtin(k): _to_builtin(v) for k, v in obj.items() }
    if isinstance(obj, (list, tuple, set)):
        return [ _to_builtin(x) for x in obj ]
    return obj  # stringhe, bool, None, ecc.

def save_experiment_metadata(results_dir: str,
                             parameter_name: str,
                             parameter_values: list[float],
                             weight_segments_rf: dict[float, dict[str, float]],
                             weight_segments_slp: dict[float, dict[str, float]],
                             mean_I):
    metadata = {
        "experiment" :{
            "task": TASK,
            "output_features": OUTPUT_FEATURES,
            "mean_I": mean_I,
        },
        "global_parameters": {
            "num_neurons": NUM_NEURONS,
            "membrane_threshold": MEMBRANE_THRESHOLD,
            "refractory_period": REFRACTORY_PERIOD,
            "num_output_neurons": NUM_OUTPUT_NEURONS,
            "leak_coefficient": LEAK_COEFFICIENT,
            "current_amplitude": CURRENT_AMPLITUDE,
            "presynaptic_degree": PRESYNAPTIC_DEGREE,
            "small_world_graph_p": SMALL_WORLD_GRAPH_P,
            "small_world_graph_k": SMALL_WORLD_GRAPH_K,
            "trace_tau": TRACE_TAU,
            "num_weight_steps": NUM_WEIGHT_STEPS,
            "cv_num_splits": CV_NUM_SPLITS,
            "accuracy_threshold": ACCURACY_THRESHOLD,
            "membrane_reset": MEMBRANE_RESET,
        },
        "tested_parameter": {
            "name": parameter_name,
            "values": parameter_values,
        },
        "weight_segments_rf": weight_segments_rf,
        "weight_segments_slp": weight_segments_slp,
    }

    # 💡 qui puliamo tutto
    metadata = _to_builtin(metadata)

    yaml_path = os.path.join(results_dir, "experiment_metadata.yaml")
    with open(yaml_path, "w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

def compute_mean_I(inputs: np.ndarray):
    """
    Estimate the average input current injected into the network.
    """
    avg_input_current = (
        np.sum(inputs)
        / (inputs.shape[0] * inputs.shape[2])
    )

    return avg_input_current

def compute_critical_weight(
    inputs: np.ndarray,
    parameter_name: str,
    parameter_value: float,
):
    """
    Estimate the average input current and the critical synaptic weight.
    """
    beta = PRESYNAPTIC_DEGREE
    current_amplitude = CURRENT_AMPLITUDE
    membrane_threshold = MEMBRANE_THRESHOLD

    if parameter_name == "beta":
        beta = parameter_value
    elif parameter_name == "current_amplitude":
        current_amplitude = parameter_value
    elif parameter_name == "membrane_threshold":
        membrane_threshold = parameter_value

    avg_input_current = (
        np.sum(inputs)
        / (NUM_NEURONS * inputs.shape[0] * inputs.shape[2])
    )

    critical_weight = (
        membrane_threshold
        - 2 * avg_input_current * current_amplitude * (REFRACTORY_PERIOD + 1)
    ) / (beta * NUM_NEURONS)

    return avg_input_current, critical_weight


def test_parameter_values(data, labels , param_name: str, param_values: list[float]):
    """
    For each value in param_values:
      - set the parameter in the SNN simulation
      - sweep several mean_weight values
      - run simulate_trace + cross_validation_rf

    Returns:
      list[dict]: each dict has keys: param_value, weight, accuracy
    """
    sim_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=0.0,
        weight_variance=0.0,
        current_amplitude=CURRENT_AMPLITUDE,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_GRAPH_P,
        small_world_graph_k=SMALL_WORLD_GRAPH_K,
        input_spike_times=np.zeros(
            (data.shape[1], data.shape[2]),
            dtype=np.uint8,
        ),
    )

    all_results: list[dict] = []

    for param_value in param_values:
        print(f"\n### Testing {param_name} = {param_value} ###")

        # set the parameter we are testing
        if param_name == "beta":
            # convert beta to k = beta * 2 * N
            sim_params.small_world_graph_k = int(param_value * NUM_NEURONS * 2)
        elif param_name == "current_amplitude":
            sim_params.current_amplitude = param_value
        elif param_name == "membrane_threshold":
            sim_params.membrane_threshold = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        _, critical_weight = compute_critical_weight(
            data,
            param_name,
            param_value,
        )

        weight_values = np.linspace(
            critical_weight / 100,
            critical_weight * 1.4,
            NUM_WEIGHT_STEPS,
        )
        
        cnt=0
        for weight in weight_values:
            cnt += 1
            print(f"\n--- mean_weight = {weight:.6f} --- execution {cnt}/{NUM_WEIGHT_STEPS} with {param_name} = {param_value}")
            sim_params.mean_weight = weight
            sim_params.weight_variance =  15
            
            if OUTPUT_FEATURES == "statistics":
                trace_dataset, spike_count = simulate_statistic_features(
                    data=data,
                    labels=labels,
                    parameters=sim_params,
                    statistic_set=STATISTIC_SET,
                    membrane_reset = MEMBRANE_RESET,
                )
            elif OUTPUT_FEATURES == "trace":
                trace_dataset, spike_count = simulate_trace(
                    data=data,
                    labels=labels,
                    parameters=sim_params,
                    trace_tau=TRACE_TAU,
                    membrane_reset = MEMBRANE_RESET,
                )


            mean_accuracy_rf, std_accuracy_rf, mean_f1_rf, std_f1_rf, mean_mcc_rf, std_mcc_rf = cross_validation_rf(
                trace_dataset,
                n_splits=CV_NUM_SPLITS,
            )
            print("Mean accuracy: ", mean_accuracy_rf, " std accuracy: ", std_accuracy_rf)

            mean_accuracy_slp, std_accuracy_slp, mean_f1_slp, std_f1_slp, mean_mcc_slp, std_mcc_slp = cross_validation_slp(
                trace_dataset,
                n_splits=CV_NUM_SPLITS,
            )
            print("Mean accuracy: ", mean_accuracy_slp, " std accuracy: ", std_accuracy_slp)

            all_results.append(
                {
                    "param_value": float(param_value),
                    "weight": float(weight),
                    "accuracy_rf": float(mean_accuracy_rf),
                    "std_accuracy_rf": float(std_accuracy_rf),                    
                    "accuracy_slp": float(mean_accuracy_slp),
                    "std_accuracy_slp": float(std_accuracy_slp),
                    "f1_rf": float(mean_f1_rf),
                    "std_f1_rf": float(std_f1_rf),                    
                    "f1_slp": float(mean_f1_slp),
                    "std_f1_slp": float(std_f1_slp),
                    "mcc_rf": float(mean_mcc_rf),
                    "std_mcc_rf": float(std_mcc_rf),                    
                    "mcc_slp": float(mean_mcc_slp),
                    "std_mcc_slp": float(std_mcc_slp),
                    "spike_count": float(spike_count)
                }
            )

    return all_results

def main():
    load_config()
    
    os.makedirs("dati", exist_ok=True)

    data, labels = load_dataset(DATASET_PATH)
    print(f"Loaded data: {data.shape}, labels: {labels.shape}")
    mean_I = compute_mean_I(data)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = test_parameter_values(data, labels, PARAM_NAME, PARAMETER_VALUES)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(CSV_NAME, index=False)
    print(f"Saved results to {CSV_NAME}")

    # 🔹 calcolo solo i segmenti, senza plottare ancora
    weight_segments_rf = {}
    weight_segments_slp = {}

    for parameter in PARAMETER_VALUES:
        parameter_df = results_df[results_df["param_value"] == parameter].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        # ====== segmento RF ======
        max_accuracy_rf = parameter_df["accuracy_rf"].max()
        threshold_rf = ACCURACY_THRESHOLD * max_accuracy_rf
        eligible_rf = parameter_df[parameter_df["accuracy_rf"] >= threshold_rf]

        if not eligible_rf.empty:
            w1_rf = float(eligible_rf["weight"].min())
            w2_rf = float(eligible_rf["weight"].max())
            segment_length_rf = float(w2_rf - w1_rf)
            weight_segments_rf[float(parameter)] = {
                "w1": w1_rf,
                "w2": w2_rf,
                "delta": segment_length_rf,
            }
        else:
            weight_segments_rf[float(parameter)] = {
                "w1": None,
                "w2": None,
                "delta": None,
            }

        # ====== segmento SLP ======
        max_accuracy_slp = parameter_df["accuracy_slp"].max()
        threshold_slp = ACCURACY_THRESHOLD * max_accuracy_slp
        eligible_slp = parameter_df[parameter_df["accuracy_slp"] >= threshold_slp]

        if not eligible_slp.empty:
            w1_slp = float(eligible_slp["weight"].min())
            w2_slp = float(eligible_slp["weight"].max())
            segment_length_slp = float(w2_slp - w1_slp)
            weight_segments_slp[float(parameter)] = {
                "w1": w1_slp,
                "w2": w2_slp,
                "delta": segment_length_slp,
            }
        else:
            weight_segments_slp[float(parameter)] = {
                "w1": None,
                "w2": None,
                "delta": None,
            }

    # 🔹 salvataggio metadata
    save_experiment_metadata(
        results_dir=RESULTS_DIR,
        parameter_name=PARAM_NAME,
        parameter_values=PARAMETER_VALUES,
        weight_segments_rf=weight_segments_rf,
        weight_segments_slp=weight_segments_slp,
        mean_I=mean_I,
    )

    # === FIGURE SEPARATE ===
    plt_rf = plt.figure()
    ax_rf = plt_rf.add_subplot(111)

    plt_slp = plt.figure()
    ax_slp = plt_slp.add_subplot(111)

    for parameter in PARAMETER_VALUES:
        parameter_df = results_df[results_df["param_value"] == parameter].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        # ----- RF -----
        line_rf, = ax_rf.plot(
            parameter_df["weight"],
            parameter_df["accuracy_rf"],
            marker="o",
            label=f"RF – {PARAM_NAME}={parameter}",
        )
        lower_rf = parameter_df["accuracy_rf"] - parameter_df["std_accuracy_rf"]
        upper_rf = parameter_df["accuracy_rf"] + parameter_df["std_accuracy_rf"]
        ax_rf.fill_between(
            parameter_df["weight"],
            lower_rf,
            upper_rf,
            color=line_rf.get_color(),
            alpha=0.2,
        )

        # qui posso riusare il segmento già calcolato
        seg_rf = weight_segments_rf[float(parameter)]
        if seg_rf["w1"] is not None:
            max_accuracy_rf = parameter_df["accuracy_rf"].max()
            threshold_rf = ACCURACY_THRESHOLD * max_accuracy_rf
            ax_rf.hlines(
                y=threshold_rf,
                xmin=seg_rf["w1"],
                xmax=seg_rf["w2"],
                colors="black",
                linestyles="dashed",
            )

        # ----- SLP -----
        line_slp, = ax_slp.plot(
            parameter_df["weight"],
            parameter_df["accuracy_slp"],
            marker="s",
            linestyle="--",
            label=f"SLP – {PARAM_NAME}={parameter}",
        )
        lower_slp = parameter_df["accuracy_slp"] - parameter_df["std_accuracy_slp"]
        upper_slp = parameter_df["accuracy_slp"] + parameter_df["std_accuracy_slp"]
        ax_slp.fill_between(
            parameter_df["weight"],
            lower_slp,
            upper_slp,
            color=line_slp.get_color(),
            alpha=0.2,
        )

        seg_slp = weight_segments_slp[float(parameter)]
        if seg_slp["w1"] is not None:
            max_accuracy_slp = parameter_df["accuracy_slp"].max()
            threshold_slp = ACCURACY_THRESHOLD * max_accuracy_slp
            ax_slp.hlines(
                y=threshold_slp,
                xmin=seg_slp["w1"],
                xmax=seg_slp["w2"],
                colors="black",
                linestyles="dashed",
            )

    # setup RF
    ax_rf.set_xlabel("Mean synaptic weight")
    ax_rf.set_ylabel("Mean CV accuracy (RF)")
    ax_rf.set_title(f"RF – accuracy vs. weight for different {PARAM_NAME} values")
    ax_rf.grid(True)
    ax_rf.legend()
    plt_rf.tight_layout()
    plot_path_rf = os.path.join(RESULTS_DIR, f"{PARAM_NAME}_{NUM_WEIGHT_STEPS}_rf.png")
    plt_rf.savefig(plot_path_rf)
    print(f"Saved RF plot to {plot_path_rf}")

    # setup SLP
    ax_slp.set_xlabel("Mean synaptic weight")
    ax_slp.set_ylabel("Mean CV accuracy (SLP)")
    ax_slp.set_title(f"SLP – accuracy vs. weight for different {PARAM_NAME} values")
    ax_slp.grid(True)
    ax_slp.legend()
    plt_slp.tight_layout()
    plot_path_slp = os.path.join(RESULTS_DIR, f"{PARAM_NAME}_{NUM_WEIGHT_STEPS}_slp.png")
    plt_slp.savefig(plot_path_slp)
    print(f"Saved SLP plot to {plot_path_slp}")

    plt.show()


if __name__ == "__main__":
    main()