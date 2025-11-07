import os
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== PARAMETRI "DI TESTA" (come nello script originale) ==================
TASK = "MNIST"  # es: "MNIST"
OUTPUT_FEATURES = "trace"  # "statistics" oppure "trace"

# se vuoi ricalcolare mean_I metti il path qui, altrimenti lascia None
DATASET_PATH = "dati/mnist_rate_encoded.npz"

CV_NUM_SPLITS = 10
ACCURACY_THRESHOLD = 0.8

NUM_NEURONS = 1000
MEMBRANE_THRESHOLD = 2
REFRACTORY_PERIOD = 2
NUM_OUTPUT_NEURONS = 50
LEAK_COEFFICIENT = 0
CURRENT_AMPLITUDE = MEMBRANE_THRESHOLD  # come nel tuo codice
PRESYNAPTIC_DEGREE = 0.2
SMALL_WORLD_GRAPH_P = 0.2
SMALL_WORLD_GRAPH_K = int(PRESYNAPTIC_DEGREE * NUM_NEURONS * 2)

TRACE_TAU = 60
NUM_WEIGHT_STEPS = 51

# questo deve corrispondere a quello usato per generare il CSV
PARAM_NAME = "current_amplitude"  # "beta", "membrane_threshold", "current_amplitude"
#PARAMETER_VALUES = [2, 1.42963091165, 1.1048193827]    # lista dei valori testati per membrane_threshold (serve per fare i plot e lo yaml)
#PARAMETER_VALUES = [0.2, 0.3, 0.4] # lista dei valori testati per beta (serve per fare i plot e lo yaml)
PARAMETER_VALUES = [0.5, 1, 2] # lista dei valori testati per current_amplitude (serve per fare i plot e lo yaml)

# ricostruiamo nomi/cartelle come nello script originale
today_str = "2025_11_06"
RESULTS_DIR = f"results/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{today_str}"
CSV_NAME = os.path.join(
    RESULTS_DIR,
    f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
)
PLOT_NAME = os.path.join(RESULTS_DIR, f"{PARAM_NAME}_{NUM_WEIGHT_STEPS}.png")
YAML_NAME = os.path.join(RESULTS_DIR, "experiment_metadata.yaml")
# ======================================================================


def load_dataset_for_mean_I(filename: str) -> float | None:
    """
    Se esiste un dataset .npz lo usiamo per ricalcolare mean_I come nel codice originale.
    Altrimenti restituiamo None.
    """
    if filename is None:
        return None
    if not os.path.exists(filename):
        return None

    npz_data = np.load(filename)
    inputs = npz_data["data"]  # shape: (n_samples, ?, timesteps)
    # nel tuo codice originale: np.sum(inputs) / (inputs.shape[0] * inputs.shape[2])
    avg_input_current = np.sum(inputs) / (inputs.shape[0] * inputs.shape[2])
    return float(avg_input_current)


def save_experiment_metadata(
    yaml_path: str,
    mean_I,
    weight_segments: dict,
):
    """
    Scrive il file YAML con la stessa struttura del codice originale.
    """
    metadata = {
        "experiment": {
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
        },
        "tested_parameter": {
            "name": PARAM_NAME,
            "values": [float(v) for v in PARAMETER_VALUES],
        },
        "weight_segments": weight_segments,
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def reconstruct_from_csv():
    # 1. leggi csv
    if not os.path.exists(CSV_NAME):
        raise FileNotFoundError(f"CSV non trovato: {CSV_NAME}")

    df = pd.read_csv(CSV_NAME)

    # check colonnine
    expected_cols = {"param_value", "weight", "accuracy"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Il CSV deve contenere almeno le colonne {expected_cols}, trovato: {df.columns}"
        )

    # 2. ricostruisci i segmenti come nel main originale
    weight_segments: dict[float, dict[str, float | None]] = {}

    plt.figure()

    for parameter in PARAMETER_VALUES:
        # filtra per param_value
        param_df = df[df["param_value"] == float(parameter)].copy()
        if param_df.empty:
            # se nel csv non c'è quel parametro lo segniamo vuoto
            weight_segments[float(parameter)] = {
                "w1": None,
                "w2": None,
                "delta": None,
            }
            continue

        # ordina per weight
        param_df = param_df.sort_values(by="weight")

        # plottiamo la curva
        plt.plot(
            param_df["weight"],
            param_df["accuracy"],
            marker="o",
            label=f"{PARAM_NAME}={parameter}",
        )

        # stesso criterio: soglia relativa al massimo di quel parametro
        max_accuracy = param_df["accuracy"].max()
        threshold = ACCURACY_THRESHOLD * max_accuracy

        eligible = param_df[param_df["accuracy"] >= threshold]

        if not eligible.empty:
            w1 = float(eligible["weight"].min())
            w2 = float(eligible["weight"].max())
            delta = float(w2 - w1)

            weight_segments[float(parameter)] = {
                "w1": w1,
                "w2": w2,
                "delta": delta,
            }

            # disegniamo la linea orizzontale tratteggiata come prima
            plt.hlines(
                y=threshold,
                xmin=w1,
                xmax=w2,
                colors="black",
                linestyles="dashed",
            )
        else:
            weight_segments[float(parameter)] = {
                "w1": None,
                "w2": None,
                "delta": None,
            }

    # 3. ricalcola (eventuale) mean_I
    mean_I = load_dataset_for_mean_I(DATASET_PATH)

    # 4. salva yaml
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_experiment_metadata(
        yaml_path=YAML_NAME,
        mean_I=mean_I,
        weight_segments=weight_segments,
    )

    # 5. systematizziamo il plot
    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean CV accuracy")
    plt.title(f"Accuracy vs. weight for different {PARAM_NAME} values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_NAME)
    print(f"Salvato plot in {PLOT_NAME}")
    print(f"Salvato yaml in {YAML_NAME}")

    # se vuoi anche vedere il grafico a schermo
    # plt.show()


if __name__ == "__main__":
    reconstruct_from_csv()
