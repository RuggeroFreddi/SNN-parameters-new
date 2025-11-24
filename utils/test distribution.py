import os
import numpy as np

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent  # la cartella sopra utils
sys.path.append(str(ROOT))
from LSM.model import SimulationParams, Reservoir

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
LEAK_COEFFICIENT = 0.002
CURRENT_AMPLITUDE = MEMBRANE_THRESHOLD
PRESYNAPTIC_DEGREE = 0.1
SMALL_WORLD_GRAPH_P = 0.2

TRACE_TAU = 60

import numpy as np
from scipy.sparse import csr_matrix, issparse

def analyze_synaptic_weights(W: csr_matrix):
    """
    Analizza una matrice di pesi sinaptici (CSR).

    Restituisce:
        mean_weight: float              # media globale dei pesi non nulli
        cv_weight: float                # std/mean globale
        n_exc_neurons: int              # numero neuroni eccitatori (righe solo > 0)
        n_inh_neurons: int              # numero neuroni inibitori (righe solo < 0)
        mean_exc: float                 # media pesi eccitatori (su singole sinapsi)
        var_exc: float                  # varianza pesi eccitatori
        mean_inh: float                 # media pesi inibitori (negativa)
        var_inh: float                  # varianza pesi inibitori
    Convenzioni:
        - neurone eccitatorio: tutte le connessioni in uscita > 0
        - neurone inibitorio: tutte le connessioni in uscita < 0
        - righe senza uscite o con segno misto non vengono conteggiate in nessuno dei due.
    """
    if not issparse(W):
        W = csr_matrix(W)
    W = W.tocsr()

    data = W.data
    if data.size == 0:
        return 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0

    mean_weight = float(data.mean())
    std_weight = float(data.std())
    cv_weight = std_weight / mean_weight if mean_weight != 0 else 0.0

    N = W.shape[0]
    indptr = W.indptr
    data = W.data  # alias

    excitatory_mask = np.zeros(N, dtype=bool)
    inhibitory_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue  # nessuna connessione in uscita
        row_w = data[start:end]
        if np.all(row_w > 0):
            excitatory_mask[i] = True
        elif np.all(row_w < 0):
            inhibitory_mask[i] = True
        # misto -> non classificato

    n_exc_neurons = int(excitatory_mask.sum())
    n_inh_neurons = int(inhibitory_mask.sum())

    # ---- statistiche sui singoli pesi E/I ----
    # costruiamo l'indice di riga per ogni entry in data
    row_indices = np.repeat(
        np.arange(N, dtype=np.int32),
        np.diff(indptr).astype(np.int32)
    )

    edge_is_exc = excitatory_mask[row_indices]
    edge_is_inh = inhibitory_mask[row_indices]

    exc_weights = data[edge_is_exc]
    inh_weights = data[edge_is_inh]

    if exc_weights.size > 0:
        mean_exc = float(exc_weights.mean())
        var_exc = float(exc_weights.var())
    else:
        mean_exc = 0.0
        var_exc = 0.0

    if inh_weights.size > 0:
        mean_inh = float(inh_weights.mean())
        var_inh = float(inh_weights.var())
    else:
        mean_inh = 0.0
        var_inh = 0.0

    return (
        mean_weight,
        cv_weight,
        n_exc_neurons,
        n_inh_neurons,
        mean_exc,
        var_exc,
        mean_inh,
        var_inh,
    )



import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def plot_weight_distribution(
    W: csr_matrix,
    bins: int = 100,
    log_y: bool = False,
    clip_quantiles=(0.001, 0.999),  # quantili per tagliare le code
) -> None:
    """
    Plot della distribuzione dei pesi di una matrice sparsa CSR.
    
    Parametri
    ----------
    W : csr_matrix
        Matrice dei pesi (solo i valori non nulli vengono considerati).
    bins : int, default=100
        Numero di bin dell'istogramma.
    log_y : bool, default=False
        Se True usa scala logaritmica sull'asse y.
    clip_quantiles : tuple(float, float) or None
        Se non None, considera solo i pesi tra questi quantili (es. (0.001, 0.999)).
    """
    if not isinstance(W, csr_matrix):
        raise TypeError("W deve essere una scipy.sparse.csr_matrix")

    weights = W.data
    if weights.size == 0:
        print("La matrice non contiene pesi non nulli.")
        return

    # Taglia le code sottili usando i quantili
    if clip_quantiles is not None:
        q_low, q_high = clip_quantiles
        lo = np.quantile(weights, q_low)
        hi = np.quantile(weights, q_high)
        mask = (weights >= lo) & (weights <= hi)
        weights_plot = weights[mask]
    else:
        weights_plot = weights

    # Medie dei pesi positivi, negativi e globale (su TUTTI i pesi non nulli)
    mean_all = weights.mean()
    pos_weights = weights[weights > 0]
    neg_weights = weights[weights < 0]
    mean_pos = pos_weights.mean() if pos_weights.size > 0 else None
    mean_neg = neg_weights.mean() if neg_weights.size > 0 else None

    plt.figure(figsize=(6, 4))
    plt.hist(weights_plot, bins=bins, density=True, edgecolor="black", alpha=0.7)

    # Linee verticali:
    # - nere: media globale
    # - rosse tratteggiate: medie positiva e negativa
    plt.axvline(mean_all, color="black", linestyle="-", linewidth=1.5, label="mean all")

    if mean_pos is not None:
        plt.axvline(mean_pos, color="red", linestyle="--", linewidth=1.5, label="mean +")
    if mean_neg is not None:
        plt.axvline(mean_neg, color="red", linestyle="--", linewidth=1.5, label="mean -")

    plt.xlabel("Peso sinaptico")
    plt.ylabel("Densità")
    plt.title("Distribuzione dei pesi sinaptici (non nulli)")
    if log_y:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.show()

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
    # critical_weight = 0.0038 

    small_world_graph_k = int(PRESYNAPTIC_DEGREE * NUM_NEURONS * 2)

    sim_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=critical_weight,
        weight_variance= 50,
        leak_cv = 10,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_GRAPH_P,
        small_world_graph_k=small_world_graph_k,
        mean_distance = 5 * critical_weight , 
        # mean_distance / critical_weight = radice ((sdt_desiderata^2 - weight_variance^2)/ (fefi)), dove fefi=0-16
        # es radice ((20^2 - 10^2)/0.16) = 43.3 => weight_variance= 10, mean_distance = 44 * critical_weight danno una std totale di circa 20
        input_spike_times=np.zeros(
            (data.shape[1], data.shape[2]),
            dtype=np.uint8,
        ),
    )
    
    snn = Reservoir(sim_params)

    (
        mean_w,
        cv_w,
        n_exc,
        n_inh,
        mean_exc,
        var_exc,
        mean_inh,
        var_inh,
    ) = analyze_synaptic_weights(snn.synaptic_weights)
    print("critical weight: ", critical_weight)
    print("global mean:", mean_w, "cv:", cv_w)
    print("neurons  E:", n_exc, "I:", n_inh)
    print("E weights mean/var:", mean_exc, np.sqrt(var_exc))
    print("I weights mean/var:", mean_inh, np.sqrt(var_inh))


    plot_weight_distribution(snn.synaptic_weights, clip_quantiles=(0.01, 0.99))


if __name__ == "__main__":
    main()
