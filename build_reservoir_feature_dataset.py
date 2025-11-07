from pathlib import Path
from typing import List, Tuple

import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pprint import pprint

from snnpy.snn import SNN, SimulationParams, STDPParams

# ------------------------------
# Paths
# ------------------------------
DATA_DIR = Path("dati")
CFG_DIR = Path("config")

DATASET_PATH = DATA_DIR / "retina_gesture_dataset_32x32_with_random.npz"
FEATURES_OUT_CSV = DATA_DIR / "snn_features_scaled.csv"
TOPOLOGY_OUT_NPZ = DATA_DIR / "topology_stdp.npz"
MEMBRANES_OUT_NPY = DATA_DIR / "membrane_potentials.npy"
OUTPUT_NEURONS_OUT_NPY = DATA_DIR / "output_neurons.npy"

SIM_YAML = CFG_DIR / "simulation.yaml"
STDP_YAML = CFG_DIR / "stdp.yaml"  # optional


# ------------------------------
# Config loaders
# ------------------------------
def load_simulation_params(path: Path = SIM_YAML) -> SimulationParams:
    """Load SimulationParams from YAML (field names match dataclass)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing simulation config: {path}")
    with path.open("r", encoding="utf-8") as f:
        sim_cfg = yaml.safe_load(f) or {}

    # Do NOT load input_spike_times from YAML; will be set from data.
    sim_cfg.pop("input_spike_times", None)
    return SimulationParams(**sim_cfg)


def load_stdp_params_if_any(path: Path = STDP_YAML) -> STDPParams | None:
    """Load STDPParams from YAML if present; otherwise return None."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        stdp_cfg = yaml.safe_load(f) or {}
    # If lock_A_minus is true, avoid user-provided A_minus to prevent conflicts.
    if stdp_cfg.get("lock_A_minus", False) and "A_minus" in stdp_cfg:
        stdp_cfg.pop("A_minus")
    return STDPParams(**stdp_cfg)


# ------------------------------
# Data helpers
# ------------------------------
def load_dataset(path: Path = DATASET_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from NPZ: returns (data, labels)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path) as npz:
        return npz["data"], npz["labels"]


def build_feature_names(n_outputs: int) -> List[str]:
    """Column names per output neuron."""
    cols: List[str] = []
    for i in range(n_outputs):
        cols += [
            f"neuron{i}_mean_time",
            f"neuron{i}_first_time",
            f"neuron{i}_last_time",
            f"neuron{i}_mean_isi",
            f"neuron{i}_isi_var",
        ]
    cols.append("label")
    return cols


# ------------------------------
# Main pipeline
# ------------------------------
def main() -> None:
    # Ensure folders exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    # Load config
    params = load_simulation_params()
    stdp = load_stdp_params_if_any()  # may be None

    # Init model
    if stdp is not None:
        snn = SNN(simulation_params=params, stdp_params=stdp)
    else:
        snn = SNN(simulation_params=params)

    # Save initial state (useful for reproducibility)
    snn.save_membrane_potentials(str(MEMBRANES_OUT_NPY))
    snn.save_output_neurons(str(OUTPUT_NEURONS_OUT_NPY))
    snn.save_topology(str(TOPOLOGY_OUT_NPZ))

    snn.apply_global_scalar_threshold(use_abs_weights=False)

    # Load data
    data, labels = load_dataset()
    n_samples = len(data)
    init_v = snn.get_membrane_potentials()

    rows: list[list[float]] = []
    
    for i in range(n_samples):
        if i % 10 == 0:
            print(f"[{i}/{n_samples}] processing…")

        # Dataset-specific reshape: to (input_neurons, time)
        spikes = data[i].reshape(100, -1).T  # (time, 100) -> transpose to (100, time)
        snn.set_input_spike_times(spikes)
        snn.set_membrane_potentials(init_v)

        # If STDP is enabled, re-init traces after resetting potentials
        if stdp is not None and stdp.enabled:
            snn._init_stdp()
            # Optional: homeostatic thresholding from mean-field heuristic
            snn.apply_global_scalar_threshold(use_abs_weights=False)

        # Run simulation
        snn.simulate()

        # Collect features (per output neuron)
        feats = np.stack(
            [
                snn.get_mean_spike_times(),
                snn.get_first_spike_times(),
                snn.get_last_spike_times(),
                snn.get_mean_isi_per_neuron(),
                snn.get_isi_variance_per_neuron(),
            ],
            axis=1,
        )
        row = feats.flatten().tolist()
        row.append(float(labels[i]))
        rows.append(row)

    # Build DataFrame and scale features
    num_outputs = params.num_output_neurons if params.num_output_neurons is not None else len(snn.get_output_neurons())
    columns = build_feature_names(num_outputs)
    df = pd.DataFrame(rows, columns=columns)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=["label"]))
    df_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])
    df_scaled["label"] = df["label"].to_numpy()

    # Persist outputs
    df_scaled.to_csv(FEATURES_OUT_CSV, index=False)

    # Inspect run parameters
    pprint(snn.get_network_parameters())
    print(f"\n✔ Features saved to: {FEATURES_OUT_CSV}")
    print(f"✔ Topology saved to: {TOPOLOGY_OUT_NPZ}")
    if stdp is None:
        print("ℹ STDP config not found; simulation ran without STDP.")
    else:
        print("ℹ STDP enabled.")

    from scipy.sparse import issparse, csr_matrix
    
    W = snn.get_topology()            # CSR (non-zero weights in W.data)
    if not issparse(W):
        W = csr_matrix(W)            # in caso non fosse già sparsa

    if W.nnz == 0:
        mean_w = 0.0                 # nessun peso non-zero
    else:
        mean_w = float(W.data.mean())            # media dei pesi non-zero
        mean_abs_w = float(np.mean(np.abs(W.data)))  # (opzionale) media in valore assoluto

if __name__ == "__main__":
    main()
