import os
import pandas as pd
import numpy as np

# ==============================
# CONFIGURAZIONE
# ==============================

BASE_RESULTS_DIR = "results"

TASKS = ["MNIST", "TRAJECTORY"]
OUTPUT_FEATURES_LIST = ["statistics", "trace"]
PARAM_NAMES = ["beta", "membrane_threshold", "current_amplitude"]

# NUM_WEIGHT_STEPS per task
NUM_WEIGHT_STEPS_MAP = {
    "MNIST": 101,
    "TRAJECTORY": 71,
}

# metriche e colonne std corrispondenti
METRIC_PAIRS = [
    ("accuracy_rf",  "std_accuracy_rf"),
    ("accuracy_slp", "std_accuracy_slp"),
    ("f1_rf",        "std_f1_rf"),
    ("f1_slp",       "std_f1_slp"),
    ("mcc_rf",       "std_mcc_rf"),
    ("mcc_slp",      "std_mcc_slp"),
]

# mappa curva -> categoria (per calcolare i rapporti medi per acc / f1 / mcc)
METRIC_CATEGORY = {
    "accuracy_rf": "accuracy",
    "accuracy_slp": "accuracy",
    "f1_rf": "f1",
    "f1_slp": "f1",
    "mcc_rf": "mcc",
    "mcc_slp": "mcc",
}

# Struttura globale: global_stats[task][metric] = lista di dict con info
global_stats = {
    task: {metric: [] for metric, _ in METRIC_PAIRS}
    for task in TASKS
}

# ==============================
# FUNZIONI DI SUPPORTO
# ==============================

def find_latest_result_dir(task: str, output_features: str, param_name: str):
    """
    Cerca in BASE_RESULTS_DIR una directory del tipo:
        results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{DATE}
    e restituisce quella con la data "massima" (più recente).
    Se non ne trova, restituisce None.
    """
    if not os.path.isdir(BASE_RESULTS_DIR):
        return None

    prefix = f"results_{task}_{output_features}_{param_name}_"
    candidates = []

    for d in os.listdir(BASE_RESULTS_DIR):
        full_path = os.path.join(BASE_RESULTS_DIR, d)
        if os.path.isdir(full_path) and d.startswith(prefix):
            candidates.append(d)

    if not candidates:
        return None

    # dato il formato YYYY_MM_DD, l'ordine lessicografico coincide con l'ordine cronologico
    latest_dir_name = sorted(candidates)[-1]
    return os.path.join(BASE_RESULTS_DIR, latest_dir_name)


def process_csv(task: str, output_features: str, param_name: str, csv_path: str):
    """
    Carica il CSV e aggiorna global_stats[task] con il coefficiente di variazione medio
    (media di std/mean sui weight) per ciascun param_value e ciascuna curva.
    """
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV non trovato: {csv_path}")
        return

    print(f"\n=== Processing: TASK={task}, OUT={output_features}, PARAM={param_name} ===")
    print(f"CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # controllo colonne minime
    expected_cols = {
        "param_value",
        "weight",
        "accuracy_rf", "std_accuracy_rf",
        "accuracy_slp", "std_accuracy_slp",
        "f1_rf", "std_f1_rf",
        "f1_slp", "std_f1_slp",
        "mcc_rf", "std_mcc_rf",
        "mcc_slp", "std_mcc_slp",
    }
    if not expected_cols.issubset(df.columns):
        print(f"[WARNING] Colonne mancanti nel CSV {csv_path}. Skipping.")
        return

    # per ogni valore di parametro
    for param_value, group in df.groupby("param_value"):
        for metric, metric_std in METRIC_PAIRS:
            means = group[metric].to_numpy()
            stds  = group[metric_std].to_numpy()

            # usiamo solo i punti con mean > 0 e finiti
            mask = np.isfinite(means) & np.isfinite(stds) & (means > 0)
            valid_means = means[mask]
            valid_stds  = stds[mask]

            if valid_means.size == 0:
                mean_cv = np.nan
            else:
                cv = valid_stds / valid_means          # cv_i = std_i / mean_i per ogni w
                mean_cv = float(np.nanmean(cv))        # media delle cv sulla curva

            # memorizzo info globale
            global_stats[task][metric].append({
                "mean_cv": mean_cv,
                "task": task,
                "output_features": output_features,
                "param_name": param_name,
                "param_value": float(param_value),
                "csv_path": csv_path,
            })


# ==============================
# SCANSIONE DI TUTTE LE COMBINAZIONI
# ==============================

for task in TASKS:
    for output_features in OUTPUT_FEATURES_LIST:
        for param_name in PARAM_NAMES:
            latest_dir = find_latest_result_dir(task, output_features, param_name)
            if latest_dir is None:
                print(f"[INFO] Nessuna directory trovata per "
                      f"TASK={task}, OUT={output_features}, PARAM={param_name}")
                continue

            num_steps = NUM_WEIGHT_STEPS_MAP[task]
            csv_name = os.path.join(
                latest_dir,
                f"experiment_{param_name}_{num_steps}.csv",
            )

            process_csv(task, output_features, param_name, csv_name)

# ==============================
# RAPPORTO TRA CV TRAJECTORY / MNIST + RAPPORTI MEDI PER CATEGORIA
# ==============================

print("\n\n====================================")
print("RAPPORTO MEAN(CV): TRAJECTORY / MNIST")
print("====================================\n")

# per accumulare rapporti medi per categoria (accuracy / f1 / mcc)
category_ratios = {"accuracy": [], "f1": [], "mcc": []}

for metric, _ in METRIC_PAIRS:
    print(f"\n==============================")
    print(f"Curva: {metric}")
    print(f"==============================")

    # indicizzazione: (output_features, param_name, param_value) -> mean_cv
    idx_mnist = {}
    idx_traj  = {}

    for entry in global_stats["MNIST"][metric]:
        key = (entry["output_features"], entry["param_name"], entry["param_value"])
        idx_mnist[key] = entry

    for entry in global_stats["TRAJECTORY"][metric]:
        key = (entry["output_features"], entry["param_name"], entry["param_value"])
        idx_traj[key] = entry

    common_keys = sorted(set(idx_mnist.keys()) & set(idx_traj.keys()),
                         key=lambda x: (x[0], x[1], x[2]))  # ordina per OUT, PARAM, param_value

    if not common_keys:
        print("  Nessuna coppia MNIST/TRAJECTORY corrispondente trovata.")
        continue

    for (out_feat, param_name, param_value) in common_keys:
        mn_entry = idx_mnist[(out_feat, param_name, param_value)]
        tr_entry = idx_traj[(out_feat, param_name, param_value)]

        mn_cv = mn_entry["mean_cv"]
        tr_cv = tr_entry["mean_cv"]

        if (not np.isfinite(mn_cv)) or mn_cv <= 0 or (not np.isfinite(tr_cv)):
            ratio = np.nan
        else:
            ratio = tr_cv / mn_cv

        print(f"  OUT={out_feat}, PARAM={param_name}, param_value={param_value:.6g}")
        print(f"    mean(CV) MNIST      = {mn_cv:.6g}")
        print(f"    mean(CV) TRAJECTORY = {tr_cv:.6g}")
        print(f"    TRAJECTORY / MNIST  = {ratio:.6g}")

        # accumulo per categoria, se il rapporto è valido
        if np.isfinite(ratio):
            category = METRIC_CATEGORY[metric]  # "accuracy", "f1" o "mcc"
            category_ratios[category].append(ratio)

# ==============================
# RIEPILOGO: RAPPORTI MEDI PER ACC / F1 / MCC
# ==============================

print("\n\n====================================")
print("RAPPORTI MEDI (TRAJECTORY / MNIST) PER CATEGORIA")
print("====================================\n")

for cat in ["accuracy", "f1", "mcc"]:
    vals = np.array(category_ratios[cat], dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        print(f"{cat}: nessun rapporto valido.")
    else:
        mean_ratio = float(np.mean(vals))
        std_ratio  = float(np.std(vals))
        print(f"{cat}: mean ratio = {mean_ratio:.6g}, std = {std_ratio:.6g}, n = {vals.size}")
