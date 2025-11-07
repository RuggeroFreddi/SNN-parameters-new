import os
import pandas as pd
import yaml

# ================== PARAMETRI COME NEL TUO PROGRAMMA ==================
TASK = "MNIST"  # "MNIST"
OUTPUT_FEATURES = "trace"  # "statistics" | "trace"

# questo è importante: nel tuo secondo programma la cartella ha anche il nome del parametro
PARAM_NAME = "membrane_threshold"  # "beta" | "membrane_threshold" | "current_amplitude"

# PARAMETER_VALUES = [0.2, 0.3, 0.4] # use it when PARM_NAME = "beta"
PARAMETER_VALUES = [2, 1.42963091165, 1.1048193827] # use it when PARM_NAME = "membrane_threshold"
# PARAMETER_VALUES = [0.5, 1, 2] # use it when PARM_NAME = "current_amplitude"

NUM_WEIGHT_STEPS = 51  # deve essere lo stesso di quando hai fatto l'esperimento
# ======================================================================

# ricostruiamo la stessa directory e lo stesso csv
date_str = "2025_11_04"
RESULTS_DIR = f"results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{date_str}"
CSV_NAME = os.path.join(
    RESULTS_DIR,
    f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
)


def main():
    if not os.path.exists(CSV_NAME):
        raise FileNotFoundError(f"CSV non trovato: {CSV_NAME}")

    df = pd.read_csv(CSV_NAME)

    # ci aspettiamo almeno queste colonne
    for col in ["param_value", "weight", "accuracy", "spike_count"]:
        if col not in df.columns:
            raise ValueError(f"Nel CSV manca la colonna '{col}'")

    # qui metteremo i risultati da stampare in yaml
    out = {}

    # vogliamo farlo per i parametri che DICIAMO noi in testa
    # (così è parametrico e riproducibile)
    for p in PARAMETER_VALUES:
        sub = df[df["param_value"] == float(p)]
        if sub.empty:
            # se nel csv non c'è quel parametro, lo segniamo esplicitamente
            out[float(p)] = {
                "weight": None,
                "accuracy": None,
                "spike_count": None,
            }
            continue

        # indice della riga con accuracy massima
        idx = sub["accuracy"].idxmax()
        row = sub.loc[idx]

        out[float(p)] = {
            "weight": float(row["weight"]),
            "accuracy": float(row["accuracy"]),
            "spike_count": float(row["spike_count"]),
        }

    # stampiamo in formato yaml pronto da copiare
    # default_flow_style=False -> blocco leggibile
    print(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    main()
