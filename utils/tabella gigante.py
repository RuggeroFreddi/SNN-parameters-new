import os
import pandas as pd

# Cartella che contiene TUTTI e SOLI i file {A}_{B}_{C}_{D}_{E}_{F}.csv
BASE_DIR = "tutte le delta"  # cambia se la cartella ha un altro nome/percorso


def parse_filename(fname: str):
    """
    fname: es.
      'MNIST_statistic_reset_rf_accuracy_71.csv'
      'trajectory_statistic_noreset_rf_current_amplitude_71.csv'
      'trajectory_trace_reset__slp_beta_71.csv'
      'trajectory_statistic_reset_slp_membrane_threshold_71_rf_beta.csv'

    Ritorna:
      Task (A), Output (B), reset (C), Readout (D), Parameter (E), N.steps (F)
    """
    name = os.path.splitext(fname)[0]  # senza .csv
    parts = name.split("_")
    if len(parts) < 6:
        raise ValueError(f"Filename non valido (attesi almeno 6 parti): {fname}")

    task = parts[0]    # A
    output = parts[1]  # B
    reset = parts[2]   # C

    # --- trova Readout ---
    readout_idx = None
    for i in range(3, len(parts)):  # cerco 'rf' o 'slp'
        if parts[i] in ("rf", "slp"):
            readout_idx = i
            break

    if readout_idx is None:
        raise ValueError(f"Non trovato readout ('rf' o 'slp') in filename: {fname}")

    readout = parts[readout_idx]

    # --- definizione pattern parametri ---
    # lavoriamo sui token dopo il readout
    tokens = parts[readout_idx + 1:]

    # patterns: nome_parametro -> lista di token
    param_patterns = {
        "accuracy": ["accuracy"],
        "membrane_threshold": ["membrane", "threshold"],
        "current_amplitude": ["current", "amplitude"],
    }

    parameter = None
    param_start_global = None
    param_end_global = None

    # cerco un match dei pattern nei token
    for j in range(len(tokens)):
        for pname, pattern in param_patterns.items():
            L = len(pattern)
            if j + L <= len(tokens) and tokens[j:j + L] == pattern:
                parameter = pname
                # indice nel vettore completo `parts`
                param_start_global = readout_idx + 1 + j
                param_end_global = param_start_global + L  # esclusivo
                break
        if parameter is not None:
            break

    if parameter is None:
        # fallback: come prima, se non riconosco pattern specifici
        nsteps = parts[-1]
        parameter = "_".join(parts[readout_idx + 1:-1])
        return task, output, reset, readout, parameter, nsteps

    # --- trova N.steps ---
    # cerco il primo token '71' o '101' dopo la fine del parametro
    valid_nsteps = {"71", "101"}
    nsteps = None
    for k in range(param_end_global, len(parts)):
        if parts[k] in valid_nsteps:
            nsteps = parts[k]
            break

    # se non trovata, uso l'ultimo pezzo come fallback
    if nsteps is None:
        nsteps = parts[-1]

    return task, output, reset, readout, parameter, nsteps


def process_file(path, fname):
    """
    Per ogni file genera UNA RIGA PER OGNI param_value.

    Colonne finali per ogni riga:
      Task, Output, reset, Readout, Parameter, N.steps, param_value,
      accuracy_start, accuracy_end, accuracy_diff,
      f1_start,       f1_end,       f1_diff,
      mcc_start,      mcc_end,      mcc_diff
    """
    task, output, reset, readout, parameter, nsteps = parse_filename(fname)
    full_path = os.path.join(path, fname)
    df = pd.read_csv(full_path)

    rows = []

    # mi aspetto colonne: param_value, metric, start_weight, end_weight, width
    for pv in sorted(df["param_value"].unique()):
        df_p = df[df["param_value"] == pv]

        row = {
            "Task": task,
            "Output": output,
            "reset": reset,
            "Readout": readout,
            "Parameter": parameter,
            "N.steps": nsteps,
            "param_value": pv,
            "accuracy_start": float("nan"),
            "accuracy_end":   float("nan"),
            "accuracy_diff":  float("nan"),
            "f1_start":       float("nan"),
            "f1_end":         float("nan"),
            "f1_diff":        float("nan"),
            "mcc_start":      float("nan"),
            "mcc_end":        float("nan"),
            "mcc_diff":       float("nan"),
        }

        for _, r in df_p.iterrows():
            metric = str(r["metric"]).strip().lower()
            s = r["start_weight"]
            e = r["end_weight"]

            if metric == "accuracy":
                row["accuracy_start"] = s
                row["accuracy_end"]   = e
                row["accuracy_diff"]  = e - s
            elif metric == "f1":
                row["f1_start"] = s
                row["f1_end"]   = e
                row["f1_diff"]  = e - s
            elif metric == "mcc":
                row["mcc_start"] = s
                row["mcc_end"]   = e
                row["mcc_diff"]  = e - s

        rows.append(row)

    return rows


def main():
    all_rows = []

    for fname in os.listdir(BASE_DIR):
        if not fname.endswith(".csv"):
            continue
        # evita di inglobare il file di output se rilanci lo script
        if fname == "delta_summary.csv":
            continue

        rows = process_file(BASE_DIR, fname)
        all_rows.extend(rows)

    df_out = pd.DataFrame(
        all_rows,
        columns=[
            "Task", "Output", "reset", "Readout", "Parameter", "N.steps",
            "param_value",
            "accuracy_start", "accuracy_end", "accuracy_diff",
            "f1_start",       "f1_end",       "f1_diff",
            "mcc_start",      "mcc_end",      "mcc_diff",
        ],
    )

    out_path = os.path.join(BASE_DIR, "delta_summary.csv")
    df_out.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
