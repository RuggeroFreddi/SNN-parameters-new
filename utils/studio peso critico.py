import os
import pandas as pd
import numpy as np

# Cartella e file di input/output
BASE_DIR = "tutte le delta"
INPUT_FILE = os.path.join(BASE_DIR, "delta_summary.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "delta_summary_wcrit.csv")

# Costanti
T_REF = 3.0

TASK_PARAMS = {
    "MNIST": {
        "N": 1000,
        "I_ext": 26.203294166666666,
    },
    "trajectory": {
        "N": 2000,
        "I_ext": 38.94898571428571,
    },
}

# Valori di riferimento
BETA_REF = 0.2
CA_REF = 2.0
MT_REF = 2.0


def compute_wcrit(row):
    """
    Calcola w_critico per una riga di delta_summary.
    """
    task = row["Task"]
    if task not in TASK_PARAMS:
        return np.nan

    N = TASK_PARAMS[task]["N"]
    I_ext = TASK_PARAMS[task]["I_ext"]

    # parametri iniziali = valori di riferimento
    beta = BETA_REF
    ca = CA_REF
    mt = MT_REF

    # sovrascrivo il parametro che è stato variato nell'esperimento
    param = str(row["Parameter"]).strip()
    val = float(row["param_value"])

    if param == "beta":
        beta = val
    elif param == "current_amplitude":
        ca = val
    elif param == "membrane_threshold":
        mt = val

    # I = I_ext * current_amplitude / N
    I = I_ext * ca / N

    # w_critico = (membrane_threshold - 2 I T_ref) / (beta N)
    denom = beta * N
    if denom == 0:
        return np.nan

    return (mt - 2.0 * I * T_REF) / denom


def main():
    df = pd.read_csv(INPUT_FILE)

    # 1) w_critico per ogni riga
    df["w_crit"] = df.apply(compute_wcrit, axis=1)

    # 2) flag: w_critico compreso tra gli intervalli
    # uso min/max per sicurezza nel caso start > end
    def in_interval(w, a, b):
        if pd.isna(w) or pd.isna(a) or pd.isna(b):
            return False
        lo = min(a, b)
        hi = max(a, b)
        return lo <= w <= hi

    df["acc_in"] = df.apply(
        lambda r: in_interval(r["w_crit"], r["accuracy_start"], r["accuracy_end"]),
        axis=1,
    )
    df["f1_in"] = df.apply(
        lambda r: in_interval(r["w_crit"], r["f1_start"], r["f1_end"]),
        axis=1,
    )
    df["mcc_in"] = df.apply(
        lambda r: in_interval(r["w_crit"], r["mcc_start"], r["mcc_end"]),
        axis=1,
    )

    # 3) flag: (start - end)/2 < w_critico
    def half_diff_less_than_wcrit(w, start, end):
        if pd.isna(w) or pd.isna(start) or pd.isna(end):
            return False
        return (start - end) / 2.0 < w

    df["acc_halfdiff_lt_wcrit"] = df.apply(
        lambda r: half_diff_less_than_wcrit(r["w_crit"], r["accuracy_start"], r["accuracy_end"]),
        axis=1,
    )
    df["f1_halfdiff_lt_wcrit"] = df.apply(
        lambda r: half_diff_less_than_wcrit(r["w_crit"], r["f1_start"], r["f1_end"]),
        axis=1,
    )
    df["mcc_halfdiff_lt_wcrit"] = df.apply(
        lambda r: half_diff_less_than_wcrit(r["w_crit"], r["mcc_start"], r["mcc_end"]),
        axis=1,
    )

    # 4) colonne in output
    df_out = df[
        [
            "Task",
            "Output",
            "reset",
            "Readout",
            "Parameter",
            "N.steps",
            "param_value",
            "w_crit",
            "acc_in",
            "f1_in",
            "mcc_in",
            "acc_halfdiff_lt_wcrit",
            "f1_halfdiff_lt_wcrit",
            "mcc_halfdiff_lt_wcrit",
        ]
    ]

    df_out.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
