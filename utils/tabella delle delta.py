# produce: tutte le delta/delta_summary_paramdiff.csv

import os
import pandas as pd

# Cartella e file di input
BASE_DIR = "tutte le delta"
INPUT_FILE = os.path.join(BASE_DIR, "delta_summary.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "delta_summary_paramdiff.csv")


def pick_min_mid_max(values):
    """
    Dato un array/lista di valori, restituisce (min, mid, max)
    dove mid è quello centrale dopo ordinamento.
    """
    vals = sorted(values)
    if len(vals) < 3:
        return None, None, None
    v_min = vals[0]
    v_max = vals[-1]
    v_mid = vals[len(vals) // 2]
    return v_min, v_mid, v_max


def have_same_sign(diffs):
    """
    Ritorna True se tutte le differenze hanno lo stesso segno
    (tutte >=0 oppure tutte <=0, ignorando eventuali NaN),
    altrimenti False.
    """
    clean = [d for d in diffs if pd.notna(d)]
    if not clean:
        return False

    has_pos = any(d > 0 for d in clean)
    has_neg = any(d < 0 for d in clean)

    # stesso segno se non ci sono sia positivi sia negativi
    return not (has_pos and has_neg)


def same_sign_pair(d1, d2):
    """
    Ritorna True se tra d1 e d2 NON c'è cambio di segno
    (entrambi >=0 oppure entrambi <=0, ignorando 0 e NaN),
    altrimenti False.
    """
    clean = [d for d in (d1, d2) if pd.notna(d) and d != 0]
    if len(clean) < 2:
        # con meno di due valori "utili" consideriamo il segno "coerente"
        return True

    has_pos = any(d > 0 for d in clean)
    has_neg = any(d < 0 for d in clean)

    return not (has_pos and has_neg)


def main():
    df = pd.read_csv(INPUT_FILE)

    group_cols = ["Task", "Output", "reset", "Readout", "Parameter", "N.steps"]

    rows_out = []

    for keys, g in df.groupby(group_cols):
        param_vals = g["param_value"].unique()
        p_min, p_mid, p_max = pick_min_mid_max(param_vals)
        if p_min is None:
            continue

        g_min = g[g["param_value"] == p_min].iloc[0]
        g_mid = g[g["param_value"] == p_mid].iloc[0]
        g_max = g[g["param_value"] == p_max].iloc[0]

        # differenze: medio - minimo, massimo - medio
        acc_mid_min = g_mid["accuracy_diff"] - g_min["accuracy_diff"]
        acc_max_mid = g_max["accuracy_diff"] - g_mid["accuracy_diff"]

        f1_mid_min = g_mid["f1_diff"] - g_min["f1_diff"]
        f1_max_mid = g_max["f1_diff"] - g_mid["f1_diff"]

        mcc_mid_min = g_mid["mcc_diff"] - g_min["mcc_diff"]
        mcc_max_mid = g_max["mcc_diff"] - g_mid["mcc_diff"]

        # per-metrica: hanno lo stesso segno le due differenze (mid-min) e (max-mid)?
        acc_same_sign = same_sign_pair(acc_mid_min, acc_max_mid)
        f1_same_sign  = same_sign_pair(f1_mid_min,  f1_max_mid)
        mcc_same_sign = same_sign_pair(mcc_mid_min, mcc_max_mid)

        # colonna globale: tutte le differenze (tutte le metriche) hanno lo stesso segno?
        same_sign_all = have_same_sign([
            acc_mid_min, acc_max_mid,
            f1_mid_min,  f1_max_mid,
            mcc_mid_min, mcc_max_mid,
        ])

        row = {
            "Task":   keys[0],
            "Output": keys[1],
            "reset":  keys[2],
            "Readout": keys[3],
            "Parameter": keys[4],
            "N.steps": keys[5],
            "acc_diff_mid_min": acc_mid_min,
            "acc_diff_max_mid": acc_max_mid,
            "f1_diff_mid_min":  f1_mid_min,
            "f1_diff_max_mid":  f1_max_mid,
            "mcc_diff_mid_min": mcc_mid_min,
            "mcc_diff_max_mid": mcc_max_mid,
            "acc_same_sign": acc_same_sign,
            "f1_same_sign":  f1_same_sign,
            "mcc_same_sign": mcc_same_sign,
            "same_sign":     same_sign_all,
        }

        rows_out.append(row)

    df_out = pd.DataFrame(
        rows_out,
        columns=[
            "Task", "Output", "reset", "Readout", "Parameter", "N.steps",
            "acc_diff_mid_min", "acc_diff_max_mid",
            "f1_diff_mid_min",  "f1_diff_max_mid",
            "mcc_diff_mid_min", "mcc_diff_max_mid",
            "acc_same_sign", "f1_same_sign", "mcc_same_sign",
            "same_sign",
        ],
    )

    df_out.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
