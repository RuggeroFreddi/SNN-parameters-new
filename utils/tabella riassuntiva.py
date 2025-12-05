import os
import numpy as np
import pandas as pd

# Cartella e file di input
BASE_DIR = "tutte le delta"

# Qui usiamo il file dettagliato, perché per calcolare
# (val_mid - val_min) / max(val_mid, val_min)
# servono i valori val_mid e val_min, non solo le differenze.
INPUT_FILE_DETAILED = os.path.join(BASE_DIR, "delta_summary.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "delta_summary_paramdiff_stats.csv")


def norm_delta(a, b):
    """
    (a - b) / max(|a|, |b|), con gestione del caso denom = 0.
    """
    denom = max(abs(a), abs(b))
    if denom == 0:
        return 0.0
    return (a - b) / denom


def same_sign_pair(d1, d2):
    """
    True se d1 e d2 hanno lo stesso segno (entrambi >=0 o entrambi <=0),
    ignorando NaN.
    """
    clean = [d for d in (d1, d2) if pd.notna(d)]
    if len(clean) < 2:
        return True

    has_pos = any(d > 0 for d in clean)
    has_neg = any(d < 0 for d in clean)

    return not (has_pos and has_neg)


def have_same_sign_all(diffs):
    """
    True se tutte le differenze in lista hanno lo stesso segno
    (tutte >=0 oppure tutte <=0, ignorando NaN).
    """
    clean = [d for d in diffs if pd.notna(d)]
    if not clean:
        return False

    has_pos = any(d > 0 for d in clean)
    has_neg = any(d < 0 for d in clean)

    return not (has_pos and has_neg)


def main():
    # Legge il file dettagliato con i valori per param_value
    df = pd.read_csv(INPUT_FILE_DETAILED)

    # Colonne di raggruppamento "locali" (per min/medio/max)
    group_cols_full = ["Task", "Output", "reset", "Readout", "Parameter", "N.steps"]

    rows_local = []

    for keys, g in df.groupby(group_cols_full):
        # ordina per param_value e prende min / medio / max
        g_sorted = g.sort_values("param_value").reset_index(drop=True)
        if len(g_sorted) < 3:
            continue

        g_min = g_sorted.iloc[0]
        g_mid = g_sorted.iloc[len(g_sorted) // 2]
        g_max = g_sorted.iloc[-1]

        # valori di accuracy_diff, f1_diff, mcc_diff per min/mid/max
        acc_min = g_min["accuracy_diff"]
        acc_mid = g_mid["accuracy_diff"]
        acc_max = g_max["accuracy_diff"]

        f1_min = g_min["f1_diff"]
        f1_mid = g_mid["f1_diff"]
        f1_max = g_max["f1_diff"]

        mcc_min = g_min["mcc_diff"]
        mcc_mid = g_mid["mcc_diff"]
        mcc_max = g_max["mcc_diff"]

        # differenze NORMALIZZATE:
        # (mid - min) / max(|mid|, |min|)
        # (max - mid) / max(|max|, |mid|)
        acc_rel_mid_min = norm_delta(acc_mid, acc_min)
        acc_rel_max_mid = norm_delta(acc_max, acc_mid)

        f1_rel_mid_min = norm_delta(f1_mid, f1_min)
        f1_rel_max_mid = norm_delta(f1_max, f1_mid)

        mcc_rel_mid_min = norm_delta(mcc_mid, mcc_min)
        mcc_rel_max_mid = norm_delta(mcc_max, mcc_mid)

        # same_sign per metrica (tra le due differenze normalizzate)
        acc_same_sign = same_sign_pair(acc_rel_mid_min, acc_rel_max_mid)
        f1_same_sign = same_sign_pair(f1_rel_mid_min, f1_rel_max_mid)
        mcc_same_sign = same_sign_pair(mcc_rel_mid_min, mcc_rel_max_mid)

        # same_sign globale su tutte le differenze normalizzate
        same_sign_all = have_same_sign_all([
            acc_rel_mid_min, acc_rel_max_mid,
            f1_rel_mid_min,  f1_rel_max_mid,
            mcc_rel_mid_min, mcc_rel_max_mid,
        ])

        row = {
            "Task":   keys[0],
            "Output": keys[1],
            "reset":  keys[2],
            "Readout": keys[3],
            "Parameter": keys[4],
            "N.steps": keys[5],
            "acc_rel_mid_min": acc_rel_mid_min,
            "acc_rel_max_mid": acc_rel_max_mid,
            "f1_rel_mid_min":  f1_rel_mid_min,
            "f1_rel_max_mid":  f1_rel_max_mid,
            "mcc_rel_mid_min": mcc_rel_mid_min,
            "mcc_rel_max_mid": mcc_rel_max_mid,
            "acc_same_sign": acc_same_sign,
            "f1_same_sign":  f1_same_sign,
            "mcc_same_sign": mcc_same_sign,
            "same_sign":     same_sign_all,
        }

        rows_local.append(row)

    # DataFrame con le differenze NORMALIZZATE per ogni combinazione
    df_local = pd.DataFrame(rows_local)

    # Adesso calcolo media e std al variare di Task, Output, reset,
    # Readout, N.steps (cioè raggruppando per Parameter)
    group_col = ["Parameter"]

    num_cols = [
        "acc_rel_mid_min", "acc_rel_max_mid",
        "f1_rel_mid_min",  "f1_rel_max_mid",
        "mcc_rel_mid_min", "mcc_rel_max_mid",
    ]

    bool_cols = ["acc_same_sign", "f1_same_sign", "mcc_same_sign", "same_sign"]

    g = df_local.groupby(group_col)

    # mean/std sulle differenze normalizzate
    stats = g[num_cols].agg(["mean", "std"])
    stats.columns = [
        f"{col}_{stat}"
        for col, stat in stats.columns.to_flat_index()
    ]
    stats = stats.reset_index()

    # per le colonne booleane: True se tutte le righe del parametro sono True
    bool_stats = g[bool_cols].agg(lambda x: x.astype(bool).all()).reset_index()

    df_out = pd.merge(stats, bool_stats, on=group_col, how="inner")

    df_out.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
