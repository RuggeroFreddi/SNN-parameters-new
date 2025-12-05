# salva areas_between_curves.csv e areas_under_curves.csv

import os
import pandas as pd
import numpy as np

# ================== PATH GLOBALE ==================
BASE_PATH = "compare/trace_accuracy_reset/"
# ==================================================


def compute_auc(x, y):
    """
    Area under curve con regola del trapezio.
    x e y: array 1D della stessa lunghezza.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    return np.trapz(y_sorted, x_sorted)


def compute_single_file_aucs(df, exp_name, readout_name, metrics_map):
    """
    Calcola area sotto le curve per ogni param_value e metrica (un file, un readout).
    Ritorna una lista di dict, pronti per DataFrame.
    """
    rows = []
    param_values = sorted(df["param_value"].unique())

    for p in param_values:
        df_p = df[df["param_value"] == p].copy()
        df_p = df_p.sort_values("weight")

        x = df_p["weight"].values

        for metric_short, col_name in metrics_map.items():
            if col_name not in df_p.columns:
                continue
            y = df_p[col_name].values
            auc = compute_auc(x, y)

            rows.append({
                "exp": exp_name,        # 'beta' o 'mt'
                "readout": readout_name,  # 'rf' o 'slp'
                "param": p,
                "metric": metric_short, # 'acc', 'f1', 'mcc'
                "auc": auc
            })

    return rows


def choose_param_triplet(unique_values):
    """
    Dato un array/serie di valori unici, ritorna (min, mid, max).
    Se sono esattamente 3 valori, mid è il centrale.
    Se sono di più, mid è quello con indice len//2.
    """
    vals = sorted(unique_values)
    if len(vals) == 0:
        raise ValueError("Nessun valore di param_value trovato.")
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    if len(vals) == 2:
        return vals[0], vals[0], vals[1]

    min_v = vals[0]
    max_v = vals[-1]
    mid_v = vals[len(vals) // 2]
    return min_v, mid_v, max_v


def compute_area_between_curves(df_beta, df_mt, readout_name, metrics_map):
    """
    Calcola l'area compresa tra le curve dei due esperimenti secondo le regole:

    - beta max  vs mt min
    - beta mid  vs mt mid
    - beta min  vs mt max

    Per ogni coppia si calcolano le metriche (acc, f1, mcc).

    Le curve vengono interpolate linearmente su una griglia comune di weight
    (solo nel tratto di overlap tra i domini in x), poi si calcolano:

    - AUC_beta, AUC_mt
    - area_btwn = AUC(|beta - mt|)
    - diff_rel_b  = area_btwn / AUC_beta
    - diff_rel_mt = area_btwn / AUC_mt
    """
    rows = []

    beta_vals = df_beta["param_value"].unique()
    mt_vals   = df_mt["param_value"].unique()

    beta_min, beta_mid, beta_max = choose_param_triplet(beta_vals)
    mt_min,   mt_mid,   mt_max   = choose_param_triplet(mt_vals)

    pairs = [
        ("beta_max_mt_min", beta_max, mt_min),
        ("beta_mid_mt_mid", beta_mid, mt_mid),
        ("beta_min_mt_max", beta_min, mt_max),
    ]

    for pair_name, beta_p, mt_p in pairs:
        df_b = df_beta[df_beta["param_value"] == beta_p].copy()
        df_m = df_mt[df_mt["param_value"] == mt_p].copy()

        if df_b.empty or df_m.empty:
            continue

        # ascisse originali
        x_b = np.asarray(df_b["weight"].values, dtype=float)
        x_m = np.asarray(df_m["weight"].values, dtype=float)

        # dominio di overlap per l'interpolazione
        x_min = max(x_b.min(), x_m.min())
        x_max = min(x_b.max(), x_m.max())
        if x_max <= x_min:
            # nessun overlap in x: salto questa coppia
            continue

        # griglia comune di weight (lineare). Uso una densità basata sulla max len
        n_points = max(len(x_b), len(x_m))
        x_common = np.linspace(x_min, x_max, n_points)

        for metric_short, col_name in metrics_map.items():
            if col_name not in df_b.columns or col_name not in df_m.columns:
                continue

            y_b_orig = np.asarray(df_b[col_name].values, dtype=float)
            y_m_orig = np.asarray(df_m[col_name].values, dtype=float)

            # interpolazione lineare sulle ascisse comuni
            y_beta = np.interp(x_common, x_b, y_b_orig)
            y_mt   = np.interp(x_common, x_m, y_m_orig)

            # AUC delle due curve sul dominio comune
            auc_beta = compute_auc(x_common, y_beta)
            auc_mt   = compute_auc(x_common, y_mt)

            # area tra le curve = AUC della differenza assoluta
            y_diff = np.abs(y_beta - y_mt)
            area_btwn = compute_auc(x_common, y_diff)

            diff_rel_b  = area_btwn / auc_beta if auc_beta != 0 else np.nan
            diff_rel_mt = area_btwn / auc_mt   if auc_mt   != 0 else np.nan

            rows.append({
                "pair": pair_name,          # es: 'beta_max_mt_min'
                "readout": readout_name,    # 'rf' o 'slp'
                "beta_param": beta_p,
                "mt_param": mt_p,
                "metric": metric_short,     # 'acc', 'f1', 'mcc'
                "area_btwn": area_btwn,
                "diff_rel_b": diff_rel_b,   # area/AUC_beta (0–1)
                "diff_rel_mt": diff_rel_mt  # area/AUC_mt (0–1)
            })

    return rows


def main():
    # metriche RF
    metrics_map_rf = {
        "acc": "accuracy_rf",
        "f1": "f1_rf",
        "mcc": "mcc_rf",
    }
    # metriche SLP
    metrics_map_slp = {
        "acc": "accuracy_slp",
        "f1": "f1_slp",
        "mcc": "mcc_slp",
    }

    # path dei file di input
    beta_file = os.path.join(BASE_PATH, "experiment_beta_71.csv")
    mt_file   = os.path.join(BASE_PATH, "experiment_membrane_threshold_71.csv")

    df_beta = pd.read_csv(beta_file)
    df_mt   = pd.read_csv(mt_file)

    # 1) Aree sotto le curve per ciascun file e ciascun readout
    rows_single = []
    # RF
    rows_single.extend(compute_single_file_aucs(df_beta, "beta", "rf", metrics_map_rf))
    rows_single.extend(compute_single_file_aucs(df_mt,   "mt",   "rf", metrics_map_rf))
    # SLP
    rows_single.extend(compute_single_file_aucs(df_beta, "beta", "slp", metrics_map_slp))
    rows_single.extend(compute_single_file_aucs(df_mt,   "mt",   "slp", metrics_map_slp))

    df_single = pd.DataFrame(rows_single)

    # Ordine colonne: exp, readout, param, metric, auc
    col_order_single = ["exp", "readout", "param", "metric", "auc"]
    df_single = df_single[col_order_single]

    out_under = os.path.join(BASE_PATH, "areas_under_curves.csv")
    df_single.to_csv(out_under, index=False)
    print("Scritto:", out_under)

    # 2) Aree tra le curve beta vs membrane_threshold + rapporti (0–1) per RF e SLP
    rows_between = []
    # RF
    rows_between.extend(compute_area_between_curves(df_beta, df_mt, "rf",  metrics_map_rf))
    # SLP
    rows_between.extend(compute_area_between_curves(df_beta, df_mt, "slp", metrics_map_slp))

    df_between = pd.DataFrame(rows_between)

    if not df_between.empty:
        # Ordine colonne: pair, readout, beta_param, mt_param, metric, area_btwn, diff_rel_b, diff_rel_mt
        col_order_between = [
            "pair", "readout", "beta_param", "mt_param",
            "metric", "area_btwn", "diff_rel_b", "diff_rel_mt"
        ]
        df_between = df_between[col_order_between]

    out_between = os.path.join(BASE_PATH, "areas_between_curves.csv")
    df_between.to_csv(out_between, index=False)
    print("Scritto:", out_between)


if __name__ == "__main__":
    main()
