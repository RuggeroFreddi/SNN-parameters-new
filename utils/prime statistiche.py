import os
import csv
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 0.80
THRESHOLD_PERC = int(THRESHOLD * 100)

# -------------------------
# Parametri esperimento
# -------------------------
TASK = "MNIST"  # "MNIST", "TRAJECTORY"
OUTPUT_FEATURES = "trace"  # "statistics", "trace"
PARAM_NAME = "current_amplitude"  # possible value: "beta", "membrane_threshold", "current_amplitude"
NUM_WEIGHT_STEPS = 101
DATE = "2025_11_28"

RESULTS_DIR = f"results reset/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{DATE}"
CSV_NAME = os.path.join(RESULTS_DIR, f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv")
# YAML_NAME = os.path.join(RESULTS_DIR, "experiment_metadata.yaml")  # non usato


# -------------------------
# Funzioni di utilità
# -------------------------

def compute_interval(weights, means, stds):
    """
    Usa tre curve:
      v_plus  = mean + std
      v_mean  = mean
      v_minus = mean - std

    Tutte confrontate con la stessa soglia T = THRESHOLD * max(mean).

    Per ciascuna curva calcola:
      - primo weight con valore >= soglia (start_*)
      - ultimo weight con valore >= soglia (end_*), con +inf se l'ultimo punto è ancora >= soglia
      - ampiezza = end_* - start_* (o +inf)

    Ritorna:
      {
        "max": max_mean,
        "threshold": T,
        "start_plus", "end_plus", "width_plus",
        "start_mean", "end_mean", "width_mean",
        "start_minus", "end_minus", "width_minus",
        "width_min", "width_max",
      }
    """
    if not weights or not means or len(weights) != len(means):
        raise ValueError("Liste weight/means non valide")
    if stds is None or len(stds) != len(means):
        raise ValueError("Liste std non valida")

    # ordina per weight
    paired = sorted(zip(weights, means, stds), key=lambda x: x[0])
    weights_sorted = [p[0] for p in paired]
    means_sorted   = [p[1] for p in paired]
    stds_sorted    = [p[2] for p in paired]

    max_mean = max(means_sorted)
    threshold = THRESHOLD * max_mean

    def find_interval(values):
        # values: lista di valori (mean, mean±std) già in ordine di weight
        # ritorna (start_weight, end_weight, width)
        start_w = None
        end_w = None

        for w, v in zip(weights_sorted, values):
            if v >= threshold:
                start_w = w
                break

        if start_w is None:
            # nessun crossing: nessun intervallo; rappresentiamo come None/None/0
            return None, None, 0.0

        for w, v in zip(weights_sorted, values):
            if v >= threshold:
                end_w = w

        # se l'ultimo punto è ancora >= soglia → +inf
        if values[-1] >= threshold:
            end_w_inf = math.inf
            width = math.inf
        else:
            end_w_inf = end_w
            width = end_w_inf - start_w

        return start_w, end_w_inf, width

    # tre curve
    vals_plus  = [m + s for m, s in zip(means_sorted, stds_sorted)]
    vals_mean  = means_sorted
    vals_minus = [m - s for m, s in zip(means_sorted, stds_sorted)]

    start_p, end_p, width_p = find_interval(vals_plus)
    start_m, end_m, width_m = find_interval(vals_mean)
    start_n, end_n, width_n = find_interval(vals_minus)

    # calcolo width_min/max tra le tre ampiezze disponibili
    widths = []
    for w in (width_p, width_m, width_n):
        if w is not None:
            widths.append(w)

    if not widths:
        width_min = 0.0
        width_max = 0.0
    else:
        # se c'è almeno un inf, max diventa inf automaticamente
        width_min = min(w for w in widths if not math.isinf(w)) if any(
            not math.isinf(w) for w in widths
        ) else math.inf
        width_max = max(widths)

    return {
        "max": max_mean,
        "threshold": threshold,
        "start_plus": start_p,
        "end_plus": end_p,
        "width_plus": width_p,
        "start_mean": start_m,
        "end_mean": end_m,
        "width_mean": width_m,
        "start_minus": start_n,
        "end_minus": end_n,
        "width_minus": width_n,
        "width_min": width_min,
        "width_max": width_max,
    }


def widths_for_plot(widths):
    """
    Prepara le ampiezze per il plotting:
    - sostituisce inf con un valore grande finito
    - evita zeri
    """
    finite = [w for w in widths if not math.isinf(w) and w > 0]
    if finite:
        max_w = max(finite)
        eps = max_w / 1000.0
    else:
        max_w = 1.0
        eps = 1e-6

    out = []
    for w in widths:
        if math.isinf(w):
            out.append(max_w * 10.0)  # valore molto grande ma finito
        elif w <= 0:
            out.append(eps)           # niente zeri
        else:
            out.append(w)
    return out


def fit_and_plot_line(x, y, color, label_suffix=""):
    """
    Fit lineare y = a x + b (minimi quadrati) e plottala.
    Ignora eventuali NaN/inf.
    """
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return

    a, b = np.polyfit(x_arr[mask], y_arr[mask], 1)
    x_line = np.linspace(x_arr[mask].min(), x_arr[mask].max(), 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, color=color, linestyle="-", linewidth=1.5,
             label=label_suffix if label_suffix else None)


# -------------------------
# Main
# -------------------------

def main():
    numeric_fields = [
        "param_value",
        "weight",
        "accuracy_rf", "std_accuracy_rf",
        "accuracy_slp", "std_accuracy_slp",
        "f1_rf", "std_f1_rf",
        "f1_slp", "std_f1_slp",
        "mcc_rf", "std_mcc_rf",
        "mcc_slp", "std_mcc_slp",
        "spike_count",
    ]

    rows = []
    with open(CSV_NAME, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in numeric_fields and v is not None and v != "":
                    parsed[k] = float(v)
                else:
                    parsed[k] = v
            rows.append(parsed)

    if not rows:
        print("CSV vuoto o non leggibile.")
        return

    # Raggruppa per param_value: ogni gruppo è una curva
    groups = defaultdict(lambda: {
        "weight": [],
        "accuracy_rf": [], "std_accuracy_rf": [],
        "accuracy_slp": [], "std_accuracy_slp": [],
        "f1_rf": [], "std_f1_rf": [],
        "f1_slp": [], "std_f1_slp": [],
        "mcc_rf": [], "std_mcc_rf": [],
        "mcc_slp": [], "std_mcc_slp": [],
    })

    for r in rows:
        p = r["param_value"]
        groups[p]["weight"].append(r["weight"])

        groups[p]["accuracy_rf"].append(r["accuracy_rf"])
        groups[p]["std_accuracy_rf"].append(r["std_accuracy_rf"])

        groups[p]["accuracy_slp"].append(r["accuracy_slp"])
        groups[p]["std_accuracy_slp"].append(r["std_accuracy_slp"])

        groups[p]["f1_rf"].append(r["f1_rf"])
        groups[p]["std_f1_rf"].append(r["std_f1_rf"])

        groups[p]["f1_slp"].append(r["f1_slp"])
        groups[p]["std_f1_slp"].append(r["std_f1_slp"])

        groups[p]["mcc_rf"].append(r["mcc_rf"])
        groups[p]["std_mcc_rf"].append(r["std_mcc_rf"])

        groups[p]["mcc_slp"].append(r["mcc_slp"])
        groups[p]["std_mcc_slp"].append(r["std_mcc_slp"])

    # ordina i parametri
    param_values = sorted(groups.keys())

    # per i grafici: ampiezze min/mean/max per ogni metrica e parametro
    rf_widths = {
        "accuracy": {"min": [], "mean": [], "max": []},
        "f1":       {"min": [], "mean": [], "max": []},
        "mcc":      {"min": [], "mean": [], "max": []},
    }
    slp_widths = {
        "accuracy": {"min": [], "mean": [], "max": []},
        "f1":       {"min": [], "mean": [], "max": []},
        "mcc":      {"min": [], "mean": [], "max": []},
    }

    # per salvare tutte le statistiche sugli intervalli
    rf_results = {
        "accuracy": {},
        "f1": {},
        "mcc": {},
    }
    slp_results = {
        "accuracy": {},
        "f1": {},
        "mcc": {},
    }

    # -------------------------
    # RF
    # -------------------------
    print("=== RANDOM FOREST (rf) ===")
    rf_metrics = [
        ("accuracy", "accuracy_rf", "std_accuracy_rf"),
        ("f1",       "f1_rf",       "std_f1_rf"),
        ("mcc",      "mcc_rf",      "std_mcc_rf"),
    ]

    for metric_label, colname, stdcol in rf_metrics:
        print(f"--- {metric_label} ---")
        for p in param_values:
            g = groups[p]
            res = compute_interval(g["weight"], g[colname], g[stdcol])
            rf_results[metric_label][p] = res

            width_min  = res["width_min"]
            width_mean = res["width_mean"]
            width_max  = res["width_max"]

            rf_widths[metric_label]["min"].append(width_min)
            rf_widths[metric_label]["mean"].append(width_mean)
            rf_widths[metric_label]["max"].append(width_max)

            end_mean_str = "+inf" if math.isinf(res["end_mean"]) else f"{res['end_mean']}"
            width_min_str  = "+inf" if math.isinf(width_min)  else f"{width_min}"
            width_mean_str = "+inf" if math.isinf(width_mean) else f"{width_mean}"
            width_max_str  = "+inf" if math.isinf(width_max)  else f"{width_max}"

            print(
                f"param = {p}: "
                f"max = {res['max']:.6f}, "
                f"soglia{THRESHOLD_PERC} = {res['threshold']:.6f}, "
                f"[start-, start, start+] = "
                f"[{res['start_minus']}, {res['start_mean']}, {res['start_plus']}], "
                f"[end-, end, end+] = "
                f"[{res['end_minus']}, {res['end_mean']}, {res['end_plus']}], "
                f"width_min = {width_min_str}, "
                f"width_mean = {width_mean_str}, "
                f"width_max = {width_max_str}"
            )
        print()

    # -------------------------
    # SLP
    # -------------------------
    print("=== SLP (slp) ===")
    slp_metrics = [
        ("accuracy", "accuracy_slp", "std_accuracy_slp"),
        ("f1",       "f1_slp",       "std_f1_slp"),
        ("mcc",      "mcc_slp",      "std_mcc_slp"),
    ]

    for metric_label, colname, stdcol in slp_metrics:
        print(f"--- {metric_label} ---")
        for p in param_values:
            g = groups[p]
            res = compute_interval(g["weight"], g[colname], g[stdcol])
            slp_results[metric_label][p] = res

            width_min  = res["width_min"]
            width_mean = res["width_mean"]
            width_max  = res["width_max"]

            slp_widths[metric_label]["min"].append(width_min)
            slp_widths[metric_label]["mean"].append(width_mean)
            slp_widths[metric_label]["max"].append(width_max)

            end_mean_str = "+inf" if math.isinf(res["end_mean"]) else f"{res['end_mean']}"
            width_min_str  = "+inf" if math.isinf(width_min)  else f"{width_min}"
            width_mean_str = "+inf" if math.isinf(width_mean) else f"{width_mean}"
            width_max_str  = "+inf" if math.isinf(width_max)  else f"{width_max}"

            print(
                f"param = {p}: "
                f"max = {res['max']:.6f}, "
                f"soglia{THRESHOLD_PERC} = {res['threshold']:.6f}, "
                f"[start-, start, start+] = "
                f"[{res['start_minus']}, {res['start_mean']}, {res['start_plus']}], "
                f"[end-, end, end+] = "
                f"[{res['end_minus']}, {res['end_mean']}, {res['end_plus']}], "
                f"width_min = {width_min_str}, "
                f"width_mean = {width_mean_str}, "
                f"width_max = {width_max_str}"
            )
        print()

    # -------------------------
    # Salva statistiche intervalli in CSV
    # -------------------------
    rf_stats_path = os.path.join(RESULTS_DIR, "interval_stats_rf.csv")
    slp_stats_path = os.path.join(RESULTS_DIR, "interval_stats_slp.csv")

    rf_header = [
        "param_value", "metric",
        "max_mean", "threshold",
        "start_plus", "end_plus", "width_plus",
        "start_mean", "end_mean", "width_mean",
        "start_minus", "end_minus", "width_minus",
        "width_min", "width_max",
    ]

    with open(rf_stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rf_header)
        for metric_label in ["accuracy", "f1", "mcc"]:
            for p in param_values:
                res = rf_results[metric_label][p]
                writer.writerow([
                    p, metric_label,
                    res["max"], res["threshold"],
                    res["start_plus"], res["end_plus"], res["width_plus"],
                    res["start_mean"], res["end_mean"], res["width_mean"],
                    res["start_minus"], res["end_minus"], res["width_minus"],
                    res["width_min"], res["width_max"],
                ])

    with open(slp_stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rf_header)  # stessa struttura
        for metric_label in ["accuracy", "f1", "mcc"]:
            for p in param_values:
                res = slp_results[metric_label][p]
                writer.writerow([
                    p, metric_label,
                    res["max"], res["threshold"],
                    res["start_plus"], res["end_plus"], res["width_plus"],
                    res["start_mean"], res["end_mean"], res["width_mean"],
                    res["start_minus"], res["end_minus"], res["width_minus"],
                    res["width_min"], res["width_max"],
                ])

    # -------------------------
    # Grafici ampiezze intervalli + rette di regressione
    # -------------------------

    # piccolo offset orizzontale per separare i 3 punti per ogni parametro
    if len(param_values) > 1:
        base_span = max(param_values) - min(param_values)
        dx = base_span / (len(param_values) * 10.0)
    else:
        dx = 0.1  # caso limite, un offset fisso

    # colori fissi per coerenza linea/punti
    color_acc = "tab:blue"
    color_f1  = "tab:orange"
    color_mcc = "tab:green"

    # -------------------------
    # RF
    # -------------------------
    plt.figure()

    # mean/min/max per il plot
    rf_acc_mean = widths_for_plot(rf_widths["accuracy"]["mean"])
    rf_acc_min  = widths_for_plot(rf_widths["accuracy"]["min"])
    rf_acc_max  = widths_for_plot(rf_widths["accuracy"]["max"])

    rf_f1_mean  = widths_for_plot(rf_widths["f1"]["mean"])
    rf_f1_min   = widths_for_plot(rf_widths["f1"]["min"])
    rf_f1_max   = widths_for_plot(rf_widths["f1"]["max"])

    rf_mcc_mean = widths_for_plot(rf_widths["mcc"]["mean"])
    rf_mcc_min  = widths_for_plot(rf_widths["mcc"]["min"])
    rf_mcc_max  = widths_for_plot(rf_widths["mcc"]["max"])

    x_rf_acc  = [p - dx for p in param_values]
    x_rf_f1   = [p       for p in param_values]
    x_rf_mcc  = [p + dx  for p in param_values]

    # error bar: mean con barra da min a max
    def err_from_min_max(mean_vals, min_vals, max_vals):
        lower = [m - mn for m, mn in zip(mean_vals, min_vals)]
        upper = [mx - m for m, mx in zip(mean_vals, max_vals)]
        # correzione nel caso di numerica strana
        lower = [max(l, 0.0) for l in lower]
        upper = [max(u, 0.0) for u in upper]
        return [lower, upper]

    acc_err = err_from_min_max(rf_acc_mean, rf_acc_min, rf_acc_max)
    f1_err  = err_from_min_max(rf_f1_mean,  rf_f1_min,  rf_f1_max)
    mcc_err = err_from_min_max(rf_mcc_mean, rf_mcc_min, rf_mcc_max)

    plt.errorbar(x_rf_acc, rf_acc_mean, yerr=acc_err,
                 fmt="o", color=color_acc, label="accuracy")
    plt.errorbar(x_rf_f1, rf_f1_mean, yerr=f1_err,
                 fmt="s", color=color_f1, label="f1")
    plt.errorbar(x_rf_mcc, rf_mcc_mean, yerr=mcc_err,
                 fmt="^", color=color_mcc, label="mcc")

    # rette di regressione sulle ampiezze medie
    fit_and_plot_line(x_rf_acc,  rf_acc_mean,  color=color_acc)
    fit_and_plot_line(x_rf_f1,   rf_f1_mean,   color=color_f1)
    fit_and_plot_line(x_rf_mcc,  rf_mcc_mean,  color=color_mcc)

    # retta complessiva (tutti i punti medi) in nero
    x_rf_all = x_rf_acc + x_rf_f1 + x_rf_mcc
    y_rf_all = rf_acc_mean + rf_f1_mean + rf_mcc_mean
    fit_and_plot_line(x_rf_all, y_rf_all, color="black", label_suffix="regressione totale")

    plt.xlabel("param_value")
    plt.ylabel("ampiezza intervallo (width)")
    plt.title(f"Ampiezza intervalli RF (soglia {THRESHOLD_PERC}%)")
    plt.legend()
    plt.grid(True)

    rf_img_path = os.path.join(RESULTS_DIR, "interval_widths_rf.png")
    plt.savefig(rf_img_path, dpi=300)

    # -------------------------
    # SLP
    # -------------------------
    plt.figure()

    slp_acc_mean = widths_for_plot(slp_widths["accuracy"]["mean"])
    slp_acc_min  = widths_for_plot(slp_widths["accuracy"]["min"])
    slp_acc_max  = widths_for_plot(slp_widths["accuracy"]["max"])

    slp_f1_mean  = widths_for_plot(slp_widths["f1"]["mean"])
    slp_f1_min   = widths_for_plot(slp_widths["f1"]["min"])
    slp_f1_max   = widths_for_plot(slp_widths["f1"]["max"])

    slp_mcc_mean = widths_for_plot(slp_widths["mcc"]["mean"])
    slp_mcc_min  = widths_for_plot(slp_widths["mcc"]["min"])
    slp_mcc_max  = widths_for_plot(slp_widths["mcc"]["max"])

    x_slp_acc = [p - dx for p in param_values]
    x_slp_f1  = [p       for p in param_values]
    x_slp_mcc = [p + dx  for p in param_values]

    acc_err_s = err_from_min_max(slp_acc_mean, slp_acc_min, slp_acc_max)
    f1_err_s  = err_from_min_max(slp_f1_mean,  slp_f1_min,  slp_f1_max)
    mcc_err_s = err_from_min_max(slp_mcc_mean, slp_mcc_min, slp_mcc_max)

    plt.errorbar(x_slp_acc, slp_acc_mean, yerr=acc_err_s,
                 fmt="o", color=color_acc, label="accuracy")
    plt.errorbar(x_slp_f1,  slp_f1_mean,  yerr=f1_err_s,
                 fmt="s", color=color_f1, label="f1")
    plt.errorbar(x_slp_mcc, slp_mcc_mean, yerr=mcc_err_s,
                 fmt="^", color=color_mcc, label="mcc")

    fit_and_plot_line(x_slp_acc, slp_acc_mean,  color=color_acc)
    fit_and_plot_line(x_slp_f1,  slp_f1_mean,   color=color_f1)
    fit_and_plot_line(x_slp_mcc, slp_mcc_mean,  color=color_mcc)

    x_slp_all = x_slp_acc + x_slp_f1 + x_slp_mcc
    y_slp_all = slp_acc_mean + slp_f1_mean + slp_mcc_mean
    fit_and_plot_line(x_slp_all, y_slp_all, color="black", label_suffix="regressione totale")

    plt.xlabel("param_value")
    plt.ylabel("ampiezza intervallo (width)")
    plt.title(f"Ampiezza intervalli SLP (soglia {THRESHOLD_PERC}%)")
    plt.legend()
    plt.grid(True)

    slp_img_path = os.path.join(RESULTS_DIR, "interval_widths_slp.png")
    plt.savefig(slp_img_path, dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
