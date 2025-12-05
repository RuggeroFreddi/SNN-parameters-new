import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "compare/trace_accuracy_reset/"

# =========================
#  Funzione smoothing
# =========================
def smooth_series(series, weights):
    """
    Smoothing 1D tramite convoluzione:
    - series: pd.Series o array 1D
    - weights: lista di pesi (es. [1,1,1,1,1])
    Ritorna un np.array smussato, stessa lunghezza della serie.
    """
    series = np.asarray(series, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(series) < len(w):
        # troppo corta per applicare la finestra completa
        return series

    w = w / w.sum()
    smoothed = np.convolve(series, w, mode="same")
    return smoothed

# =========================
#  Normalizzazione 0–1
# =========================
def normalize_0_1(series):
    """
    Normalizza una serie 1D in [0,1].
    Ritorna un np.array.
    """
    arr = np.asarray(series, dtype=float)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax == vmin:
        # tutti uguali: restituisco zeri (oppure tutti 0.5 se preferisci)
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)

# =========================
#  Funzione selezione k-esimo param_value
# =========================
def select_by_kth_param(df, k, order="asc"):
    """
    Seleziona tutte le righe di df corrispondenti al k-esimo valore
    (distinto) di param_value.
    - k: indice 0-based (0 = minimo o massimo a seconda di 'order')
    - order: "asc" per ordinare crescente (minimo, secondo, ...),
             "desc" per decrescente (massimo, secondo massimo, ...)
    Ritorna un DataFrame (eventualmente vuoto se non esiste quel k).
    """
    if df.empty:
        return df

    unique_vals = np.sort(df["param_value"].unique())
    if order == "desc":
        unique_vals = unique_vals[::-1]

    if k >= len(unique_vals):
        # non esiste quel k-esimo valore
        return df.iloc[0:0].copy()  # dataframe vuoto con stesse colonne

    target_val = unique_vals[k]
    return df[df["param_value"] == target_val].copy()

# =========================
#  Lettura file
# =========================
base_dir = BASE_DIR

beta_file = os.path.join(base_dir, "experiment_beta_71.csv")
current_file = os.path.join(base_dir, "experiment_current_amplitude_71.csv")
membrane_file = os.path.join(base_dir, "experiment_membrane_threshold_71.csv")

df_beta = pd.read_csv(beta_file)
df_curr = pd.read_csv(current_file)
df_mem  = pd.read_csv(membrane_file)

# Ordina sempre per weight dopo la selezione
def prepare_sel(df, k, order):
    df_sel = select_by_kth_param(df, k=k, order=order)
    if not df_sel.empty:
        df_sel = df_sel.sort_values("weight")
    return df_sel

kernel_weights = [1, 1, 1, 1, 1]

# =========================
#  SET 1
#  - beta: min param_value
#  - current_amplitude: min param_value
#  - membrane_threshold: max param_value
# =========================

df_beta_1 = prepare_sel(df_beta, k=0, order="asc")
df_curr_1 = prepare_sel(df_curr, k=0, order="asc")
df_mem_1  = prepare_sel(df_mem,  k=0, order="desc")  # max

# --- RF, smoothing + normalizzazione ---
if not df_beta_1.empty:
    sm = smooth_series(df_beta_1["accuracy_rf"], kernel_weights)
    df_beta_1["accuracy_rf_smooth"] = normalize_0_1(sm)

if not df_curr_1.empty:
    sm = smooth_series(df_curr_1["accuracy_rf"], kernel_weights)
    df_curr_1["accuracy_rf_smooth"] = normalize_0_1(sm)

if not df_mem_1.empty:
    sm = smooth_series(df_mem_1["accuracy_rf"], kernel_weights)
    df_mem_1["accuracy_rf_smooth"] = normalize_0_1(sm)

plt.figure()
if not df_beta_1.empty:
    plt.plot(df_beta_1["weight"], df_beta_1["accuracy_rf_smooth"],
             marker="o", label="beta (min param)")
if not df_curr_1.empty:
    plt.plot(df_curr_1["weight"], df_curr_1["accuracy_rf_smooth"],
             marker="s", label="current_amp (min param)")
if not df_mem_1.empty:
    plt.plot(df_mem_1["weight"], df_mem_1["accuracy_rf_smooth"],
             marker="^", label="membrane (max param)")

plt.xlabel("weight")
plt.ylabel("accuracy_rf (smoothed, norm 0–1)")
plt.title("RF - Set 1: min beta, min current_amp, max membrane")
plt.legend()
plt.grid(True)
plt.tight_layout()
out1 = os.path.join(base_dir, "accuracy_rf_vs_weight_smoothed_set1.png")
plt.savefig(out1, dpi=300)
plt.close()
print(f"Grafico RF 1 salvato in: {out1}")

# =========================
#  SET 2
#  - beta: secondo valore di param_value
#  - current_amplitude: secondo valore
#  - membrane_threshold: secondo valore
# =========================

df_beta_2 = prepare_sel(df_beta, k=1, order="asc")
df_curr_2 = prepare_sel(df_curr, k=1, order="asc")
df_mem_2  = prepare_sel(df_mem,  k=1, order="asc")

# --- RF, smoothing + normalizzazione ---
if not df_beta_2.empty:
    sm = smooth_series(df_beta_2["accuracy_rf"], kernel_weights)
    df_beta_2["accuracy_rf_smooth"] = normalize_0_1(sm)

if not df_curr_2.empty:
    sm = smooth_series(df_curr_2["accuracy_rf"], kernel_weights)
    df_curr_2["accuracy_rf_smooth"] = normalize_0_1(sm)

if not df_mem_2.empty:
    sm = smooth_series(df_mem_2["accuracy_rf"], kernel_weights)
    df_mem_2["accuracy_rf_smooth"] = normalize_0_1(sm)

plt.figure()
if not df_beta_2.empty:
    plt.plot(df_beta_2["weight"], df_beta_2["accuracy_rf_smooth"],
             marker="o", label="beta (2° param)")
if not df_curr_2.empty:
    plt.plot(df_curr_2["weight"], df_curr_2["accuracy_rf_smooth"],
             marker="s", label="current_amp (2° param)")
if not df_mem_2.empty:
    plt.plot(df_mem_2["weight"], df_mem_2["accuracy_rf_smooth"],
             marker="^", label="membrane (2° param)")

plt.xlabel("weight")
plt.ylabel("accuracy_rf (smoothed, norm 0–1)")
plt.title("RF - Set 2: secondo valore di param_value")
plt.legend()
plt.grid(True)
plt.tight_layout()
out2 = os.path.join(base_dir, "accuracy_rf_vs_weight_smoothed_set2.png")
plt.savefig(out2, dpi=300)
plt.close()
print(f"Grafico RF 2 salvato in: {out2}")

# =========================
#  SET 3
#  - beta: max param_value
#  - current_amplitude: max param_value
#  - membrane_threshold: min param_value
# =========================

df_beta_3 = prepare_sel(df_beta, k=0, order="desc")  # max
df_curr_3 = prepare_sel(df_curr, k=0, order="desc")  # max
df_mem_3  = prepare_sel(df_mem,  k=0, order="asc")   # min

# --- RF, smoothing + normalizzazione ---
if not df_beta_3.empty:
    sm = smooth_series(df_beta_3["accuracy_rf"], kernel_weights)
    df_beta_3["accuracy_rf_smooth"] = normalize_0_1(sm)

if not df_curr_3.empty:
    sm = smooth_series(df_curr_3["accuracy_rf"], kernel_weights)
    df_curr_3["accuracy_rf_smooth"] = normalize_0_1(sm)

if not df_mem_3.empty:
    sm = smooth_series(df_mem_3["accuracy_rf"], kernel_weights)
    df_mem_3["accuracy_rf_smooth"] = normalize_0_1(sm)

plt.figure()
if not df_beta_3.empty:
    plt.plot(df_beta_3["weight"], df_beta_3["accuracy_rf_smooth"],
             marker="o", label="beta (max param)")
if not df_curr_3.empty:
    plt.plot(df_curr_3["weight"], df_curr_3["accuracy_rf_smooth"],
             marker="s", label="current_amp (max param)")
if not df_mem_3.empty:
    plt.plot(df_mem_3["weight"], df_mem_3["accuracy_rf_smooth"],
             marker="^", label="membrane (min param)")

plt.xlabel("weight")
plt.ylabel("accuracy_rf (smoothed, norm 0–1)")
plt.title("RF - Set 3: max beta, max current_amp, min membrane")
plt.legend()
plt.grid(True)
plt.tight_layout()
out3 = os.path.join(base_dir, "accuracy_rf_vs_weight_smoothed_set3.png")
plt.savefig(out3, dpi=300)
plt.close()
print(f"Grafico RF 3 salvato in: {out3}")

# ============================================================
#                GRAFICI SLP (altri 3 grafici)
# ============================================================

# SET 1 - SLP
if not df_beta_1.empty and "accuracy_slp" in df_beta_1.columns:
    sm = smooth_series(df_beta_1["accuracy_slp"], kernel_weights)
    df_beta_1["accuracy_slp_smooth"] = normalize_0_1(sm)
if not df_curr_1.empty and "accuracy_slp" in df_curr_1.columns:
    sm = smooth_series(df_curr_1["accuracy_slp"], kernel_weights)
    df_curr_1["accuracy_slp_smooth"] = normalize_0_1(sm)
if not df_mem_1.empty and "accuracy_slp" in df_mem_1.columns:
    sm = smooth_series(df_mem_1["accuracy_slp"], kernel_weights)
    df_mem_1["accuracy_slp_smooth"] = normalize_0_1(sm)

plt.figure()
if "accuracy_slp_smooth" in df_beta_1:
    plt.plot(df_beta_1["weight"], df_beta_1["accuracy_slp_smooth"],
             marker="o", label="beta (min param)")
if "accuracy_slp_smooth" in df_curr_1:
    plt.plot(df_curr_1["weight"], df_curr_1["accuracy_slp_smooth"],
             marker="s", label="current_amp (min param)")
if "accuracy_slp_smooth" in df_mem_1:
    plt.plot(df_mem_1["weight"], df_mem_1["accuracy_slp_smooth"],
             marker="^", label="membrane (max param)")

plt.xlabel("weight")
plt.ylabel("accuracy_slp (smoothed, norm 0–1)")
plt.title("SLP - Set 1: min beta, min current_amp, max membrane")
plt.legend()
plt.grid(True)
plt.tight_layout()
out1_slp = os.path.join(base_dir, "accuracy_slp_vs_weight_smoothed_set1.png")
plt.savefig(out1_slp, dpi=300)
plt.close()
print(f"Grafico SLP 1 salvato in: {out1_slp}")

# SET 2 - SLP
if not df_beta_2.empty and "accuracy_slp" in df_beta_2.columns:
    sm = smooth_series(df_beta_2["accuracy_slp"], kernel_weights)
    df_beta_2["accuracy_slp_smooth"] = normalize_0_1(sm)
if not df_curr_2.empty and "accuracy_slp" in df_curr_2.columns:
    sm = smooth_series(df_curr_2["accuracy_slp"], kernel_weights)
    df_curr_2["accuracy_slp_smooth"] = normalize_0_1(sm)
if not df_mem_2.empty and "accuracy_slp" in df_mem_2.columns:
    sm = smooth_series(df_mem_2["accuracy_slp"], kernel_weights)
    df_mem_2["accuracy_slp_smooth"] = normalize_0_1(sm)

plt.figure()
if "accuracy_slp_smooth" in df_beta_2:
    plt.plot(df_beta_2["weight"], df_beta_2["accuracy_slp_smooth"],
             marker="o", label="beta (2° param)")
if "accuracy_slp_smooth" in df_curr_2:
    plt.plot(df_curr_2["weight"], df_curr_2["accuracy_slp_smooth"],
             marker="s", label="current_amp (2° param)")
if "accuracy_slp_smooth" in df_mem_2:
    plt.plot(df_mem_2["weight"], df_mem_2["accuracy_slp_smooth"],
             marker="^", label="membrane (2° param)")

plt.xlabel("weight")
plt.ylabel("accuracy_slp (smoothed, norm 0–1)")
plt.title("SLP - Set 2: secondo valore di param_value")
plt.legend()
plt.grid(True)
plt.tight_layout()
out2_slp = os.path.join(base_dir, "accuracy_slp_vs_weight_smoothed_set2.png")
plt.savefig(out2_slp, dpi=300)
plt.close()
print(f"Grafico SLP 2 salvato in: {out2_slp}")

# SET 3 - SLP
if not df_beta_3.empty and "accuracy_slp" in df_beta_3.columns:
    sm = smooth_series(df_beta_3["accuracy_slp"], kernel_weights)
    df_beta_3["accuracy_slp_smooth"] = normalize_0_1(sm)
if not df_curr_3.empty and "accuracy_slp" in df_curr_3.columns:
    sm = smooth_series(df_curr_3["accuracy_slp"], kernel_weights)
    df_curr_3["accuracy_slp_smooth"] = normalize_0_1(sm)
if not df_mem_3.empty and "accuracy_slp" in df_mem_3.columns:
    sm = smooth_series(df_mem_3["accuracy_slp"], kernel_weights)
    df_mem_3["accuracy_slp_smooth"] = normalize_0_1(sm)

plt.figure()
if "accuracy_slp_smooth" in df_beta_3:
    plt.plot(df_beta_3["weight"], df_beta_3["accuracy_slp_smooth"],
             marker="o", label="beta (max param)")
if "accuracy_slp_smooth" in df_curr_3:
    plt.plot(df_curr_3["weight"], df_curr_3["accuracy_slp_smooth"],
             marker="s", label="current_amp (max param)")
if "accuracy_slp_smooth" in df_mem_3:
    plt.plot(df_mem_3["weight"], df_mem_3["accuracy_slp_smooth"],
             marker="^", label="membrane (min param)")

plt.xlabel("weight")
plt.ylabel("accuracy_slp (smoothed, norm 0–1)")
plt.title("SLP - Set 3: max beta, max current_amp, min membrane")
plt.legend()
plt.grid(True)
plt.tight_layout()
out3_slp = os.path.join(base_dir, "accuracy_slp_vs_weight_smoothed_set3.png")
plt.savefig(out3_slp, dpi=300)
plt.close()
print(f"Grafico SLP 3 salvato in: {out3_slp}")
