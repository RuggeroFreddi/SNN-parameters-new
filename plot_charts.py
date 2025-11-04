import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

# 👇 metti qui la cartella dove hai salvato i risultati
RESULTS_DIR = "results_2025_11_03"  # cambia la data
CSV_NAME = os.path.join(RESULTS_DIR, "experiment_beta_51.csv")
YAML_NAME = os.path.join(RESULTS_DIR, "experiment_metadata.yaml")


def load_metadata(yaml_path: str):
    with open(yaml_path, "r") as file:
        metadata = yaml.safe_load(file)
    return metadata


def plot_accuracy(results_df: pd.DataFrame, metadata: dict):
    param_values = metadata["tested_parameter"]["values"]
    accuracy_threshold = metadata["global_parameters"]["accuracy_threshold"]

    plt.figure()

    for value in param_values:
        beta_df = results_df[results_df["param_value"] == value].copy()
        beta_df = beta_df.sort_values(by="weight")

        plt.plot(
            beta_df["weight"],
            beta_df["accuracy"],
            marker="o",
            label=f"beta={value}",
        )

        max_accuracy = beta_df["accuracy"].max()
        threshold = accuracy_threshold * max_accuracy

        eligible = beta_df[beta_df["accuracy"] >= threshold]
        if not eligible.empty:
            w1 = eligible["weight"].min()
            w2 = eligible["weight"].max()
            plt.hlines(
                y=threshold,
                xmin=w1,
                xmax=w2,
                colors="black",
                linestyles="dashed",
            )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean CV accuracy")
    plt.title("Accuracy vs weight for different beta values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "plot_accuracy.png")
    plt.savefig(plot_path)
    print(f"saved {plot_path}")


def plot_spike_count(results_df: pd.DataFrame, metadata: dict):
    param_values = metadata["tested_parameter"]["values"]

    plt.figure()

    for value in param_values:
        beta_df = results_df[results_df["param_value"] == value].copy()
        beta_df = beta_df.sort_values(by="weight")

        plt.plot(
            beta_df["weight"],
            beta_df["spike_count"],
            marker="o",
            label=f"beta={value}",
        )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean spike count")
    plt.title("Spike count vs weight for different beta values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "plot_spike_count.png")
    plt.savefig(plot_path)
    print(f"saved {plot_path}")


def main():
    results_df = pd.read_csv(CSV_NAME)
    metadata = load_metadata(YAML_NAME)

    plot_accuracy(results_df, metadata)
    plot_spike_count(results_df, metadata)
    plt.show()


if __name__ == "__main__":
    main()
