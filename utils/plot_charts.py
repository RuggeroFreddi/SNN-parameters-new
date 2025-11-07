import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

TASK = "MNIST" # possible values: "MNIST"
OUTPUT_FEATURES = "trace" # possible values: "statistics", "trace"
PARAM_NAME = "current_amplitude" # possible value: "beta", "membrane_threshold", "current_amplitude"
NUM_WEIGHT_STEPS = 51  # how many mean_weight values have been testes
DATE = "2025_11_06"

RESULTS_DIR = f"results/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{DATE}"  # cambia la data
CSV_NAME = os.path.join(RESULTS_DIR, f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv")
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
        parameter_df = results_df[results_df["param_value"] == value].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        plt.plot(
            parameter_df["weight"],
            parameter_df["accuracy"],
            marker="o",
            label=f"{PARAM_NAME}={value}",
        )

        max_accuracy = parameter_df["accuracy"].max()
        threshold = accuracy_threshold * max_accuracy

        eligible = parameter_df[parameter_df["accuracy"] >= threshold]
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
    plt.title(f"Accuracy vs weight for different {PARAM_NAME} values")
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
        parameter_df = results_df[results_df["param_value"] == value].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        plt.plot(
            parameter_df["weight"],
            parameter_df["spike_count"],
            marker="o",
            label=f"{PARAM_NAME}={value}",
        )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean spike count")
    plt.title(f"Spike count vs weight for different {PARAM_NAME} values")
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