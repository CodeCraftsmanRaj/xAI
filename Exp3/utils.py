import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def create_directories(config):
    model_path = config["paths"]["model"]
    metrics_path = config["paths"]["metrics"]
    pdp_data_path = config["paths"]["pdp_data"]
    pdp_plot_path = config["paths"]["pdp_plot"]
    importance_plot_path = config["paths"]["importance_plot"]

    directories = [
        os.path.dirname(model_path),
        os.path.dirname(metrics_path),
        os.path.dirname(pdp_data_path),
        os.path.dirname(pdp_plot_path),
        os.path.dirname(importance_plot_path),
    ]

    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True)


def save_metrics(metrics, path):
    with open(path, "w") as file:
        json.dump(metrics, file, indent=4)


def plot_feature_importance(model, feature_names, path):
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))

    importance.plot(kind="barh")

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()