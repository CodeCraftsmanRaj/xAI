import json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from src.dataset import load_data
from src.interpret import compute_ice, compute_ale
from src.plots import save_ice_plot, save_ale_plot

def main():
    cfg = load_config()
    model = joblib.load(cfg["paths"]["model"])

    _, X_test, _, _, _, _ = load_data(
        random_state=cfg["experiment"]["random_state"],
        test_size=cfg["data"]["test_size"]
    )

    feature = cfg["interpretability"]["feature"]
    if feature not in X_test.columns:
        raise ValueError(f"Feature '{feature}' not found in dataset.")

    grid, curves, chosen = compute_ice(
        model,
        X_test,
        feature,
        n_points=cfg["interpretability"]["ice_points"],
        n_samples=cfg["interpretability"]["ice_samples"],
        random_state=cfg["experiment"]["random_state"]
    )
    save_ice_plot(grid, curves, feature, cfg["paths"]["ice_plot"])

    # ALE is computed on the full test distribution to study local changes.
    centers, ale, counts, edges = compute_ale(
        model, X_test, feature,
        n_bins=cfg["interpretability"]["ale_bins"]
    )
    save_ale_plot(centers, ale, feature, cfg["paths"]["ale_plot"])

    summary = {
        "feature": feature,
        "ice": {
            "grid_min": float(grid.min()),
            "grid_max": float(grid.max()),
            "grid_points": len(grid),
            "samples_plotted": len(chosen),
            "mean_probability_min": float(curves.mean(axis=0).min()),
            "mean_probability_max": float(curves.mean(axis=0).max())
        },
        "ale": {
            "bins_used": len(centers),
            "ale_min": float(ale.min()),
            "ale_max": float(ale.max()),
            "bin_counts": counts.tolist(),
            "bin_edges": edges.tolist()
        }
    }
    with open("outputs/interpretation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Interpretability analysis complete.")
    print(f"ICE plot: {cfg['paths']['ice_plot']}")
    print(f"ALE plot: {cfg['paths']['ale_plot']}")

if __name__ == "__main__":
    main()
