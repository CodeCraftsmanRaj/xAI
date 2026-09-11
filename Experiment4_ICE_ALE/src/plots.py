from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_ice_plot(grid, curves, feature, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for curve in curves:
        plt.plot(grid, curve, alpha=0.15, linewidth=1)
    plt.plot(grid, np.mean(curves, axis=0), linewidth=3, label="Mean ICE")
    plt.xlabel(feature)
    plt.ylabel("Predicted probability of malignant class")
    plt.title(f"Individual Conditional Expectation (ICE): {feature}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def save_ale_plot(centers, ale, feature, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.plot(centers, ale, marker="o", linewidth=2)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel(feature)
    plt.ylabel("ALE effect on predicted probability")
    plt.title(f"Accumulated Local Effects (ALE): {feature}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
