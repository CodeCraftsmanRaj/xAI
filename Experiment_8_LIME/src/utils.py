from pathlib import Path
import json
import random
import numpy as np
import torch
import yaml

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(cfg):
    requested = cfg["experiment"]["device"]
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

def ensure_dirs(cfg):
    root = Path(cfg["experiment"]["output_dir"])
    for sub in ["attributions", "disagreement", "tables"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root

def save_json(obj, path):
    def convert(x):
        if isinstance(x, (np.integer, np.int64, np.int32)): return int(x)
        if isinstance(x, (np.floating, np.float64, np.float32)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        return str(x)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)
