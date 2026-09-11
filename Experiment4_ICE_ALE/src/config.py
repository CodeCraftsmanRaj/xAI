from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path="config.yaml"):
    path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
