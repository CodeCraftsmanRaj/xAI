import json
from pathlib import Path
import joblib

from src.config import load_config
from src.dataset import load_data
from src.model import build_model
from src.metrics import evaluate

def main():
    cfg = load_config()
    X_train, X_test, y_train, y_test, features, target_names = load_data(
        random_state=cfg["experiment"]["random_state"],
        test_size=cfg["data"]["test_size"]
    )

    model = build_model(cfg)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    joblib.dump(model, cfg["paths"]["model"])

    with open(cfg["paths"]["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(cfg["paths"]["feature_info"], "w", encoding="utf-8") as f:
        json.dump({
            "features": features,
            "target_names": list(target_names),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }, f, indent=2)

    print("Training complete.")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
