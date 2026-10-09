from pathlib import Path
import json
import logging
import random

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.experiment import run_experiment


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    seed = int(config.get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_name = config["model_name"]
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Using device: %s", device)

    id2label = {int(k): str(v).upper() for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    logger.info("Model labels: %s", id2label)

    results = run_experiment(
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=device,
        id2label=id2label,
        label2id=label2id,
        logger=logger,
    )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("All results saved to %s", output_dir.resolve())
    print("\nModel label mapping:", id2label)
    for item in results["sentences"]:
        print(f"\nSentence: {item['sentence']}")
        print("Probabilities:", item["original_probabilities"])
        print("Largest positive LIME weight:", item["largest_positive_word"])
        print("Largest negative LIME weight:", item["largest_negative_word"])
    print(f"\nSee {output_dir.resolve()} for CSV, JSON, HTML, and log outputs.")


if __name__ == "__main__":
    main()
