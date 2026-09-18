from pathlib import Path

import yaml

from src.patch_search import run_patch_experiment


def main():
    with Path("config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    run_patch_experiment(config)


if __name__ == "__main__":
    main()