from pathlib import Path

import yaml

from src.xrai import run_xrai_experiment


def main():
    with Path("config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    run_xrai_experiment(config)


if __name__ == "__main__":
    main()