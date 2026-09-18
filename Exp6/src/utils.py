from pathlib import Path
import json
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def normalize_uint8(image):
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    if image.ndim == 2:
        image = np.stack(
            [image] * 3,
            axis=-1,
        )

    if image.shape[-1] == 4:
        image = image[..., :3]

    return image