import numpy as np

from skimage import data
from skimage.transform import resize

from .utils import normalize_uint8


IMAGE_LOADERS = {
    "astronaut": data.astronaut,
    "coffee": data.coffee,
    "chelsea": data.chelsea,
    "rocket": data.rocket,
}


def load_image(name, size=299):

    if name not in IMAGE_LOADERS:
        raise ValueError(
            f"Unknown image '{name}'. "
            f"Available: {list(IMAGE_LOADERS.keys())}"
        )

    original = normalize_uint8(
        IMAGE_LOADERS[name]()
    )

    resized = resize(
        original,
        (size, size),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)

    return original, resized