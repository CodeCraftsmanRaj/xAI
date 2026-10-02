from pathlib import Path
from skimage import data
from PIL import Image

OUT = Path("data/images")
OUT.mkdir(parents=True, exist_ok=True)

examples = {
    "cat.jpg": data.chelsea(),
    "coffee.jpg": data.coffee(),
    "astronaut.jpg": data.astronaut(),
}

for name, arr in examples.items():
    Image.fromarray(arr).save(OUT / name, quality=95)
    print(f"Saved {OUT/name}")
