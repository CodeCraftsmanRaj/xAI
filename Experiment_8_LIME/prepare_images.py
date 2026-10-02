from pathlib import Path

import numpy as np
from PIL import Image

from skimage import data
from skimage.transform import resize


# ============================================================
# Configuration
# ============================================================

OUT = Path("data/images")
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (512, 512)


# ============================================================
# Background generation
# ============================================================

def create_background(kind: str, height: int, width: int) -> np.ndarray:
    """
    Create three clearly different synthetic backgrounds.
    """

    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]

    if kind == "indoor":
        top = np.array([235, 225, 210], dtype=np.float32)
        bottom = np.array([145, 115, 85], dtype=np.float32)

        bg = top * (1 - y[:, :, None]) + bottom * y[:, :, None]
        bg = np.repeat(bg, width, axis=1)

        pattern = 8 * np.sin(2 * np.pi * x * 5)
        bg += pattern[:, :, None]

    elif kind == "garden":
        top = np.array([120, 185, 105], dtype=np.float32)
        bottom = np.array([35, 100, 40], dtype=np.float32)

        bg = top * (1 - y[:, :, None]) + bottom * y[:, :, None]
        bg = np.repeat(bg, width, axis=1)

        pattern = 8 * np.sin(2 * np.pi * x * 14)
        bg += pattern[:, :, None]

    elif kind == "sky":
        top = np.array([65, 145, 225], dtype=np.float32)
        bottom = np.array([215, 235, 250], dtype=np.float32)

        bg = top * (1 - y[:, :, None]) + bottom * y[:, :, None]
        bg = np.repeat(bg, width, axis=1)

        pattern = 4 * np.sin(2 * np.pi * x * 3)
        bg += pattern[:, :, None]

    else:
        raise ValueError(f"Unknown background type: {kind}")

    return np.clip(bg, 0, 255).astype(np.uint8)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 8 - SAME OBJECT / DIFFERENT BACKGROUNDS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load the standard skimage cat image.
    # --------------------------------------------------------

    print("\nLoading source cat...")

    source = data.chelsea()

    source = resize(
        source,
        IMAGE_SIZE,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.uint8)

    height, width = IMAGE_SIZE

    # --------------------------------------------------------
    # Crop the cat.
    #
    # Chelsea contains one clear main cat. We use a fixed
    # central crop instead of automatic segmentation so that
    # the exact same cat pixels are reused in every image.
    # --------------------------------------------------------

    print("Preparing the same cat for all three images...")

    crop_y1 = int(0.08 * height)
    crop_y2 = int(0.98 * height)

    crop_x1 = int(0.05 * width)
    crop_x2 = int(0.95 * width)

    cat = source[crop_y1:crop_y2, crop_x1:crop_x2]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We use the SAME cat image in all three outputs.
    #
    # The cat is placed on a transparent background using
    # its original rectangular crop. This is intentional:
    # the object remains exactly identical between images.
    # --------------------------------------------------------

    cat_image = Image.fromarray(cat).convert("RGB")

    # Resize the cat consistently.
    cat_width = int(width * 0.72)
    cat_height = int(cat_image.height * (cat_width / cat_image.width))

    cat_image = cat_image.resize(
        (cat_width, cat_height),
        Image.Resampling.LANCZOS,
    )

    # --------------------------------------------------------
    # Create three different backgrounds.
    # --------------------------------------------------------

    backgrounds = {
        "cat_indoor.jpg": "indoor",
        "cat_garden.jpg": "garden",
        "cat_sky.jpg": "sky",
    }

    # Slightly different positions create different contexts.
    positions = {
        "cat_indoor.jpg": (0.50, 0.53),
        "cat_garden.jpg": (0.48, 0.57),
        "cat_sky.jpg": (0.52, 0.52),
    }

    # --------------------------------------------------------
    # Composite the same cat crop onto each background.
    # --------------------------------------------------------

    for filename, background_type in backgrounds.items():

        print(f"\nCreating {filename}...")

        background = create_background(
            background_type,
            height,
            width,
        )

        canvas = Image.fromarray(background).convert("RGB")

        px = positions[filename][0]
        py = positions[filename][1]

        x = int(px * width - cat_width / 2)
        y = int(py * height - cat_height / 2)

        canvas.paste(
            cat_image,
            (x, y),
        )

        output_path = OUT / filename

        canvas.save(
            output_path,
            quality=95,
        )

        print(f"Saved: {output_path}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Three images created successfully.")
    print("=" * 70)

    print("\nGenerated images:")

    for path in sorted(OUT.glob("cat_*.jpg")):
        print(f"  - {path}")

    print("\nObject: SAME cat")
    print("Backgrounds: indoor / garden / sky")
    print("\nExperiment 8 image preparation complete.")


if __name__ == "__main__":
    main()