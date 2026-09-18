import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import RectangleSelector

from .data import load_image
from .model import InceptionClassifier
from .utils import (
    ensure_dir,
    get_device,
)


def run_patch_experiment(
    config
):

    image_name = config[
        "patch_search"
    ]["image_name"]

    image_size = config[
        "model"
    ]["image_size"]

    _, image = load_image(
        image_name,
        image_size,
    )

    device = get_device()

    model = InceptionClassifier(
        device=device
    )

    prediction = model.predict(
        image
    )

    target_class = (
        prediction["class_index"]
    )

    original_confidence = (
        prediction["confidence"]
    )

    print("=" * 70)
    print(
        "TASK 2 - INTERACTIVE PATCH"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Image: {image_name}"
    )

    print(
        f"Target class: "
        f"{prediction['class_name']}"
    )

    print(
        f"Original confidence: "
        f"{original_confidence:.4f}"
    )

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    plt.subplots_adjust(
        bottom=0.15
    )

    axis.imshow(
        image
    )

    axis.set_title(
        f"{image_name}\n"
        f"{prediction['class_name']} | "
        f"{original_confidence:.2%}\n"
        "Drag a rectangle across the image"
    )

    axis.axis("off")

    confidence_text = (
        figure.text(
            0.5,
            0.04,
            f"Confidence: "
            f"{original_confidence:.2%}",
            ha="center",
        )
    )

    def evaluate_patch(
        click,
        release,
    ):

        if (
            click.xdata is None
            or click.ydata is None
            or release.xdata is None
            or release.ydata is None
        ):
            return

        x1 = int(
            round(click.xdata)
        )

        y1 = int(
            round(click.ydata)
        )

        x2 = int(
            round(release.xdata)
        )

        y2 = int(
            round(release.ydata)
        )

        x_min, x_max = sorted(
            (
                max(0, x1),
                min(image_size, x2),
            )
        )

        y_min, y_max = sorted(
            (
                max(0, y1),
                min(image_size, y2),
            )
        )

        if (
            x_max <= x_min
            or y_max <= y_min
        ):
            return

        patched = image.copy()

        patched[
            y_min:y_max,
            x_min:x_max
        ] = 127

        new_confidence = (
            model.score(
                patched,
                target_class,
            )
        )

        confidence_drop = (
            original_confidence
            - new_confidence
        )

        width = (
            x_max - x_min
        )

        height = (
            y_max - y_min
        )

        area = width * height

        axis.clear()

        axis.imshow(
            patched
        )

        axis.set_title(
            f"Patch: "
            f"{width} × {height}px\n"
            f"Confidence: "
            f"{new_confidence:.2%}\n"
            f"Drop: "
            f"{confidence_drop:.2%}"
        )

        axis.axis("off")

        confidence_text.set_text(
            f"Confidence: "
            f"{new_confidence:.2%} | "
            f"Drop: "
            f"{confidence_drop:.2%} | "
            f"Area: "
            f"{area}px²"
        )

        figure.canvas.draw_idle()

        print()
        print(
            f"Patch: "
            f"{width} × {height}px"
        )

        print(
            f"Area: {area}px²"
        )

        print(
            f"New confidence: "
            f"{new_confidence:.4f}"
        )

        print(
            f"Confidence drop: "
            f"{confidence_drop:.4f}"
        )

    RectangleSelector(
        axis,
        evaluate_patch,
        useblit=False,
        button=[1],
        minspanx=3,
        minspany=3,
        spancoords="pixels",
        interactive=True,
    )

    print()
    print(
        "Drag patches over different regions."
    )

    print(
        "Try to find a small patch "
        "that causes a large confidence drop."
    )

    print(
        "Close the window when finished."
    )

    plt.show()

    # ---------------------------------------------------------
    # Systematic search
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SYSTEMATIC PATCH SEARCH"
    )
    print("=" * 70)

    rows = []

    patch_sizes = config[
        "patch_search"
    ]["patch_sizes"]

    stride = config[
        "patch_search"
    ]["stride"]

    for patch_size in patch_sizes:

        print(
            f"Scanning "
            f"{patch_size} × "
            f"{patch_size}"
        )

        for y in range(
            0,
            image_size - patch_size + 1,
            stride,
        ):

            for x in range(
                0,
                image_size - patch_size + 1,
                stride,
            ):

                patched = image.copy()

                patched[
                    y:y + patch_size,
                    x:x + patch_size
                ] = 127

                confidence = (
                    model.score(
                        patched,
                        target_class,
                    )
                )

                drop = (
                    original_confidence
                    - confidence
                )

                rows.append(
                    {
                        "x": x,
                        "y": y,
                        "patch_width":
                            patch_size,
                        "patch_height":
                            patch_size,
                        "area":
                            patch_size ** 2,
                        "confidence":
                            confidence,
                        "confidence_drop":
                            drop,
                    }
                )

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        by=[
            "confidence_drop",
            "area",
        ],
        ascending=[
            False,
            True,
        ],
    )

    output_dir = ensure_dir(
        config["experiment"][
            "output_dir"
        ]
    )

    output_file = (
        output_dir
        / "patch_search_results.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "Largest confidence-drop patch:"
    )

    print(
        results.iloc[0].to_string()
    )

    print()
    print(
        f"Saved: "
        f"{output_file.resolve()}"
    )