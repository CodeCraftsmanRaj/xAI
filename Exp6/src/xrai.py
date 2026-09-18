from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from skimage.segmentation import (
    felzenszwalb,
)

from .data import load_image
from .integrated_gradients import (
    integrated_gradients,
)
from .model import InceptionClassifier
from .utils import (
    ensure_dir,
    save_json,
    set_seed,
    get_device,
)


def generate_segments(
    image,
    scale,
    sigma,
    min_size,
):

    segments = felzenszwalb(
        image,
        scale=scale,
        sigma=sigma,
        min_size=min_size,
    )

    return segments


def aggregate_region_scores(
    pixel_attribution,
    segments,
):

    pixel_score = np.mean(
        np.abs(pixel_attribution),
        axis=0,
    )

    region_scores = {}

    for region_id in np.unique(
        segments
    ):

        mask = (
            segments == region_id
        )

        region_scores[int(region_id)] = (
            float(
                pixel_score[mask].sum()
            )
        )

    return region_scores


def greedy_xrai(
    pixel_attribution,
    segments,
):

    pixel_score = np.mean(
        np.abs(pixel_attribution),
        axis=0,
    )

    region_ids = np.unique(
        segments
    )

    region_scores = (
        aggregate_region_scores(
            pixel_attribution,
            segments,
        )
    )

    ordered_regions = sorted(
        region_ids,
        key=lambda region_id:
            region_scores[int(region_id)],
        reverse=True,
    )

    heatmap = np.zeros(
        segments.shape,
        dtype=np.float32,
    )

    selected = np.zeros(
        segments.shape,
        dtype=bool,
    )

    rank = 1

    for region_id in ordered_regions:

        region_mask = (
            segments == region_id
        )

        incremental_score = (
            pixel_score[region_mask]
            .sum()
        )

        if incremental_score <= 0:
            continue

        heatmap[
            region_mask
        ] = rank

        selected |= region_mask

        rank += 1

    if heatmap.max() > 0:

        heatmap = (
            heatmap
            / heatmap.max()
        )

    return (
        heatmap,
        ordered_regions,
        region_scores,
    )


def compute_xrai(
    classifier,
    image,
    config,
):

    image_tensor = (
        classifier.image_to_tensor(
            image
        )
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(classifier.device)
    )

    prediction = classifier.predict(
        image
    )

    target_class = (
        prediction["class_index"]
    )

    baseline = torch.zeros_like(
        image_tensor
    )

    attribution_black = (
        integrated_gradients(
            model=classifier.model,
            image_tensor=image_tensor,
            target_class=target_class,
            baseline=baseline,
            steps=config["xrai"][
                "integration_steps"
            ],
        )
    )

    baseline_white = (
        torch.ones_like(
            image_tensor
        )
    )

    attribution_white = (
        integrated_gradients(
            model=classifier.model,
            image_tensor=image_tensor,
            target_class=target_class,
            baseline=baseline_white,
            steps=config["xrai"][
                "integration_steps"
            ],
        )
    )

    attribution = (
        (
            attribution_black
            + attribution_white
        )
        / 2.0
    )

    attribution = (
        attribution[0]
        .detach()
        .cpu()
        .numpy()
    )

    segments = generate_segments(
        image,
        config["xrai"][
            "segmentation_scale"
        ],
        config["xrai"][
            "segmentation_sigma"
        ],
        config["xrai"][
            "segmentation_min_size"
        ],
    )

    heatmap, ordered_regions, region_scores = (
        greedy_xrai(
            attribution,
            segments,
        )
    )

    return {
        "prediction": prediction,
        "attribution": attribution,
        "segments": segments,
        "heatmap": heatmap,
        "ordered_regions": ordered_regions,
        "region_scores": region_scores,
    }


def create_top_region_image(
    image,
    heatmap,
    top_percent,
):

    threshold = np.percentile(
        heatmap,
        100 - top_percent,
    )

    mask = (
        heatmap >= threshold
    )

    result = np.zeros_like(
        image
    )

    result[mask] = image[mask]

    return result


def save_visualization(
    image,
    heatmap,
    top_region_image,
    prediction,
    output_path,
    dpi,
    cmap,
):

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(
        image
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[1].imshow(
        heatmap,
        cmap=cmap,
    )

    axes[1].set_title(
        "XRAI Heatmap"
    )

    axes[2].imshow(
        top_region_image
    )

    axes[2].set_title(
        "Top-ranked Regions"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{prediction['class_name']} | "
        f"Confidence: "
        f"{prediction['confidence']:.2%}"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def run_xrai_experiment(
    config
):

    set_seed(
        config["experiment"]["seed"]
    )

    device = get_device()

    print("=" * 70)
    print(
        "EXPERIMENT 6 - XRAI"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    output_dir = ensure_dir(
        config["experiment"][
            "output_dir"
        ]
    )

    visualization_dir = ensure_dir(
        output_dir / "visualizations"
    )

    classifier = (
        InceptionClassifier(
            device=device,
            weights_name=
                config["model"][
                    "weights"
                ],
        )
    )

    image_names = config[
        "images"
    ]["names"]

    image_names = image_names[
        :config["images"][
            "max_images"
        ]
    ]

    results = []

    for image_name in image_names:

        print("-" * 70)
        print(
            f"Processing: {image_name}"
        )

        _, image = load_image(
            image_name,
            config["model"][
                "image_size"
            ],
        )

        result = compute_xrai(
            classifier,
            image,
            config,
        )

        prediction = (
            result["prediction"]
        )

        heatmap = (
            result["heatmap"]
        )

        top_region_image = (
            create_top_region_image(
                image,
                heatmap,
                config["xrai"][
                    "top_region_percent"
                ],
            )
        )

        np.save(
            output_dir
            / f"{image_name}_xrai.npy",
            heatmap,
        )

        np.save(
            output_dir
            / f"{image_name}_segments.npy",
            result["segments"],
        )

        visualization_path = (
            visualization_dir
            / f"{image_name}_xrai.png"
        )

        save_visualization(
            image=image,
            heatmap=heatmap,
            top_region_image=
                top_region_image,
            prediction=prediction,
            output_path=
                visualization_path,
            dpi=config[
                "visualization"
            ]["dpi"],
            cmap=config[
                "visualization"
            ]["cmap"],
        )

        results.append(
            {
                "image":
                    image_name,
                "predicted_class":
                    prediction[
                        "class_name"
                    ],
                "class_index":
                    prediction[
                        "class_index"
                    ],
                "confidence":
                    prediction[
                        "confidence"
                    ],
                "top5":
                    prediction[
                        "top5"
                    ],
                "number_of_regions":
                    int(
                        len(
                            np.unique(
                                result[
                                    "segments"
                                ]
                            )
                        )
                    ),
                "heatmap_min":
                    float(
                        heatmap.min()
                    ),
                "heatmap_max":
                    float(
                        heatmap.max()
                    ),
            }
        )

        print(
            f"Prediction: "
            f"{prediction['class_name']}"
        )

        print(
            f"Confidence: "
            f"{prediction['confidence']:.4f}"
        )

        print(
            f"Regions: "
            f"{len(np.unique(result['segments']))}"
        )

        print(
            f"Saved: "
            f"{visualization_path}"
        )

    save_json(
        {
            "experiment":
                config[
                    "experiment"
                ]["name"],
            "device":
                str(device),
            "model":
                config["model"]["name"],
            "results":
                results,
        },
        output_dir
        / "predictions_and_xrai.json",
    )

    print()
    print("=" * 70)
    print(
        "XRAI EXPERIMENT COMPLETE"
    )
    print("=" * 70)

    print(
        f"Results: "
        f"{output_dir.resolve()}"
    )

    print()
    print(
        "Next run:"
    )

    print(
        "uv run interactive_patch.py"
    )