"""
generate_report.py

Batch-runs prediction + Integrated Gradients explanation (across all four
baselines) for every image in a folder, and saves:
  - one comparison figure per image (original + overlay per baseline)
  - a predictions_summary.csv with top-1 / top-5 results

This is meant to make the "analyze whether the model focuses on relevant
regions" and "investigate the effect of different baselines" parts of the
report easy — just point it at ResNet18_Test_Images and look at the output.

Usage (run from the backend/ folder, or after `pip install -r requirements.txt`):

    python ../scripts/generate_report.py \
        --input_dir "/home/raj_99/Projects/Sem7_Labs/xAI/Exp5_Project/ResNet18_Test_Images/ResNet18_Test_Images" \
        --output_dir ../report_outputs
"""
import argparse
import base64
import csv
import io
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from model_utils import load_image, predict, explain  # noqa: E402

BASELINES = ["black", "white", "noise", "blur"]


def b64_to_pil(b64str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64str)))


def process_image(path: str, output_dir: str, csv_writer):
    with open(path, "rb") as f:
        img_bytes = f.read()
    img = load_image(img_bytes)
    preds, _ = predict(img, topk=5)
    top1 = preds[0]

    fig, axes = plt.subplots(1, len(BASELINES) + 1, figsize=(4 * (len(BASELINES) + 1), 4.2))

    orig_shown = False
    for i, baseline in enumerate(BASELINES):
        result = explain(img, top1["class_idx"], baseline_type=baseline, n_steps=50)
        if not orig_shown:
            axes[0].imshow(b64_to_pil(result["original_resized"]))
            axes[0].set_title("Original (224x224)")
            axes[0].axis("off")
            orig_shown = True
        axes[i + 1].imshow(b64_to_pil(result["overlay"]))
        axes[i + 1].set_title(f"Baseline: {baseline}\nconv_delta={result['convergence_delta']:.3f}")
        axes[i + 1].axis("off")

    fname = os.path.splitext(os.path.basename(path))[0]
    fig.suptitle(f"{fname}  —  predicted: {top1['class_name']} ({top1['confidence'] * 100:.1f}%)")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{fname}_explanation.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    csv_writer.writerow([
        fname,
        top1["class_name"],
        f"{top1['confidence'] * 100:.2f}%",
        ", ".join(f"{p['class_name']}({p['confidence'] * 100:.1f}%)" for p in preds),
    ])
    print(f"Processed {fname} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder to scan recursively for images")
    parser.add_argument("--output_dir", default="report_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "predictions_summary.csv")

    image_paths = []
    for root, _, files in os.walk(args.input_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, fname))

    if not image_paths:
        print(f"No images found under {args.input_dir}")
        return

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "top1_class", "top1_confidence", "top5"])
        for path in sorted(image_paths):
            process_image(path, args.output_dir, writer)

    print(f"\nDone. {len(image_paths)} images processed.")
    print(f"Summary CSV: {csv_path}")
    print(f"Figures saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
