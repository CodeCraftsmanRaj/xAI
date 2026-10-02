from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from skimage.segmentation import slic

def make_segmentation_fn(cfg):
    seg = cfg["lime"]["segmentation"]
    def segmentation_fn(image):
        return slic(
            image,
            n_segments=seg.get("n_segments", 80),
            compactness=seg["compactness"],
            sigma=seg["sigma"],
            start_label=seg["start_label"],
            channel_axis=-1,
        )
    return segmentation_fn

def explain_image(image, predict_fn, cfg, n_segments, num_samples, seed):
    lime_cfg = cfg["lime"]
    segmentation_fn = make_segmentation_fn(cfg)
    # LIME's segmentation function receives only the image, so n_segments is
    # temporarily attached to the segmentation config for this explanation.
    segmentation_fn = lambda arr: slic(
        arr,
        n_segments=n_segments,
        compactness=lime_cfg["segmentation"]["compactness"],
        sigma=lime_cfg["segmentation"]["sigma"],
        start_label=lime_cfg["segmentation"]["start_label"],
        channel_axis=-1,
    )
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    explanation = explainer.explain_instance(
        np.asarray(image),
        predict_fn,
        top_labels=cfg["experiment"]["top_k_classes"],
        hide_color=lime_cfg["hide_color"],
        num_samples=num_samples,
        segmentation_fn=segmentation_fn,
    )
    return explanation

def get_target_label(explanation):
    return int(explanation.top_labels[0])

def positive_mask(explanation, label, top_segments, image_shape):
    pairs = explanation.local_exp[label]
    selected = [seg_id for seg_id, weight in pairs if weight > 0][:top_segments]
    segments = explanation.segments
    mask = np.isin(segments, selected)
    return mask, selected

def render_explanation(image, explanation, label, categories, out_path, num_features):
    from skimage.segmentation import mark_boundaries
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )
    vis = mark_boundaries(temp / 255.0 if temp.max() > 1 else temp, mask)
    plt.figure(figsize=(8, 6))
    plt.imshow(vis)
    plt.axis("off")
    plt.title(f"LIME positive attribution — {categories[label]}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

def render_side_by_side(image, exp_a, exp_b, label, categories, path):
    from skimage.segmentation import mark_boundaries
    a, ma = exp_a.get_image_and_mask(label, positive_only=True, num_features=12, hide_rest=False)
    b, mb = exp_b.get_image_and_mask(label, positive_only=True, num_features=12, hide_rest=False)
    a = mark_boundaries(a / 255.0 if a.max() > 1 else a, ma)
    b = mark_boundaries(b / 255.0 if b.max() > 1 else b, mb)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.imshow(a); plt.axis("off"); plt.title("Explanation A")
    plt.subplot(1, 2, 2); plt.imshow(b); plt.axis("off"); plt.title("Explanation B")
    plt.suptitle(f"Disagreement comparison — {categories[label]}")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
