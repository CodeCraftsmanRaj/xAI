"""
model_utils.py

Loads a ResNet18 model pretrained on ImageNet-1K and provides:
  - predict(): top-k class predictions with confidence scores
  - explain(): Integrated Gradients attribution map for a chosen class,
               with a choice of baseline (black / white / noise / blurred input)

This is the only file that talks to PyTorch / Captum — main.py just calls into it.
"""
import io
import base64

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from captum.attr import IntegratedGradients
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Model setup (runs once, at import time)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.eval()
model.to(device)

# The 1000 ImageNet-1K class names, straight from the weights metadata
# (no need to ship a separate class-index JSON file).
CLASSES = weights.meta["categories"]

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Resize/crop only (kept separate from normalization so we can show the
# "human-viewable" resized image, and so baselines can be built in pixel space).
resize_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
])

to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def preprocess_image(img: Image.Image):
    """Returns (normalized_tensor[1,3,224,224], resized_224_PIL_image)."""
    img_resized = resize_crop(img)
    tensor = to_tensor_norm(img_resized).unsqueeze(0).to(device)
    return tensor, img_resized


def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(img: Image.Image, topk: int = 5):
    tensor, img_resized = preprocess_image(img)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, topk)
    results = [
        {
            "class_idx": int(idx),
            "class_name": CLASSES[int(idx)],
            "confidence": float(p),
        }
        for p, idx in zip(top_probs, top_idxs)
    ]
    return results, img_resized


# ---------------------------------------------------------------------------
# Baselines for Integrated Gradients
# ---------------------------------------------------------------------------
def make_baseline(img_resized: Image.Image, baseline_type: str) -> torch.Tensor:
    w, h = img_resized.size

    if baseline_type == "black":
        baseline_img = Image.new("RGB", (w, h), (0, 0, 0))
    elif baseline_type == "white":
        baseline_img = Image.new("RGB", (w, h), (255, 255, 255))
    elif baseline_type == "blur":
        baseline_img = img_resized.filter(ImageFilter.GaussianBlur(radius=15))
    elif baseline_type == "noise":
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        baseline_img = Image.fromarray(arr)
    else:
        # sensible default
        baseline_img = Image.new("RGB", (w, h), (0, 0, 0))

    return to_tensor_norm(baseline_img).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Integrated Gradients explanation
# ---------------------------------------------------------------------------
def explain(img: Image.Image, target_idx: int, baseline_type: str = "black", n_steps: int = 50):
    tensor, img_resized = preprocess_image(img)
    tensor.requires_grad_()

    baseline_tensor = make_baseline(img_resized, baseline_type)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        tensor,
        baselines=baseline_tensor,
        target=target_idx,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    attr = attributions.squeeze(0).cpu().detach().numpy()      # (3, H, W)
    attr = np.transpose(attr, (1, 2, 0))                       # (H, W, 3)

    # Collapse channels: sum of absolute attribution per pixel
    attr_map = np.sum(np.abs(attr), axis=2)
    attr_map -= attr_map.min()
    if attr_map.max() > 0:
        attr_map /= attr_map.max()

    heatmap = cm.jet(attr_map)[:, :, :3]                       # (H, W, 3) in [0,1]
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))

    orig_arr = np.array(img_resized).astype(np.float32) / 255.0
    overlay = 0.55 * orig_arr + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    return {
        "heatmap": _pil_to_b64(heatmap_img),
        "overlay": _pil_to_b64(overlay_img),
        "original_resized": _pil_to_b64(img_resized),
        "convergence_delta": float(delta.abs().mean().item()),
        "baseline_type": baseline_type,
        "n_steps": n_steps,
    }
