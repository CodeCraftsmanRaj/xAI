from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from .utils import load_config, seed_everything, get_device, ensure_dirs, save_json
from .model import load_model, predict_batch
from .lime_runner import explain_image, render_explanation, render_side_by_side
from .prediction_test import evaluate_superpixel_order, summarize_prediction_test

def load_images(cfg):
    image_dir = Path(cfg["data"]["image_dir"])
    image_dir.mkdir(parents=True, exist_ok=True)
    if cfg["data"].get("use_bundled_examples", True):
        files = list(image_dir.glob("*"))
        if len([p for p in files if p.suffix.lower() in cfg["data"]["extensions"]]) < 3:
            from skimage import data
            examples = {
                "cat.jpg": data.chelsea(),
                "coffee.jpg": data.coffee(),
                "astronaut.jpg": data.astronaut(),
            }
            for name, arr in examples.items():
                Image.fromarray(arr).save(image_dir / name, quality=95)

    paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in cfg["data"]["extensions"]
    ])
    if len(paths) < 3:
        raise RuntimeError("Need at least three images in data/images.")
    return paths[:3]

def predict_fn_factory(model, preprocess, device):
    def fn(images):
        pil_images = [Image.fromarray(np.uint8(np.clip(x, 0, 255))) for x in images]
        return predict_batch(model, pil_images, preprocess, device)
    return fn

def prediction_rows(probs, categories, image_name, top_k):
    idx = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "image": image_name,
            "rank": int(i + 1),
            "class_index": int(j),
            "class": categories[j],
            "probability": float(probs[j]),
        }
        for i, j in enumerate(idx)
    ]

def positive_regions(exp, label, top_n):
    pairs = sorted(exp.local_exp[label], key=lambda x: x[1], reverse=True)
    return [int(seg) for seg, weight in pairs if weight > 0][:top_n]

def pixel_mask(exp, ids):
    return np.isin(exp.segments, ids)

def run_experiment(config_path):
    cfg = load_config(config_path)
    seed_everything(cfg["experiment"]["seed"])
    root = ensure_dirs(cfg)
    device = get_device(cfg)
    print("=" * 72)
    print("EXPERIMENT 8 - LIME")
    print("=" * 72)
    print("Device:", device)

    model, preprocess, categories = load_model(cfg, device)
    predict_fn = predict_fn_factory(model, preprocess, device)
    image_paths = load_images(cfg)

    # Baseline predictions on all three images.
    images = [Image.open(p).convert("RGB") for p in image_paths]
    probs = predict_batch(model, images, preprocess, device)
    pred_rows = []
    for p, pr in zip(image_paths, probs):
        pred_rows.extend(prediction_rows(pr, categories, p.name, cfg["experiment"]["top_k_classes"]))
    pd.DataFrame(pred_rows).to_csv(root / "predictions.csv", index=False)

    # Required procedure: attribution maps for three images.
    base_nseg = cfg["lime"]["disagreement"]["candidate_num_segments"][1]
    base_nsamp = cfg["lime"]["num_samples"]
    three_results = []
    explanations = {}
    for path, image in zip(image_paths, images):
        exp = explain_image(
            image, predict_fn, cfg,
            n_segments=base_nseg,
            num_samples=base_nsamp,
            seed=cfg["experiment"]["seed"],
        )
        label = int(exp.top_labels[0])
        explanations[path.name] = exp
        out = root / "attributions" / f"{path.stem}_lime.png"
        render_explanation(image, exp, label, categories, out, cfg["lime"]["num_features"])
        three_results.append({
            "image": path.name,
            "predicted_class_index": label,
            "predicted_class": categories[label],
            "prediction_probability": float(probs[image_paths.index(path), label]),
            "lime_fidelity_r2": float(exp.score),
            "n_segments": base_nseg,
            "num_samples": base_nsamp,
            "attribution_file": str(out),
        })

    # Choose the first image as the mandatory disagreement-study image.
    study_path = image_paths[0]
    study_image = images[0]
    baseline_probs = probs[0]
    target_label = int(np.argmax(baseline_probs))
    candidate_segments = cfg["lime"]["disagreement"]["candidate_num_segments"]
    candidate_nsamp = cfg["lime"]["disagreement"]["candidate_num_samples"]
    min_fid = cfg["lime"]["disagreement"]["min_fidelity_r2"]
    min_dis = cfg["lime"]["disagreement"]["min_disagreement"]
    top_segments = cfg["lime"]["disagreement"]["top_segments"]

    candidates = []
    for nseg in candidate_segments:
        exp = explain_image(
            study_image, predict_fn, cfg,
            n_segments=nseg,
            num_samples=candidate_nsamp,
            seed=cfg["experiment"]["seed"],
        )
        label = target_label
        ids = positive_regions(exp, label, top_segments)
        mask = pixel_mask(exp, ids)
        candidates.append({
            "n_segments": nseg,
            "num_samples": candidate_nsamp,
            "seed": cfg["experiment"]["seed"],
            "fidelity_r2": float(exp.score),
            "positive_region_ids": ids,
            "positive_area_fraction": float(mask.mean()),
            "explanation": exp,
            "mask": mask,
        })

    pair_rows = []
    for a, b in itertools.combinations(candidates, 2):
        inter = np.logical_and(a["mask"], b["mask"]).sum()
        union = np.logical_or(a["mask"], b["mask"]).sum()
        jaccard = float(inter / union) if union else 1.0
        disagreement = 1.0 - jaccard
        both_reasonable = a["fidelity_r2"] >= min_fid and b["fidelity_r2"] >= min_fid
        pair_rows.append({
            "A_n_segments": a["n_segments"],
            "B_n_segments": b["n_segments"],
            "A_fidelity_r2": a["fidelity_r2"],
            "B_fidelity_r2": b["fidelity_r2"],
            "pixel_mask_jaccard": jaccard,
            "pixel_mask_disagreement": disagreement,
            "both_reasonable": both_reasonable,
            "meets_disagreement_threshold": disagreement >= min_dis,
        })

    # Prefer a pair meeting both thresholds; otherwise select the most
    # disagreeing pair so the run remains inspectable rather than silently failing.
    eligible = [r for r in pair_rows if r["both_reasonable"] and r["meets_disagreement_threshold"]]
    if eligible:
        chosen_row = max(eligible, key=lambda r: r["pixel_mask_disagreement"])
    else:
        reasonable = [r for r in pair_rows if r["both_reasonable"]]
        pool = reasonable if reasonable else pair_rows
        chosen_row = max(pool, key=lambda r: r["pixel_mask_disagreement"])

    A = next(c for c in candidates if c["n_segments"] == chosen_row["A_n_segments"])
    B = next(c for c in candidates if c["n_segments"] == chosen_row["B_n_segments"])

    exp_a = A["explanation"]
    exp_b = B["explanation"]
    render_explanation(
        study_image, exp_a, target_label, categories,
        root / "disagreement" / "explanation_A.png", cfg["lime"]["num_features"]
    )
    render_explanation(
        study_image, exp_b, target_label, categories,
        root / "disagreement" / "explanation_B.png", cfg["lime"]["num_features"]
    )
    render_side_by_side(
        study_image, exp_a, exp_b, target_label, categories,
        root / "disagreement" / "A_vs_B.png"
    )

    # Pre-registered evidence statement is written before prediction testing.
    criterion = {
        "before_test": True,
        "target_class": categories[target_label],
        "statement": (
            "Evidence for A requires A's positive regions to produce a larger "
            "target-probability decrease under deletion and/or a larger target "
            "probability under insertion than B, with the combined difference "
            "exceeding the configured decision margin. Evidence for B is the "
            "reverse. Otherwise the result is neither/inconclusive."
        ),
        "decision_margin": cfg["prediction_test"]["decision_margin"],
    }
    (root / "decision_criterion.md").write_text(
        "# Pre-registered decision criterion\n\n"
        f"Target class: **{categories[target_label]}**\n\n"
        f"{criterion['statement']}\n",
        encoding="utf-8"
    )

    ids_a = positive_regions(exp_a, target_label, top_segments)
    ids_b = positive_regions(exp_b, target_label, top_segments)
    rows_a = evaluate_superpixel_order(
        study_image, exp_a.segments, ids_a, target_label, predict_fn,
        cfg["prediction_test"]["fractions"], cfg["prediction_test"]["blur_radius"]
    )
    rows_b = evaluate_superpixel_order(
        study_image, exp_b.segments, ids_b, target_label, predict_fn,
        cfg["prediction_test"]["fractions"], cfg["prediction_test"]["blur_radius"]
    )
    decision = summarize_prediction_test(
        float(baseline_probs[target_label]),
        rows_a, rows_b,
        cfg["prediction_test"]["decision_margin"]
    )

    test_df = pd.DataFrame([
        {"explanation": "A", **r} for r in rows_a
    ] + [
        {"explanation": "B", **r} for r in rows_b
    ])
    test_df.to_csv(root / "tables" / "prediction_test.csv", index=False)

    # Plot model-prediction deletion curves.
    plt.figure(figsize=(8, 5))
    plt.plot([r["fraction"] for r in rows_a], [r["removed_target_probability"] for r in rows_a], marker="o", label="A deletion")
    plt.plot([r["fraction"] for r in rows_b], [r["removed_target_probability"] for r in rows_b], marker="o", label="B deletion")
    plt.axhline(float(baseline_probs[target_label]), linestyle="--", label="Original")
    plt.xlabel("Fraction of selected positive regions removed")
    plt.ylabel(f"P({categories[target_label]})")
    plt.title("Prediction-only deletion test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "disagreement" / "deletion_test.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot([r["fraction"] for r in rows_a], [r["inserted_target_probability"] for r in rows_a], marker="o", label="A insertion")
    plt.plot([r["fraction"] for r in rows_b], [r["inserted_target_probability"] for r in rows_b], marker="o", label="B insertion")
    plt.xlabel("Fraction of selected positive regions inserted")
    plt.ylabel(f"P({categories[target_label]})")
    plt.title("Prediction-only insertion test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "disagreement" / "insertion_test.png", dpi=180)
    plt.close()

    # Save complete numerical summary.
    candidate_public = []
    for c in candidates:
        candidate_public.append({
            "n_segments": c["n_segments"],
            "num_samples": c["num_samples"],
            "seed": c["seed"],
            "fidelity_r2": c["fidelity_r2"],
            "positive_region_ids": c["positive_region_ids"],
            "positive_area_fraction": c["positive_area_fraction"],
        })

    summary = {
        "device": str(device),
        "model": cfg["model"]["architecture"],
        "study_image": study_path.name,
        "target_class_index": target_label,
        "target_class": categories[target_label],
        "baseline_target_probability": float(baseline_probs[target_label]),
        "three_image_attributions": three_results,
        "candidates": candidate_public,
        "pair_comparisons": pair_rows,
        "selected_pair": chosen_row,
        "selected_A": {
            "n_segments": A["n_segments"],
            "fidelity_r2": A["fidelity_r2"],
            "positive_region_ids": ids_a,
        },
        "selected_B": {
            "n_segments": B["n_segments"],
            "fidelity_r2": B["fidelity_r2"],
            "positive_region_ids": ids_b,
        },
        "pre_registered_criterion": criterion,
        "prediction_test": decision,
        "prediction_test_rows_A": rows_a,
        "prediction_test_rows_B": rows_b,
    }
    save_json(summary, root / "summary.json")
    pd.DataFrame(three_results).to_csv(root / "tables" / "three_image_lime.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(root / "tables" / "disagreement_pairs.csv", index=False)

    decision_text = (
        "# Experiment 8 — Prediction-only conclusion\n\n"
        f"- Study image: `{study_path.name}`\n"
        f"- Target class: **{categories[target_label]}**\n"
        f"- Explanation A: SLIC `n_segments={A['n_segments']}`, "
        f"LIME fidelity R²={A['fidelity_r2']:.4f}\n"
        f"- Explanation B: SLIC `n_segments={B['n_segments']}`, "
        f"LIME fidelity R²={B['fidelity_r2']:.4f}\n"
        f"- Pixel-mask disagreement: {chosen_row['pixel_mask_disagreement']:.4f}\n\n"
        "## Prediction-only result\n"
        f"**Decision: {decision['decision']}**\n\n"
        f"- A deletion AUC (probability drop): {decision['deletion_auc_drop_A']:.6f}\n"
        f"- B deletion AUC (probability drop): {decision['deletion_auc_drop_B']:.6f}\n"
        f"- A insertion AUC: {decision['insertion_auc_A']:.6f}\n"
        f"- B insertion AUC: {decision['insertion_auc_B']:.6f}\n"
        f"- Combined A-minus-B support: {decision['combined_support_A_minus_B']:.6f}\n"
        f"- Decision margin: {decision['decision_margin']:.6f}\n\n"
        "The test uses only the pretrained model's predicted probabilities on "
        "modified versions of the image. It does not use a ground-truth label. "
        "Therefore it provides evidence about which explanation better tracks "
        "this model's prediction, not proof of the image's real-world causal truth.\n"
    )
    (root / "decision.md").write_text(decision_text, encoding="utf-8")

    print("\nTarget:", categories[target_label])
    print("Selected A:", A["n_segments"], "segments | R2:", round(A["fidelity_r2"], 4))
    print("Selected B:", B["n_segments"], "segments | R2:", round(B["fidelity_r2"], 4))
    print("Disagreement:", round(chosen_row["pixel_mask_disagreement"], 4))
    print("Prediction-only decision:", decision["decision"])
    print("\nResults saved to:", root.resolve())
