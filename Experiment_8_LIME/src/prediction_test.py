import numpy as np
from PIL import Image, ImageFilter

def mask_image(image, mask, mode, blur_radius=8):
    arr = np.asarray(image).copy()
    if mode == "blur":
        base = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=blur_radius)))
    elif mode == "mean":
        base = np.full_like(arr, arr.reshape(-1, 3).mean(axis=0).astype(np.uint8))
    else:
        raise ValueError(mode)
    out = arr.copy()
    out[mask] = base[mask]
    return Image.fromarray(out)

def insert_mask(image, mask, blur_radius=8):
    arr = np.asarray(image).copy()
    base = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=blur_radius)))
    out = base.copy()
    out[mask] = arr[mask]
    return Image.fromarray(out)

def evaluate_mask(image, mask, target_label, predict_fn, fractions, blur_radius):
    scores = []
    total = int(mask.sum())
    if total == 0:
        return scores
    # Rank all positive pixels by a deterministic row-major order. For LIME
    # regions, masks are added in connected superpixel blocks via cumulative
    # area fractions.
    ys, xs = np.where(mask)
    order = np.lexsort((xs, ys))
    ys, xs = ys[order], xs[order]
    n = len(xs)
    for frac in fractions:
        k = max(1, int(n * frac))
        partial = np.zeros_like(mask, dtype=bool)
        partial[ys[:k], xs[:k]] = True
        removed = mask_image(image, partial, "blur", blur_radius)
        inserted = insert_mask(image, partial, blur_radius)
        p_remove = float(predict_fn([removed])[0][target_label])
        p_insert = float(predict_fn([inserted])[0][target_label])
        scores.append({
            "fraction": float(frac),
            "removed_target_probability": p_remove,
            "inserted_target_probability": p_insert,
        })
    return scores

def evaluate_superpixel_order(image, segments, selected_ids, target_label, predict_fn, fractions, blur_radius):
    selected_ids = list(selected_ids)
    if not selected_ids:
        return []
    out = []
    n = len(selected_ids)
    for frac in fractions:
        k = max(1, int(np.ceil(n * frac)))
        ids = selected_ids[:k]
        mask = np.isin(segments, ids)
        removed = mask_image(image, mask, "blur", blur_radius)
        inserted = insert_mask(image, mask, blur_radius)
        pr = float(predict_fn([removed])[0][target_label])
        pi = float(predict_fn([inserted])[0][target_label])
        out.append({
            "fraction": float(frac),
            "removed_target_probability": pr,
            "inserted_target_probability": pi,
            "num_regions": k,
            "area_fraction": float(mask.mean()),
        })
    return out

def auc(values, x):
    return float(np.trapezoid(values, x)) if len(values) > 1 else float(values[0])

def summarize_prediction_test(baseline_prob, rows_a, rows_b, margin):
    x = [r["fraction"] for r in rows_a]
    rem_a = [r["removed_target_probability"] for r in rows_a]
    rem_b = [r["removed_target_probability"] for r in rows_b]
    ins_a = [r["inserted_target_probability"] for r in rows_a]
    ins_b = [r["inserted_target_probability"] for r in rows_b]
    drop_a = [baseline_prob - v for v in rem_a]
    drop_b = [baseline_prob - v for v in rem_b]
    # Larger deletion drop and larger insertion probability support that mask.
    auc_drop_a, auc_drop_b = auc(drop_a, x), auc(drop_b, x)
    auc_ins_a, auc_ins_b = auc(ins_a, x), auc(ins_b, x)
    support_a = (auc_drop_a - auc_drop_b) + (auc_ins_a - auc_ins_b)
    support_b = -support_a
    if support_a > margin:
        decision = "A"
    elif support_b > margin:
        decision = "B"
    else:
        decision = "neither/inconclusive"
    return {
        "baseline_target_probability": baseline_prob,
        "deletion_auc_drop_A": auc_drop_a,
        "deletion_auc_drop_B": auc_drop_b,
        "insertion_auc_A": auc_ins_a,
        "insertion_auc_B": auc_ins_b,
        "combined_support_A_minus_B": support_a,
        "decision": decision,
        "decision_margin": margin,
    }
