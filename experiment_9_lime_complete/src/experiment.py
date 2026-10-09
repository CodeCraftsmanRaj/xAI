from pathlib import Path
import html
import json
import logging
import random

import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

from src.predictor import SentimentPredictor


def probability_dict(probabilities, id2label):
    return {id2label[i]: float(probabilities[i]) for i in range(len(probabilities))}


def run_lime(explainer, predictor, sentence, positive_idx, num_samples, num_features):
    explanation = explainer.explain_instance(
        sentence,
        predictor.predict_proba,
        labels=(positive_idx,),
        num_samples=int(num_samples),
        num_features=max(1, int(num_features)),
    )
    weights = explanation.as_list(label=positive_idx)
    score_df = pd.DataFrame(weights, columns=["word", "lime_weight"])
    score_df["num_samples"] = int(num_samples)
    score_df["explained_label"] = "POSITIVE"
    return explanation, score_df


def token_words(sentence):
    # A whitespace-based token list makes the remove-one-word experiment transparent.
    return sentence.split()


def remove_one_word_experiment(sentence, predictor, id2label, positive_idx):
    words = token_words(sentence)
    rows = []
    base = predictor.predict_proba([sentence])[0]
    rows.append({
        "removed_word": "(none — original)",
        "sentence": sentence,
        **{f"probability_{label.lower()}": float(base[idx]) for idx, label in id2label.items()},
        "change_in_positive_probability": 0.0,
    })
    for i, word in enumerate(words):
        remaining = words[:i] + words[i + 1:]
        changed = " ".join(remaining)
        probs = predictor.predict_proba([changed])[0]
        rows.append({
            "removed_word": word,
            "word_position": i,
            "sentence": changed,
            **{f"probability_{label.lower()}": float(probs[idx]) for idx, label in id2label.items()},
            "change_in_positive_probability": float(probs[positive_idx] - base[positive_idx]),
        })
    return pd.DataFrame(rows)


def select_perturbations(sentence, predictor, positive_idx, id2label, random_seed, records):
    # Use only inputs logged during this sentence's full-sample LIME run.
    seen = set()
    candidates = []
    for record in records:
        text = record["sentence"]
        if text != sentence and text not in seen:
            seen.add(text)
            candidates.append(text)
    if not candidates:
        return pd.DataFrame()

    probs = predictor.predict_proba_without_logging(candidates)
    label = id2label[positive_idx].lower()
    rows = []
    for text, prob in zip(candidates, probs):
        rows.append({
            "kind": "changed_copy",
            "sentence": text,
            **{f"probability_{name.lower()}": float(prob[idx]) for idx, name in id2label.items()},
            "positive_probability": float(prob[positive_idx]),
        })
    frame = pd.DataFrame(rows)
    rng = random.Random(random_seed)
    random_count = min(8, len(frame))
    random_indices = set(rng.sample(list(frame.index), random_count))
    selected = []
    selected.append({
        "kind": "original",
        "sentence": sentence,
        **{f"probability_{name.lower()}": float(predictor.predict_proba_without_logging([sentence])[0][idx])
           for idx, name in id2label.items()},
    })
    for idx in sorted(random_indices):
        row = frame.loc[idx].to_dict()
        row["kind"] = f"random_changed_{len([r for r in selected if str(r['kind']).startswith('random_changed_')]) + 1}"
        selected.append(row)
    lowest = frame.nsmallest(min(3, len(frame)), "positive_probability")
    highest = frame.nlargest(min(3, len(frame)), "positive_probability")
    for rank, (_, row) in enumerate(lowest.iterrows(), 1):
        item = row.to_dict()
        item["kind"] = f"lowest_positive_probability_{rank}"
        selected.append(item)
    for rank, (_, row) in enumerate(highest.iterrows(), 1):
        item = row.to_dict()
        item["kind"] = f"highest_positive_probability_{rank}"
        selected.append(item)
    return pd.DataFrame(selected)


def run_experiment(config, tokenizer, model, device, id2label, label2id, logger):
    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_idx = label2id.get(str(config.get("explained_label", "POSITIVE")).upper())
    if positive_idx is None:
        raise ValueError(f"Configured explained label is missing. Available labels: {label2id}")
    predictor = SentimentPredictor(
        tokenizer, model, device, id2label,
        batch_size=config.get("batch_size", 16),
    )
    explainer = LimeTextExplainer(
        class_names=[id2label[i] for i in range(len(id2label))],
        random_state=int(config.get("random_seed", 42)),
        bow=True,
    )

    sentence_results = []
    sentences = config.get("sentences", [])
    if not sentences:
        raise ValueError("Add at least one sentence in config.yaml.")

    for sentence_number, sentence in enumerate(sentences, start=1):
        logger.info("Explaining sentence %d/%d", sentence_number, len(sentences))
        original_probs = predictor.predict_proba([sentence])[0]
        original_dict = probability_dict(original_probs, id2label)

        full_n = int(config.get("num_samples_full", 1000))
        small_n = int(config.get("num_samples_small", 100))
        full_run_log_start = len(predictor.call_records)
        full_exp, full_scores = run_lime(
            explainer, predictor, sentence, positive_idx, full_n,
            num_features=max(1, len(sentence.split())),
        )
        full_scores.insert(0, "sentence", sentence)
        full_scores.to_csv(output_dir / f"lime_scores_{full_n}_{sentence_number}.csv", index=False)

        # Keep the first sentence's score file under the exact requested friendly name.
        if sentence_number == 1:
            full_scores.to_csv(output_dir / f"lime_scores_{full_n}.csv", index=False)

        small_exp, small_scores = run_lime(
            explainer, predictor, sentence, positive_idx, small_n,
            num_features=max(1, len(sentence.split())),
        )
        small_scores.insert(0, "sentence", sentence)
        small_scores.to_csv(output_dir / f"lime_scores_{small_n}_{sentence_number}.csv", index=False)
        if sentence_number == 1:
            small_scores.to_csv(output_dir / f"lime_scores_{small_n}.csv", index=False)

        full_run_records = predictor.call_records[full_run_log_start:]
        selected = select_perturbations(
            sentence, predictor, positive_idx, id2label,
            int(config.get("random_seed", 42)) + sentence_number,
            full_run_records,
        )
        selected.insert(0, "source_sentence_number", sentence_number)
        selected.to_csv(
            output_dir / f"selected_perturbations_{sentence_number}.csv", index=False
        )
        if sentence_number == 1:
            selected.to_csv(output_dir / "selected_perturbations.csv", index=False)

        remove_df = remove_one_word_experiment(sentence, predictor, id2label, positive_idx)
        remove_df.insert(0, "source_sentence_number", sentence_number)
        remove_df.to_csv(output_dir / f"remove_one_word_{sentence_number}.csv", index=False)
        if sentence_number == 1:
            remove_df.to_csv(output_dir / "remove_one_word.csv", index=False)

        html_path = output_dir / f"lime_explanation_{sentence_number}.html"
        full_exp.save_to_file(str(html_path))
        if sentence_number == 1:
            full_exp.save_to_file(str(output_dir / "lime_explanation.html"))

        sorted_scores = sorted(
            [(str(word), float(weight)) for word, weight in full_exp.as_list(label=positive_idx)],
            key=lambda pair: pair[1],
        )
        largest_negative = sorted_scores[0] if sorted_scores else ("", 0.0)
        largest_positive = sorted_scores[-1] if sorted_scores else ("", 0.0)

        # Rank comparisons: align by word where possible; report rank correlation where defined.
        full_rank = full_scores[["word", "lime_weight"]].copy()
        small_rank = small_scores[["word", "lime_weight"]].copy()
        rank_merge = full_rank.merge(small_rank, on="word", suffixes=("_full", "_small"))
        rank_corr = None
        if len(rank_merge) >= 2:
            corr = rank_merge["lime_weight_full"].corr(rank_merge["lime_weight_small"], method="spearman")
            rank_corr = float(corr) if pd.notna(corr) else None

        sentence_results.append({
            "sentence": sentence,
            "original_probabilities": original_dict,
            "explained_label": id2label[positive_idx],
            "lime_samples_full": full_n,
            "lime_samples_small": small_n,
            "largest_positive_word": {"word": largest_positive[0], "weight": largest_positive[1]},
            "largest_negative_word": {"word": largest_negative[0], "weight": largest_negative[1]},
            "lime_scores_full": [
                {"word": str(w), "weight": float(s)}
                for w, s in full_scores[["word", "lime_weight"]].itertuples(index=False, name=None)
            ],
            "lime_scores_small": [
                {"word": str(w), "weight": float(s)}
                for w, s in small_scores[["word", "lime_weight"]].itertuples(index=False, name=None)
            ],
            "full_vs_small_spearman_rank_correlation_on_shared_words": rank_corr,
            "remove_one_word_csv": f"remove_one_word_{sentence_number}.csv",
            "selected_perturbations_csv": f"selected_perturbations_{sentence_number}.csv",
            "lime_html": html_path.name,
        })

    # Export the complete record of all sentences passed into the LIME prediction callback.
    # Probability columns are calculated afterward without expanding the call log.
    predictor.call_log_frame().to_csv(output_dir / "model_calls.csv", index=False)

    return {
        "model_name": config["model_name"],
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": label2id,
        "device": str(device),
        "sentences": sentence_results,
        "model_call_count": len(predictor.call_records),
        "notes": [
            "model_calls.csv logs every text passed to the prediction callback, including repeated inputs.",
            "The selected perturbation file deduplicates changed sentences for readability.",
            "LIME weights explain the configured label locally; they are not probabilities or causal effects.",
            "Remove-one-word results are direct model predictions and may differ from LIME's local surrogate ranking.",
        ],
    }
