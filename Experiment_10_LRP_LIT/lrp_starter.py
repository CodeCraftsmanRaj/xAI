"""Experiment 10: Layer-wise Relevance Propagation (LRP) and Gradient x Input.

Runs the provided tiny movie-sentiment model, implements LRP-0/epsilon from scratch,
and writes reproducible numerical outputs and plots to results/.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

SEED = 0
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ---------- 1. Data: 24 train + 8 test sentences ----------
train = [
    ("i love this movie",1),("great acting and a wonderful story",1),("a brilliant and moving film",1),
    ("i really liked the characters",1),("wonderful music and great direction",1),("a fun and lovely movie",1),
    ("the story was moving and the acting was great",1),("i loved the ending",1),
    ("an excellent film i love it",1),("a beautiful and wonderful story",1),("great fun and a brilliant cast",1),
    ("the movie was lovely",1),
    ("i hate this movie",0),("terrible acting and a boring story",0),("a dull and awful film",0),
    ("i really disliked the characters",0),("boring music and terrible direction",0),("a bad and boring movie",0),
    ("the story was dull and the acting was awful",0),("i hated the ending",0),
    ("an awful film i hate it",0),("a boring and terrible story",0),("bad acting and a dull cast",0),
    ("the movie was awful",0)]
test = [("a great and lovely story",1),("i love the acting",1),("a wonderful film",1),("the cast was brilliant",1),
        ("a terrible and dull movie",0),("i hate the story",0),("an awful film",0),("the cast was boring",0)]

Xtr_txt, ytr = zip(*train)
Xte_txt, yte = zip(*test)
vec = CountVectorizer(binary=True)
Xtr = vec.fit_transform(Xtr_txt).toarray().astype(float)
Xte = vec.transform(Xte_txt).toarray().astype(float)
vocab = np.array(vec.get_feature_names_out())

# ---------- 2. Train model ----------
clf = MLPClassifier(hidden_layer_sizes=(8,), activation="relu", max_iter=2000,
                    random_state=SEED, solver="adam")
clf.fit(Xtr, ytr)
W1, W2 = clf.coefs_
b1, b2 = clf.intercepts_
train_acc = accuracy_score(ytr, clf.predict(Xtr))
test_acc = accuracy_score(yte, clf.predict(Xte))


def forward(x, W1, b1, W2, b2):
    """Return hidden ReLU activations and scalar output logit."""
    a1 = np.maximum(0.0, x @ W1 + b1)
    logit = float((a1 @ W2 + b2).reshape(-1)[0])
    return a1, logit


def safe_ratio(numerator, denominator):
    """Divide safely while preserving denominator signs; zero-denominator terms -> 0."""
    denominator = np.asarray(denominator, dtype=float)
    numerator = np.asarray(numerator, dtype=float)
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=np.abs(denominator) > 1e-12)
    return result


def lrp(x, W1, b1, W2, b2, eps=0.0):
    """LRP epsilon rule for this one-hidden-layer network.

    Output relevance starts at the output logit. Bias terms are excluded from
    redistribution denominators as required by the experiment handout.
    Returns (word relevance vector, hidden relevance vector, output logit).
    """
    a1, logit = forward(x, W1, b1, W2, b2)
    # Output -> hidden: z_k = hidden_activation_k * output_weight_k
    z_out = a1 * W2[:, 0]
    denom_out = np.sum(z_out) + eps
    R_hidden = safe_ratio(z_out, denom_out) * logit
    # Hidden -> input words: z_jk = x_j * W1_jk; normalize over input words j.
    z_in = x[:, None] * W1
    denom_in = np.sum(z_in, axis=0) + eps
    messages = safe_ratio(z_in, denom_in[None, :]) * R_hidden[None, :]
    R_words = np.sum(messages, axis=1)
    return R_words, R_hidden, logit


def grad_x_input(x, W1, b1, W2, b2):
    """Gradient x Input for the scalar output logit, one value per vocabulary term."""
    preactivation = x @ W1 + b1
    mask = (preactivation > 0).astype(float)
    gradient = W1 @ (mask * W2[:, 0])
    return x * gradient


def word_table(sentence, x, relevances):
    rows=[]
    for idx in np.flatnonzero(x):
        rows.append({"sentence": sentence, "word": vocab[idx], "lrp_relevance": float(relevances[idx])})
    return rows


def save_bar_chart(sentence, names, values, title, filename):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = np.argsort(values)
    ax.barh(np.asarray(names)[order], np.asarray(values)[order])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Relevance / attribution")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def main():
    print("=" * 72)
    print("EXPERIMENT 10 - LRP AND GRADIENT x INPUT")
    print("=" * 72)
    print(f"Training sentences: {len(train)} | Test sentences: {len(test)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Hidden units: {W1.shape[1]}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"W1 shape: {W1.shape}; b1 shape: {b1.shape}")
    print(f"W2 shape: {W2.shape}; b2 shape: {b2.shape}")
    print(f"W1={W1.shape}, b1={b1.shape}, W2={W2.shape}, b2={b2.shape}")

    sent = "i love this movie"
    x = vec.transform([sent]).toarray().astype(float)[0]
    prob = float(clf.predict_proba([x])[0, 1])
    pred = "POSITIVE" if clf.predict([x])[0] == 1 else "NEGATIVE"
    R_words, R_hidden, logit = lrp(x, W1, b1, W2, b2, eps=0.0)
    a1, _ = forward(x, W1, b1, W2, b2)
    print("\n--- Step 6: real model, LRP-0 ---")
    print(f"Sentence: {sent}")
    print(f"Prediction: {pred}; P(positive)={prob:.4f}")
    print(f"Output logit: {logit:.6f}")
    print("Hidden activations:", np.round(a1, 6).tolist())
    print("Hidden relevance:", np.round(R_hidden, 6).tolist())
    print("Word relevance:")
    for row in word_table(sent, x, R_words):
        print(f"  {row['word']:<10s} {row['lrp_relevance']:+.6f}")

    # Step 7: bias-free copy. Same trained weights; set both biases to zero.
    zero_b1, zero_b2 = np.zeros_like(b1), np.zeros_like(b2)
    R0_words, R0_hidden, logit0 = lrp(x, W1, zero_b1, W2, zero_b2, eps=0.0)
    gxi0 = grad_x_input(x, W1, zero_b1, W2, zero_b2)
    print("\n--- Step 7: bias-free copy ---")
    print(f"Bias-free output logit: {logit0:.6f}")
    print("Word                 LRP-0         Gradient x Input")
    for idx in np.flatnonzero(x):
        print(f"{vocab[idx]:<20s} {R0_words[idx]:+12.6f} {gxi0[idx]:+18.6f}")

    # Step 8: compare methods on the original model with biases.
    Rreal, _, logit_real = lrp(x, W1, b1, W2, b2, eps=0.0)
    gxireal = grad_x_input(x, W1, b1, W2, b2)
    print("\n--- Step 8: original model with biases ---")
    print("Word                 LRP-0         Gradient x Input")
    comparison=[]
    for idx in np.flatnonzero(x):
        print(f"{vocab[idx]:<20s} {Rreal[idx]:+12.6f} {gxireal[idx]:+18.6f}")
        comparison.append({"word": vocab[idx], "lrp0": float(Rreal[idx]), "gradient_x_input": float(gxireal[idx])})

    # Step 9: epsilon sweep.
    print("\n--- Step 9: epsilon sensitivity ---")
    eps_rows=[]
    for eps in (0.0, 0.1, 0.5, 1.0, 2.0):
        rel, _, eps_logit = lrp(x, W1, b1, W2, b2, eps=eps)
        print(f"epsilon={eps:>3.1f} | logit={eps_logit:+.6f} | " + ", ".join(f"{vocab[i]}={rel[i]:+.5f}" for i in np.flatnonzero(x)))
        for idx in np.flatnonzero(x):
            eps_rows.append({"epsilon": eps, "word": vocab[idx], "relevance": float(rel[idx]), "logit": float(eps_logit)})

    # Evaluate the tiny held-out set, and save artifacts.
    test_predictions=[]
    for text, truth in test:
        xx = vec.transform([text]).toarray().astype(float)[0]
        pred_label = int(clf.predict([xx])[0])
        test_predictions.append({"sentence": text, "true_label": int(truth), "predicted_label": pred_label,
                                 "probability_positive": float(clf.predict_proba([xx])[0, 1]),
                                 "correct": bool(pred_label == truth)})

    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"vocabulary_size": int(len(vocab)), "hidden_units": int(W1.shape[1]),
                   "train_accuracy": float(train_acc), "test_accuracy": float(test_acc),
                   "weight_shapes": {"W1": list(W1.shape), "b1": list(b1.shape), "W2": list(W2.shape), "b2": list(b2.shape)},
                   "sentence": sent, "prediction": pred, "positive_probability": prob,
                   "output_logit": logit_real}, f, indent=2)
    for filename, rows in [("word_relevance_real.csv", word_table(sent, x, Rreal)),
                           ("comparison_bias_free.csv", [{"word": vocab[i], "lrp0": float(R0_words[i]), "gradient_x_input": float(gxi0[i])} for i in np.flatnonzero(x)]),
                           ("comparison_with_bias.csv", comparison), ("epsilon_sweep.csv", eps_rows),
                           ("test_predictions.csv", test_predictions)]:
        if rows:
            with open(OUT / filename, "w", newline="", encoding="utf-8") as f:
                writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
    words=[vocab[i] for i in np.flatnonzero(x)]
    save_bar_chart(sent, words, Rreal[np.flatnonzero(x)], "LRP-0: i love this movie", "lrp_words.png")
    save_bar_chart(sent, words, gxireal[np.flatnonzero(x)], "Gradient x Input: i love this movie", "gradient_x_input_words.png")
    print(f"\nSaved metrics, CSV tables, and plots in: {OUT.resolve()}")
    print("Note: this script covers the custom LRP/Gradient x Input portion. The handout's separate LIT interface task is described in README.md.")

if __name__ == "__main__":
    main()
