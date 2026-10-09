# Experiment 9 — LIME for Movie Review Sentiment

This project applies LIME to explain the sentiment predictions of
`distilbert/distilbert-base-uncased-finetuned-sst-2-english`.

## Requirements

- Python environment managed by `uv`
- Internet access on the first run to download the Hugging Face model

Install dependencies using the command you normally use:

```bash
uv add -r requirements.txt
```

If your installed `uv` version does not accept `uv add -r`, use:

```bash
uv pip install -r requirements.txt
```

No library versions are pinned; your package manager resolves compatible versions.

## Run

```bash
uv run main.py
```

Edit `config.yaml` to change the sentences, number of LIME samples, random seed, and output directory.

## Output files

Results are saved under `results/` by default:

- `summary.json` — model labels, probabilities, LIME scores, and experiment metadata
- `lime_scores_1000.csv` — per-word LIME weights for the configured sample count
- `lime_scores_100.csv` — per-word weights for the 100-sample comparison
- `model_calls.csv` — every sentence passed to the model prediction function, including repeated calls
- `selected_perturbations.csv` — original sentence, 8 random changed copies, and the 3 lowest / 3 highest positive-probability perturbations
- `remove_one_word.csv` — original and one-word-removed predictions for each token
- `lime_explanation.html` — readable LIME visualization for the primary sentence
- `run.log` — execution log

CSV files include the sentence and model probabilities. `model_calls.csv` may be large because it deliberately records all LIME prediction inputs.

## Experiment behavior

The script runs the required sentence and your own sentence. The default custom sentence includes negation:
`The movie was not good, not good at all, despite the beautiful soundtrack.`

For each sentence, it:
1. Loads the model and discovers the actual label mapping from the model config.
2. Predicts both class probabilities.
3. Runs LIME with 1,000 samples, saves word weights and selected perturbations.
4. Removes one word at a time and saves the resulting probabilities.
5. Repeats LIME with 100 samples and saves the comparison scores.

LIME explanations are stochastic approximations. Compare the signed word weights, remembering that a positive weight supports the class being explained (configured as POSITIVE by default) and a negative weight pushes away from that class. The one-word-removal test is a separate intervention and does not have to rank words identically to LIME.

## Reproducibility note

A fixed seed is set where supported, but model/package implementations and LIME perturbation behavior can still vary. Keep the generated results directory intact and send it back for the report.
