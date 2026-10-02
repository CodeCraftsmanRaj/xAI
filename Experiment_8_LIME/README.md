# Experiment 8 — LIME with PyTorch

## Aim
To explain prediction of a pre-trained model using LIME.

## Experiment requirements
1. Use a pretrained ImageNet model.
2. Use three pictures with one clear main object and different backgrounds.
3. Generate LIME attribution maps.
4. For one picture, create two reasonable LIME settings whose explanations disagree.
5. State in advance what model-prediction evidence would support explanation A or B.
6. Use only the model's predictions to test which explanation is better supported.

## Implementation
- Model: pretrained ResNet-50 from `torchvision`.
- Framework: PyTorch (no TensorFlow).
- Explainability: `LimeImageExplainer`.
- Default segmentation: SLIC.
- Three bundled example images are created from `skimage.data`:
  - cat (`chelsea`)
  - coffee (`coffee`)
  - astronaut (`astronaut`)
- The disagreement search varies **only the number of superpixels** while holding other LIME settings fixed.
- A prediction-only deletion/insertion-style test evaluates the two masks. No image ground-truth labels are used for this test.

## Expected run
```bash
uv add -r requirements.txt
uv run main.py
```

Or:
```bash
uv run python main.py
```

If you already have your own images, put three JPG/PNG files in `data/images/`. The bundled examples can be regenerated with:
```bash
uv run python prepare_images.py
```

## Output
All experiment outputs are stored in `results/`:
- `results/attributions/` — LIME visualizations for all three images.
- `results/disagreement/` — candidate explanations and the selected A/B pair.
- `results/tables/` — CSV/JSON numerical results.
- `results/summary.json` — machine-readable run summary.
- `results/decision.md` — prediction-only decision using the pre-registered criterion.
- `results/predictions.csv` — top model predictions for each image.

## Pre-registered prediction-only criterion
Before looking at the deletion/insertion result:

- Evidence for A: A's positive regions cause a larger target-probability decrease when removed and/or a faster target-probability increase when inserted than B's regions, by at least the configured decision margin.
- Evidence for B: the reverse.
- If the evidence does not separate A and B by the margin, report **neither/inconclusive**.

This is a model-behavior test, not proof that a LIME explanation is objectively correct.

## Important
The experiment automatically records the exact settings used, random seeds, predicted class, LIME fidelity score, explanation overlap, and prediction-test results so they can be used later for the report/documentation.
