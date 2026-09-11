# Experiment 4 — ICE and ALE for Nonlinear Model Interpretation

## Aim

Implement **Individual Conditional Expectation (ICE)** and **Accumulated Local Effects (ALE)** plots to analyze and interpret the effect of an input feature on predictions of a trained nonlinear machine-learning model.

The implementation follows the supplied Experiment 4 instructions: tabular data, preprocessing, train/test split, nonlinear model training and evaluation, followed by ICE and ALE analysis. fileciteturn0file0

## Dataset

The experiment uses the **Breast Cancer Wisconsin Diagnostic dataset** provided by `scikit-learn`.

- Samples: 569
- Numerical features: 30
- Target: malignant / benign
- Task: binary classification
- Selected interpretability feature: `mean radius`

This dataset is convenient for a reproducible lab experiment because it is bundled with scikit-learn and does not require a separate dataset download.

## Model

A **Random Forest Classifier** is used because it is a nonlinear tabular model and is explicitly one of the suitable model types listed in the experiment instructions. fileciteturn0file0

Configuration is stored in `config.yaml`.

## Project Structure

```text
Experiment4_ICE_ALE/
├── config.yaml
├── requirements.txt
├── train.py
├── evaluate.py
├── interpret.py
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── metrics.py
│   ├── interpret.py
│   └── plots.py
├── models/
│   └── random_forest.joblib
└── outputs/
    ├── metrics.json
    ├── feature_info.json
    ├── interpretation_summary.json
    └── plots/
        ├── ice_mean_radius.png
        └── ale_mean_radius.png
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

If using `uv`:

```bash
uv venv
uv pip install -r requirements.txt
```

## Running the Experiment

### 1. Train the model

```bash
python train.py
```

or:

```bash
uv run train.py
```

This trains the Random Forest, evaluates it on the test set, saves the model and writes `outputs/metrics.json`.

### 2. Generate ICE and ALE plots

```bash
python interpret.py
```

or:

```bash
uv run interpret.py
```

The plots are saved under `outputs/plots/`.

### 3. View evaluation metrics

```bash
python evaluate.py
```

## Preprocessing

The selected dataset contains numerical features and does not require categorical encoding. The Random Forest model is tree-based, so feature scaling is not required for training. The data is split using stratification so that the class proportions are preserved in training and testing.

## ICE Method

For the selected feature `mean radius`:

1. The observed feature distribution in the test set is used.
2. The 2nd to 98th percentiles define a robust analysis range.
3. 50 equally spaced values are generated to form the feature vector `R`.
4. 60 test samples are selected.
5. For each value in `R`, the selected feature is replaced by that value while all other features remain unchanged.
6. The model predicts the positive-class probability.
7. One curve is produced for each selected individual.

Thus, an ICE curve shows how the prediction for an individual changes when the selected feature is varied while the other input values are held fixed. This directly implements the procedure described in the supplied instructions. fileciteturn0file0

### Interpretation

- Similar curves indicate that the feature has a relatively consistent effect across individuals.
- Curves with different shapes or slopes indicate heterogeneous feature effects.
- Steep sections indicate regions where the prediction changes rapidly.
- Flat sections indicate relatively little sensitivity to the selected feature.

The thick curve in the plot is the mean of the displayed ICE curves and provides an overall reference.

## ALE Method

For ALE:

1. The observed range of `mean radius` is divided into quantile-based bins.
2. For every sample inside each bin, the feature is replaced by the lower and upper bin boundaries.
3. Predictions at both boundaries are obtained.
4. The prediction difference is calculated.
5. The average difference is computed for the bin.
6. These local effects are accumulated across bins.
7. The accumulated values are centered around zero.
8. The resulting ALE values are plotted against the bin centers.

This follows the ALE procedure specified in the supplied experiment instructions. fileciteturn0file0

### Interpretation

- ALE > 0 means the feature value in that region contributes toward a higher predicted probability relative to the centered ALE baseline.
- ALE < 0 means it contributes toward a lower predicted probability.
- A steep ALE segment indicates a strong local effect.
- A nearly horizontal segment indicates a weak local effect.
- Quantile bins make the analysis depend on observed data regions rather than evaluating arbitrary empty regions of the feature space.

## Outputs

### `outputs/metrics.json`

Contains:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

### `outputs/plots/ice_mean_radius.png`

Shows individual ICE curves and their mean.

### `outputs/plots/ale_mean_radius.png`

Shows the centered accumulated local effect of `mean radius`.

### `outputs/interpretation_summary.json`

Contains the feature range, number of ICE samples and points, ALE range, and bin information.

## Conclusion

A nonlinear Random Forest model was trained on tabular data and evaluated on unseen test samples. ICE provides an individual-level view of how changing `mean radius` changes the predicted probability while ALE provides an aggregated local-effect view across the observed feature distribution.

The two techniques complement each other: **ICE emphasizes individual prediction behavior and heterogeneity, whereas ALE summarizes local feature effects while restricting the analysis to regions supported by the data.**

## Experiment Requirements Covered

| Requirement | Implementation |
|---|---|
| Tabular data | Breast Cancer Wisconsin dataset |
| Preprocessing | Numerical feature handling + stratified split |
| Train/test split | 80/20 |
| Nonlinear model | Random Forest |
| Model evaluation | Accuracy, Precision, Recall, F1, ROC-AUC |
| ICE numerical feature | `mean radius` |
| ICE range | 2nd–98th percentile |
| ICE grid | 50 equally spaced values |
| ICE individuals | 60 test samples |
| ALE feature | `mean radius` |
| ALE intervals | 20 quantile bins |
| Local prediction differences | Lower vs upper bin boundary |
| ALE accumulation | Cumulative mean local effects |
| ALE centering | Centered around zero |
| Plots | ICE and ALE PNG outputs |
