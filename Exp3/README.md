# Experiment 3: Model Interpretation using Random Forest and Partial Dependence

## Aim

To train a nonlinear machine-learning model using a tabular dataset and interpret its behavior using feature importance and a manually implemented Partial Dependence Plot (PDP).

The experiment also calculates the numerical sensitivity of the PDP, which represents the rate of change of the model's predicted probability with respect to the selected feature.

---

## 1. Objective

The objectives of this experiment are:

1. Load and preprocess a suitable tabular classification dataset.
2. Train a nonlinear machine-learning model.
3. Evaluate the trained model using standard classification metrics.
4. Determine the importance of individual features.
5. Select an important feature for interpretation.
6. Manually calculate its Partial Dependence values.
7. Plot the relationship between the selected feature and the model's average prediction.
8. Calculate the numerical sensitivity of the PDP.
9. Interpret how changes in the selected feature affect the model's prediction.

---

## 2. Dataset

The experiment uses the **Breast Cancer Wisconsin Diagnostic dataset**.

The dataset contains numerical measurements computed from digitized images of breast tissue samples.

Each sample belongs to one of two classes:

- **Malignant**
- **Benign**

The dataset contains:

- **569 samples**
- **30 numerical features**
- **2 target classes**

The features describe characteristics such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

The features are provided as mean, standard error, and worst-value measurements.

---

## 3. Machine Learning Model

### Random Forest

A Random Forest classifier is used as the nonlinear machine-learning model.

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees.

Each tree produces a prediction, and the predictions from multiple trees are combined to produce the final prediction.

Conceptually:

```text
                    Dataset
                       |
          +------------+------------+
          |            |            |
          v            v            v
       Tree 1       Tree 2       Tree 3
          |            |            |
          v            v            v
     Prediction   Prediction   Prediction
          |            |            |
          +------------+------------+
                       |
                       v
                Random Forest
                       |
                       v
                Final Prediction
```

Random Forest is suitable for this experiment because it can model nonlinear relationships between features and the target variable.

---

## 4. Why Model Interpretation is Required

A machine-learning model can produce accurate predictions, but accuracy alone does not explain how the model makes those predictions.

For example, a model may achieve high accuracy, but we may still want to know:

- Which features are important?
- How does changing a feature affect the prediction?
- Is the relationship positive or negative?
- Are there regions where the model is particularly sensitive to a feature?

Model interpretation techniques help answer these questions.

This experiment uses:

1. Feature Importance
2. Partial Dependence Plot
3. PDP Sensitivity

---

## 5. Feature Importance

Random Forest provides feature importance values that indicate how useful each feature was in the decision-making process of the forest.

Feature importance answers:

> **"Which features are most important to the model?"**

The generated feature importance plot shows that the most important features include:

1. `worst perimeter`
2. `worst area`
3. `worst concave points`
4. `mean concave points`
5. `worst radius`
6. `mean radius`
7. `mean perimeter`
8. `mean area`

The selected feature for the Partial Dependence analysis is:

```text
mean radius
```

The feature importance plot shows that `mean radius` is one of the more important features used by the Random Forest.

---

## 6. Partial Dependence Plot

### Definition

A Partial Dependence Plot (PDP) is a model interpretation technique that shows how the model's average prediction changes as one feature is varied.

In this experiment, the selected feature is:

```text
mean radius
```

For each selected value of `mean radius`:

1. A copy of the test dataset is created.
2. The `mean radius` column is replaced by the selected value.
3. The trained Random Forest predicts probabilities for all test samples.
4. The probabilities are averaged.
5. The average prediction is stored as the Partial Dependence value.

This process is repeated for multiple values of `mean radius`.

---

## 7. PDP Calculation

Let:

```text
R = [r1, r2, r3, ..., rK]
```

be the selected feature values.

For each value `r` in `R`, the feature is set to `r` for all test samples.

The model then produces:

```text
P1, P2, P3, ..., PN
```

where each value is the predicted probability for a test sample.

The Partial Dependence value is calculated as:

\[
PD(r) = \frac{1}{N}\sum_{i=1}^{N}P_i
\]

Therefore:

```text
Feature value
      |
      v
Modify test data
      |
      v
Random Forest prediction
      |
      v
Average predictions
      |
      v
Partial Dependence
```

---

## 8. Feature Range

The feature range was selected using the **5th percentile** to the **95th percentile** of the observed test data.

For `mean radius`, the range used was approximately:

```text
Lower value = 9.5868
Upper value = 20.5570
```

The range was divided into:

```text
50 equally spaced points
```

This produced the vector `R` used for the PDP calculation.

---

## 9. PDP Results

The Partial Dependence Plot for `mean radius` shows how the average predicted probability changes as `mean radius` is varied.

The initial part of the curve is approximately:

| Mean Radius | Partial Dependence |
|---:|---:|
| 9.5868 | 0.622751 |
| 9.8107 | 0.622751 |
| 10.0346 | 0.622751 |
| 10.2584 | 0.622751 |
| 10.4823 | 0.622663 |
| 10.7062 | 0.622663 |

The average predicted probability is approximately **0.623** at the lower end of the selected feature range.

The PDP remains relatively stable for lower values of `mean radius`.

A more noticeable decrease occurs approximately between:

```text
mean radius ≈ 14.5 to 15.5
```

The predicted probability decreases substantially in this region.

At higher values of `mean radius`, the PDP becomes approximately flat at around:

```text
0.561
```

---

## 10. Interpretation of the PDP

The PDP indicates that increasing `mean radius` does not have a constant effect throughout its entire range.

For lower values of `mean radius`, changes in the feature have relatively little effect on the model's average prediction.

Around the region:

```text
14.5 ≤ mean radius ≤ 15.5
```

the model becomes substantially more sensitive to changes in `mean radius`.

The average predicted probability decreases as `mean radius` increases through this region.

After approximately:

```text
mean radius > 16.5
```

the PDP becomes almost flat again.

This means that further increases in `mean radius` produce very little change in the average prediction in this region.

---

## 11. PDP Sensitivity

### Definition

Sensitivity describes how much the model's predicted outcome changes for a change in the selected feature.

For the PDP, sensitivity can be expressed as:

\[
Sensitivity(x) = \frac{dPD(x)}{dx}
\]

Since the PDP is available only at discrete points, the derivative is estimated numerically.

The experiment uses:

```python
np.gradient(PD, R)
```

to calculate the numerical derivative.

Therefore:

\[
Sensitivity \approx \frac{\Delta PD}{\Delta Feature}
\]

---

## 12. Meaning of Sensitivity

The sensitivity value tells us the local rate of change of the model's prediction.

### Positive Sensitivity

If:

```text
Sensitivity > 0
```

then increasing the feature increases the model's predicted probability.

### Negative Sensitivity

If:

```text
Sensitivity < 0
```

then increasing the feature decreases the model's predicted probability.

### Zero Sensitivity

If:

```text
Sensitivity ≈ 0
```

then changing the feature has little or no effect on the model's prediction in that region.

---

## 13. Sensitivity Results

The calculated sensitivity ranges approximately from:

```text
Minimum sensitivity = -0.08165
Maximum sensitivity =  0.00451
```

The strongest negative sensitivity occurs at approximately:

```text
mean radius = 15.1838
```

with:

```text
Sensitivity = -0.08165
```

This means that around this region, an increase of one unit in `mean radius` corresponds to an approximate decrease of **0.08165** in the model's average predicted probability, based on the local numerical slope.

In percentage-point terms, this is approximately:

```text
8.165 percentage points per unit increase
```

around that region.

The strongest positive sensitivity is approximately:

```text
Sensitivity = 0.00451
```

at approximately:

```text
mean radius = 14.2883
```

However, this positive sensitivity is very small compared with the strongest negative sensitivity.

---

## 14. Overall Sensitivity

The overall/secant sensitivity across the complete PDP range is approximately:

```text
-0.00565
```

It is calculated as:

\[
Sensitivity_{overall}
=
\frac{PD_{last}-PD_{first}}
{x_{last}-x_{first}}
\]

For this experiment:

```text
Feature range:
9.5868 → 20.5570

PDP:
approximately 0.62275 → 0.56077
```

Therefore:

\[
Sensitivity_{overall} \approx -0.00565
\]

This indicates that, when considering the entire feature range, the average predicted probability decreases as `mean radius` increases.

However, the overall sensitivity hides the fact that the relationship is nonlinear.

The local sensitivity plot provides more useful information about where the model is actually sensitive.

---

## 15. Why the Sensitivity Plot Contains Sharp Changes

The model used is a Random Forest.

A Random Forest consists of Decision Trees, and Decision Trees make decisions using threshold-based rules.

For example, a tree may contain rules similar to:

```text
mean radius <= 14.5
```

When the feature changes but remains on the same side of a threshold, the prediction may remain unchanged.

When the feature crosses important decision boundaries, the prediction can change.

Therefore, the PDP can contain:

- Flat regions
- Sudden decreases
- Small increases
- Sharp transitions

This explains the sensitivity plot.

In the generated result, the most prominent change occurs around:

```text
mean radius ≈ 14.5 – 15.5
```

where the sensitivity becomes strongly negative.

---

## 16. Important Terminology: PDP Sensitivity vs Classification Recall

The term "sensitivity" can have two different meanings in machine learning.

### Classification Sensitivity

In binary classification, sensitivity often means:

\[
Sensitivity =
\frac{TP}{TP+FN}
\]

This is also called:

```text
Recall
```

In this experiment, the classification recall is:

```text
Recall = 0.9722
```

### PDP Sensitivity

In the model interpretation part of this experiment, sensitivity means:

\[
\frac{dPD}{dFeature}
\]

It measures how quickly the model's predicted probability changes as the selected feature changes.

These are **different concepts**.

Therefore:

```text
Classification sensitivity
→ How well the classifier detects the positive class.

PDP sensitivity
→ How strongly the model's prediction changes
  with respect to a feature.
```

---

## 17. Model Evaluation Results

The trained Random Forest achieved the following results on the test set:

| Metric | Result |
|---|---:|
| Accuracy | 0.9561 |
| Precision | 0.9589 |
| Recall | 0.9722 |
| F1 Score | 0.9655 |
| ROC-AUC | 0.9940 |

The model therefore achieved approximately:

- **95.61% Accuracy**
- **95.89% Precision**
- **97.22% Recall**
- **96.55% F1 Score**
- **99.40% ROC-AUC**

These values indicate strong classification performance on the test data.

---

## 18. Confusion Matrix

The confusion matrix is:

```text
[[39,  3],
 [ 2, 70]]
```

Using the class ordering:

```text
Malignant
Benign
```

the matrix can be interpreted as:

| Actual / Predicted | Malignant | Benign |
|---|---:|---:|
| Malignant | 39 | 3 |
| Benign | 2 | 70 |

Therefore:

- 39 malignant samples were correctly classified.
- 3 malignant samples were classified as benign.
- 2 benign samples were classified as malignant.
- 70 benign samples were correctly classified.

The relatively small number of misclassifications contributes to the high accuracy and F1 score.

---

## 19. Classification Report

The classification results were:

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Malignant | 0.95 | 0.93 | 0.94 | 42 |
| Benign | 0.96 | 0.97 | 0.97 | 72 |
| **Accuracy** | | | **0.96** | **114** |
| Macro Avg | 0.96 | 0.95 | 0.95 | 114 |
| Weighted Avg | 0.96 | 0.96 | 0.96 | 114 |

The model performs well for both classes.

---

## 20. ROC-AUC

The ROC-AUC score is:

```text
0.9940
```

ROC-AUC measures how well the model separates the two classes across different classification thresholds.

A value close to `1.0` indicates strong class discrimination.

The obtained value of approximately **99.40%** indicates that the trained Random Forest has very strong discrimination between the two classes on the test set.

---

## 21. Complete Experiment Workflow

```text
                  Dataset
                     |
                     v
              Load Dataset
                     |
                     v
             Train/Test Split
                     |
          +----------+----------+
          |                     |
          v                     v
     Training Data          Test Data
          |                     |
          v                     |
    Random Forest                |
          |                     |
          v                     |
    Trained Model <-------------+
          |
          +-----------------------------+
          |                             |
          v                             v
     Model Evaluation             Feature Importance
          |                             |
          v                             v
 Accuracy / Precision             Important Features
 Recall / F1 / ROC-AUC                   |
                                          v
                                  Select mean radius
                                          |
                                          v
                                  Generate R values
                                          |
                                          v
                              Modify test feature values
                                          |
                                          v
                                 Random Forest
                                          |
                                          v
                                Average Predictions
                                          |
                                          v
                                         PDP
                                          |
                                          v
                                  Numerical Gradient
                                          |
                                          v
                                     Sensitivity
```

---

## 22. Generated Outputs

The experiment generates the following outputs:

```text
models/
└── random_forest.joblib

plots/
├── feature_importance.png
├── pdp.png
└── pdp_sensitivity.png

results/
├── metrics.json
└── pdp.csv
```

### `random_forest.joblib`

Contains the trained Random Forest model.

### `feature_importance.png`

Shows the relative importance of the 30 input features.

### `pdp.png`

Shows the relationship between `mean radius` and the model's average predicted probability.

### `pdp_sensitivity.png`

Shows the numerical derivative of the PDP:

```text
d(Predicted Probability) / d(mean radius)
```

### `metrics.json`

Stores the classification evaluation metrics.

### `pdp.csv`

Stores:

```text
mean radius
partial_dependence
sensitivity
```

for every point used in the PDP calculation.

---

## 23. Interpretation of the Complete Results

The Random Forest produced strong classification performance, with:

```text
Accuracy = 95.61%
F1 Score = 96.55%
ROC-AUC = 99.40%
```

Feature importance analysis identified several influential features, with `worst perimeter`, `worst area`, `worst concave points`, `mean concave points`, and `worst radius` among the most important. `mean radius` was also one of the more important features and was selected for further interpretation.

The PDP shows that the model's average predicted probability changes nonlinearly as `mean radius` varies. The most significant decrease occurs approximately between `14.5` and `15.5`.

The sensitivity plot confirms this observation.

The strongest negative local sensitivity occurs around:

```text
mean radius ≈ 15.18
```

where the numerical sensitivity is approximately:

```text
-0.08165
```

This means the model's average predicted probability is decreasing most rapidly with respect to `mean radius` around this region.

After approximately:

```text
mean radius > 16.5
```

the PDP becomes almost flat and the sensitivity approaches zero.

This indicates that the model is much more responsive to changes in `mean radius` in the middle portion of the selected range than at the extremes.

---

## 24. Conclusion

The experiment successfully demonstrates the interpretation of a nonlinear Random Forest classifier using feature importance and Partial Dependence Analysis.

The Random Forest achieved strong classification performance with an accuracy of approximately **95.61%**, an F1 score of **96.55%**, and a ROC-AUC of **99.40%**.

Feature importance analysis identified several influential features, with `mean radius` being one of the relatively important features. It was therefore selected for detailed interpretation.

The manually calculated Partial Dependence Plot showed that the model's average predicted probability changes nonlinearly as `mean radius` varies. The most significant decrease occurs approximately between `14.5` and `15.5`.

The numerical derivative of the PDP was used to calculate feature sensitivity. The strongest negative sensitivity was approximately **-0.08165** around a `mean radius` of **15.18**, indicating that the model's predicted probability changes most rapidly in this region.

The experiment demonstrates that model accuracy alone does not explain model behavior. Feature importance identifies which features are influential, while PDP and its numerical sensitivity provide additional insight into **how the model's prediction responds to changes in a specific feature**.

---

## 25. Key Takeaways

### Random Forest

An ensemble of Decision Trees capable of learning nonlinear relationships.

### Feature Importance

Shows which features are most useful to the trained model.

### PDP

Shows how the model's average prediction changes when one feature is varied.

### PDP Sensitivity

Measures the rate of change of the PDP with respect to the selected feature.

\[
\boxed{
Sensitivity =
\frac{dPD}{dFeature}
}
\]

### Main Observation

```text
mean radius
     |
     v
PDP decreases strongly
around 14.5 – 15.5
     |
     v
Strong negative sensitivity
     |
     v
PDP becomes almost flat
above approximately 16.5
```

Thus, the model is **not equally sensitive to `mean radius` across its entire range**.

---

## 26. How to Run

From the `Exp3` directory:

### Train the Random Forest

```bash
uv run train.py
```

### Evaluate the model

```bash
uv run evaluate.py
```

### Generate PDP and sensitivity analysis

```bash
uv run pdp.py
```

The final PDP-related outputs are saved under:

```text
plots/
results/
```

---

## 27. Experiment Summary

| Component | Result |
|---|---|
| Dataset | Breast Cancer Wisconsin Diagnostic |
| Problem | Binary Classification |
| Samples | 569 |
| Features | 30 |
| Model | Random Forest |
| Test Samples | 114 |
| Accuracy | 95.61% |
| Precision | 95.89% |
| Recall | 97.22% |
| F1 Score | 96.55% |
| ROC-AUC | 99.40% |
| PDP Feature | `mean radius` |
| PDP Points | 50 |
| PDP Range | 9.5868 – 20.5570 |
| Maximum PDP Sensitivity | 0.00451 |
| Minimum PDP Sensitivity | -0.08165 |
| Strongest Sensitivity Region | `mean radius ≈ 15.18` |
| Overall PDP Sensitivity | -0.00565 |
