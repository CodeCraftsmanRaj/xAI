# Experiment 4: ICE and ALE Analysis of a Nonlinear Machine-Learning Model

## Student Details

| Field | Details |
|---|---|
| **Name** | Raj Mathuria |
| **UID** | 2023300139 |
| **Batch** | C |
| **Class / Division** | PE-C |
| **Experiment** | Experiment 4 – ICE and ALE |

## 1. Aim

To implement **Individual Conditional Expectation (ICE)** plots and **Accumulated Local Effects (ALE)** plots to analyze and interpret the effect of an input feature on the predictions of a trained nonlinear machine-learning model.

The experiment uses a tabular dataset, trains a nonlinear Random Forest classifier, evaluates it on unseen test data, and then uses ICE and ALE to understand the effect of **`mean radius`** on the model's predicted probability.

## 2. Dataset

The experiment uses the **Breast Cancer Wisconsin Diagnostic dataset** available through `scikit-learn`.

- Total samples: **569**
- Numerical input features: **30**
- Target classes: **malignant** and **benign**
- Training samples: **455**
- Testing samples: **114**
- Selected feature: **`mean radius`**

## 3. Methodology

```text
Dataset
   ↓
Data preprocessing
   ↓
Train/Test Split
   ↓
Random Forest Training
   ↓
Model Evaluation
   ↓
Select numerical feature
   ↓
ICE Analysis
   ↓
ALE Analysis
   ↓
Interpretation
```

### 3.1 Data Preparation

The dataset is loaded using `scikit-learn`. Since the input features are numerical, categorical encoding is not required. A tree-based Random Forest does not require feature scaling for training.

The data is divided into training and testing subsets using a stratified split.

## 4. Nonlinear Model

A **Random Forest Classifier** is used as the nonlinear model.

A Random Forest combines predictions from multiple decision trees:

```text
                 Random Forest
                       |
          +------------+------------+
          |            |            |
        Tree 1       Tree 2       Tree 3
          |            |            |
       Prediction   Prediction   Prediction
          |            |            |
          +------------+------------+
                       |
                Final prediction
```

The model is trained once and remains fixed during ICE and ALE analysis.

## 5. Model Evaluation

The trained Random Forest was evaluated on **114 unseen test samples**.

| Metric | Result |
|---|---:|
| Accuracy | **94.74%** |
| Precision | **95.83%** |
| Recall | **95.83%** |
| F1 Score | **95.83%** |
| ROC-AUC | **99.40%** |

Confusion matrix:

```text
[[39, 3],
 [ 3, 69]]
```

The strong accuracy and ROC-AUC indicate that the model performs well on the test data, providing a suitable model for subsequent interpretation.

## 6. Individual Conditional Expectation (ICE)

### 6.1 Concept

**ICE shows how the prediction for an individual sample changes when one feature is varied while all other features are kept unchanged.**

The selected feature is:

```text
mean radius
```

For each selected test sample, `mean radius` is replaced by a sequence of values while the other features remain fixed. The trained Random Forest predicts the probability for every modified sample.

Each individual produces one ICE curve.

### 6.2 ICE Algorithm

1. Select `mean radius`.
2. Determine an appropriate range from the observed feature distribution.
3. Generate equally spaced values.
4. Copy the selected test samples.
5. Replace `mean radius` with one grid value.
6. Obtain model predictions.
7. Store the predictions.
8. Repeat for all grid values.
9. Repeat for multiple test samples.
10. Plot feature values against predicted probabilities.

For this experiment:

- ICE grid points: **50**
- Samples plotted: **60**
- Feature range: approximately **9.07 to 21.46**

### 6.3 Interpretation of ICE Result

The ICE plot contains many thin individual curves and a thick **Mean ICE** curve.

A major observation is the **sharp downward transition around `mean radius ≈ 14.5–15`**. Before this region, many curves remain relatively stable. Around 15, a large number of curves drop and then become comparatively flat.

The curves are spread over a large probability range, from values close to 0 to values close to 1. This demonstrates that the effect of `mean radius` is **not identical for every individual**.

Some samples remain at high predicted probabilities, some remain at low probabilities, and others show stronger changes around the transition region.

The mean ICE curve is approximately in the **0.58–0.64** range. Therefore, the average change across the displayed individuals is relatively modest compared with the large differences between individual curves.

### Main ICE conclusion

> **`mean radius` has heterogeneous effects across individuals, with a particularly noticeable change in many individual predictions around `mean radius ≈ 15`.**

## 7. Accumulated Local Effects (ALE)

### 7.1 Concept

**ALE measures the average local change in model prediction across different regions of a feature's observed distribution.**

Instead of directly evaluating arbitrary feature values, ALE divides the observed feature distribution into intervals and measures the prediction change between the lower and upper boundaries of each interval.

### 7.2 ALE Algorithm

1. Select `mean radius`.
2. Divide its observed distribution into bins.
3. Identify samples belonging to each bin.
4. Replace the feature with the lower bin boundary and predict.
5. Replace the feature with the upper bin boundary and predict.
6. Calculate the difference:

```text
Local Effect =
Prediction at upper boundary
-
Prediction at lower boundary
```

7. Average the local effects within the bin.
8. Accumulate the average effects across bins.
9. Center the accumulated effects around zero.
10. Plot the resulting ALE values against the feature values.

The experiment uses **20 ALE bins**.

## 8. Interpretation of ALE Result

The ALE plot shows three important regions.

### Region 1: Approximately 8.5–10.5

The ALE value is positive, reaching approximately **+0.014**.

This indicates a positive local contribution relative to the centered ALE baseline.

### Region 2: Approximately 10.5–14

The ALE value gradually decreases but remains positive, approximately around **+0.012 to +0.013**.

This represents a relatively stable region with only small changes.

### Region 3: Approximately 14–15

This is the **most important region**.

The ALE curve shows a sharp downward transition around **14.5–15**, changing from positive values to approximately **−0.02**.

This indicates that the Random Forest is particularly sensitive to changes in `mean radius` in this region.

### Region 4: Above approximately 15.5

The ALE curve becomes almost horizontal and remains around **−0.022 to −0.023**.

This means that after the major transition, additional changes in `mean radius` produce comparatively little additional local effect according to the ALE analysis.

### Main ALE conclusion

> **The strongest local effect of `mean radius` occurs around 14.5–15, after which the model response becomes comparatively stable.**

## 9. ICE and ALE Comparison

| Observation | ICE | ALE |
|---|---|---|
| Main purpose | Individual effects | Average local effect |
| Individual variation | Clearly visible | Averaged |
| Important transition | Around 15 | Around 14.5–15 |
| Higher feature values | Curves largely stabilize | ALE becomes nearly flat |
| Main insight | Heterogeneous individual behavior | Strong local effect around the transition |

Both methods identify a consistent transition region around **`mean radius ≈ 14.5–15`**.

ICE additionally shows that different individuals have very different baseline prediction levels and responses, while ALE compresses this behavior into a single interpretable local-effect curve.

## 10. Overall Interpretation

The Random Forest achieved strong predictive performance on unseen test data, with **94.74% accuracy and approximately 99.40% ROC-AUC**.

The ICE analysis demonstrates that `mean radius` does not affect all individuals identically. The individual curves occupy a wide range of prediction probabilities, showing clear heterogeneity. Many curves exhibit a noticeable downward transition around `mean radius ≈ 15`.

The ALE analysis supports this finding. The ALE curve remains positive at lower feature values, gradually changes through the middle region, and then shows its strongest downward transition around **14.5–15**. Above approximately **15–16**, the curve becomes nearly flat.

Therefore, the two interpretation methods provide complementary evidence:

- **ICE:** shows how the feature affects individual predictions.
- **ALE:** shows the average local effect across the observed feature distribution.
- **Both:** identify approximately **14.5–15** as the most important transition region for `mean radius`.

### Important note about ALE magnitude

The ALE values are centered relative effects, not raw prediction probabilities. Therefore, their small numerical range (approximately **−0.023 to +0.014**) should not be interpreted as meaning that the feature is unimportant.

The important information is the **shape of the curve and the sharp change around 14.5–15**.

## 11. Conclusion

The experiment successfully implemented **Individual Conditional Expectation (ICE)** and **Accumulated Local Effects (ALE)** for interpreting a nonlinear Random Forest model trained on tabular data.

The Random Forest achieved **94.74% accuracy, 95.83% precision, 95.83% recall, 95.83% F1-score, and approximately 99.40% ROC-AUC** on the test set.

ICE demonstrated that the effect of `mean radius` varies substantially between individual samples. A prominent change is visible around `mean radius ≈ 15`.

ALE provided an aggregated local interpretation and identified the same region as the strongest transition. The ALE curve changes sharply around **14.5–15** and becomes comparatively stable at larger values.

Thus:

> **ICE is useful for understanding individual prediction behavior, while ALE summarizes the local effect of a feature across its observed distribution. In this experiment, both methods identify `mean radius ≈ 14.5–15` as a particularly important region of model behavior.**

## 12. Viva Points

### What is ICE?

ICE shows the effect of changing one feature on individual predictions while keeping the other features fixed.

### What is ALE?

ALE calculates local prediction differences within feature intervals, averages them, accumulates them, and centers the result to show the feature's local effect.

### Why use a nonlinear model?

The experiment requires interpretation of a nonlinear model. Random Forest is a suitable nonlinear model for tabular data.

### Which feature was selected?

`mean radius`.

### How many ICE points were used?

50 equally spaced feature values.

### How many individual ICE curves were plotted?

60 test samples.

### How many ALE bins were used?

20 bins.

### What is the most important result?

Both ICE and ALE indicate a strong change in model behavior around **`mean radius ≈ 14.5–15`**.

### Was the model retrained during ICE/ALE?

**No.** The Random Forest was trained once. ICE and ALE only modify input values and pass the modified samples through the already-trained model.
