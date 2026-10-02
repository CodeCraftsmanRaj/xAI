# Experiment 8 — LIME

## Student Information

| Field | Details |
|---|---|
| **Name** | Raj Kalpesh Mathuria |
| **UID** | 2023300139 |
| **Batch** | PE - C |
| **Experiment** | Experiment 8 — LIME |
| **Model** | ResNet-50 (pretrained ImageNet) |
| **Device** | CUDA / GPU |
| **Code Repository** | [Experiment_8_LIME](https://github.com/CodeCraftsmanRaj/xAI/tree/main/Experiment_8_LIME) |

## 1. Introduction

LIME (Local Interpretable Model-agnostic Explanations) is a post-hoc explainability technique that approximates a complex model locally with an interpretable surrogate model. For image classification, LIME divides an image into superpixels, creates perturbed versions by hiding selected regions, observes the model's predictions, and assigns importance weights to the regions that contribute to a selected class prediction.

This experiment applies LIME to a pretrained ResNet-50 ImageNet classifier. Three dog images with different backgrounds were evaluated. The experiment then deliberately varied the SLIC segmentation resolution to produce two reasonable but disagreeing explanations for one study image. Finally, deletion and insertion tests were performed using only the pretrained model's output probabilities to determine whether the model itself provides stronger evidence for either explanation.

## 2. Aim

> To explain prediction of a pre-trained model using LIME.

The experiment also investigates whether different reasonable LIME settings can produce substantially different explanations, and whether model-prediction-based tests can distinguish between them.

## 3. Objectives

1. Apply LIME to three images and visualize the important image regions.
2. Compare LIME explanations under different numbers of SLIC superpixels.
3. Identify two explanations that are individually reasonable but disagree substantially.
4. Define the decision criterion before evaluating the competing explanations.
5. Use only model predictions on modified images through deletion and insertion tests.
6. Determine whether the prediction-only evidence supports explanation A, explanation B, or neither.

## 4. Experimental Setup

| Component | Configuration |
|---|---|
| Model | ResNet-50 |
| Pretraining | ImageNet |
| Device | CUDA / GPU |
| LIME samples | 1200 per explanation |
| LIME features | 12 |
| Segmentation | SLIC |
| Base segmentation | 80 segments |
| Candidate segment counts | 40, 80, 120 |
| Study image | `dog1.jpg` |
| Target class | `golden retriever` |
| Prediction-test fractions | 10%, 20%, 30%, 40%, 50% |
| Baseline for perturbation | Blur |
| Decision margin | 0.05 |

## 5. Methodology

### 5.1 Model prediction
Each image was passed through pretrained ResNet-50. The top ImageNet predictions and their probabilities were recorded. The predicted class for all three study images was `golden retriever`.

### 5.2 LIME attribution
LIME generated superpixel-level explanations using 1200 perturbed samples. Positive regions were used to represent regions supporting the target prediction. Fidelity was measured using the local surrogate R² value.

### 5.3 Creating disagreement
For `dog1.jpg`, SLIC segmentation was varied while keeping the other major settings fixed. Candidate settings used 40, 80, and 120 segments. The pair with the largest disagreement while satisfying the configured reasonableness criteria was selected:

- **Explanation A:** 40 SLIC segments.
- **Explanation B:** 120 SLIC segments.

### 5.4 Pre-registered decision criterion
Before the prediction-only test, the criterion was specified as follows: A would be supported if its positive regions caused a larger target-probability decrease during deletion and/or retained a larger target probability during insertion than B, with the combined difference exceeding the 0.05 decision margin. B would be supported by the reverse pattern. Otherwise the result would be classified as neither/inconclusive.

This criterion avoids selecting an explanation merely because its visualization looks more intuitive.

### 5.5 Prediction-only validation
The selected positive regions were progressively deleted from the image and progressively inserted into a blurred baseline. At each fraction, only the model's target-class probability was measured. No ground-truth segmentation or external human label was used for the final comparison.

## 6. Predictions on the Three Images

| Image | Predicted class | Probability | LIME R² | Segments | Samples |
|---|---|---:|---:|---:|---:|

| dog1.jpg | golden retriever | 0.5142 (51.42%) | 0.6093 | 80 | 1200 |

| dog2.jpg | golden retriever | 0.4172 (41.72%) | 0.4979 | 80 | 1200 |

| dog3.jpg | golden retriever | 0.3979 (39.79%) | 0.6798 | 80 | 1200 |


All three images were classified as **golden retriever**. The probabilities were 0.5142, 0.4172, and 0.3979 respectively. The LIME local fidelity values were 0.6093, 0.4979, and 0.6798.

## 7. LIME Attribution Results

### dog1.jpg

![dog1 LIME attribution](images/dog1_lime.png)

### dog2.jpg

![dog2 LIME attribution](images/dog2_lime.png)

### dog3.jpg

![dog3 LIME attribution](images/dog3_lime.png)

The attribution maps show which superpixel regions LIME associates with the `golden retriever` prediction. Importantly, LIME is a local approximation: the highlighted regions describe the model's local behavior around the given image rather than proving that those pixels are the real-world reason a dog is a golden retriever.

## 8. Disagreement Experiment

The candidate explanations were:

| A segments | B segments | A R² | B R² | Pixel-mask Jaccard | Disagreement | Both reasonable? |
|---:|---:|---:|---:|---:|---:|---|

| 40 | 80 | 0.8242 | 0.6093 | 0.2659 | 73.41% | Yes |

| 40 | 120 | 0.8242 | 0.5374 | 0.2519 | 74.81% | Yes |

| 80 | 120 | 0.6093 | 0.5374 | 0.4903 | 50.97% | Yes |


The selected pair was **40 segments vs 120 segments**. Their pixel-mask disagreement was **74.81%**, with a Jaccard overlap of only **0.2519**. Both explanations met the configured reasonableness criterion, so this is a genuine example of explanation sensitivity to the segmentation setting.

### Explanation A — 40 segments
![Explanation A](images/explanation_A.png)

### Explanation B — 120 segments
![Explanation B](images/explanation_B.png)

### Direct comparison
![A vs B](images/A_vs_B.png)

## 9. Prediction-Only Test

For the selected pair, the measured summary was:

| Metric | Explanation A | Explanation B |
|---|---:|---:|
| Deletion AUC (probability drop) | 0.033839 | 0.058156 |
| Insertion AUC | 0.190840 | 0.169268 |
| Combined A − B support | -0.002746 | — |
| Decision margin | 0.050000 | — |
| Final decision | **Neither / inconclusive** | |

![Deletion test](images/deletion_test.png)

![Insertion test](images/insertion_test.png)

### Detailed prediction-test observations

| Fraction | A deletion P | B deletion P | A insertion P | B insertion P |
|---:|---:|---:|---:|---:|

| 10% | 0.4645 | 0.4006 | 0.4684 | 0.3819 |

| 20% | 0.4288 | 0.3886 | 0.4474 | 0.4121 |

| 30% | 0.4162 | 0.3357 | 0.5146 | 0.4471 |

| 40% | 0.4384 | 0.3384 | 0.4655 | 0.4382 |

| 50% | 0.4058 | 0.4246 | 0.4933 | 0.4087 |


The deletion test gives B a larger aggregate probability drop, while the insertion test gives A a larger aggregate insertion score. These effects point in opposite directions. More importantly, the combined A-minus-B support is **-0.002746**, which is far smaller in magnitude than the required **0.05** decision margin. Therefore, the experiment does not justify choosing either explanation.

## 10. What the Results Mean

### 10.1 The classifier prediction is stable across the three images
All three dog images were assigned the same ImageNet class, `golden retriever`, although confidence varied from 39.79% to 51.42%. This indicates that the pretrained classifier recognized the common object consistently despite the different image contexts.

### 10.2 LIME is sensitive to segmentation resolution
Changing only the number of SLIC segments changed the spatial support selected by LIME substantially. The 40-vs-120 comparison produced 74.81% pixel-mask disagreement. Thus, an explanation should not automatically be treated as uniquely determined just because it is produced by LIME.

### 10.3 Higher local fidelity does not automatically establish causal correctness
Explanation A had a higher local surrogate R² (0.8242) than B (0.5374). However, the subsequent prediction-only test did not provide sufficient evidence to declare A correct. Fidelity tells us how well the local surrogate approximates the sampled model behavior; it does not by itself prove that the highlighted pixels are the uniquely correct causal evidence.

### 10.4 The model-only validation is deliberately conservative
The prediction-only experiment resulted in **neither/inconclusive**. B produced a larger deletion probability drop, whereas A produced a larger insertion score. Because these signals disagree and the combined difference is much smaller than the pre-defined margin, selecting one explanation would not be justified by the experiment.

### 10.5 Explanation uncertainty is itself an important XAI result
The outcome demonstrates a useful lesson in explainable AI: explanations can depend on methodological choices such as segmentation granularity. When competing explanations disagree, a validation experiment should be used rather than assuming that the visually appealing map is correct.

## 11. Interpretation of LIME Fidelity

The local R² values were:

- `dog1.jpg`: **0.6093**
- `dog2.jpg`: **0.4979**
- `dog3.jpg`: **0.6798**
- Explanation A (`dog1`, 40 segments): **0.8242**
- Explanation B (`dog1`, 120 segments): **0.5374**

These values indicate how well the local surrogate represents the sampled model behavior around each image. The values are not classification accuracy and should not be interpreted as percentages of model correctness.

## 12. Limitations

1. The experiment uses a pretrained ImageNet ResNet-50 rather than a model trained specifically for this image set.
2. The ImageNet classifier's confidence is moderate rather than near certainty for the three images.
3. LIME is stochastic and local; different seeds or sampling choices can also affect explanations.
4. The prediction-only test evaluates agreement with the model's own behavior, not real-world causal truth.
5. The deletion/insertion baseline and perturbation mechanism can influence validation results.
6. A model can consistently use contextual or spurious visual cues; therefore, agreement with model predictions should not automatically be equated with semantic correctness.

## 13. Reproducibility

The experiment was executed with the modular project available in the repository below:

**Repository:** https://github.com/CodeCraftsmanRaj/xAI/tree/main/Experiment_8_LIME

Important generated result files include:

```text
results/
├── predictions.csv
├── summary.json
├── decision.md
├── decision_criterion.md
├── attributions/
│   ├── dog1_lime.png
│   ├── dog2_lime.png
│   └── dog3_lime.png
├── disagreement/
│   ├── explanation_A.png
│   ├── explanation_B.png
│   ├── A_vs_B.png
│   ├── deletion_test.png
│   └── insertion_test.png
└── tables/
    ├── three_image_lime.csv
    ├── disagreement_pairs.csv
    └── prediction_test.csv
```

The experiment uses CUDA when available and records the selected model, image, LIME settings, disagreement measurements, and prediction-only decision in the result files.

## 14. Conclusion

This experiment successfully applied LIME to a pretrained ResNet-50 ImageNet classifier and generated image-level attribution maps for three dog images. All three images were classified as `golden retriever`, with prediction probabilities between 0.3979 and 0.5142.

For the central disagreement study, changing the SLIC segmentation from 40 to 120 segments produced two individually reasonable LIME explanations with substantially different highlighted regions. The selected explanations disagreed over **74.81%** of the evaluated pixel mask, demonstrating that LIME explanations can be sensitive to segmentation granularity.

The experiment then used a pre-defined, model-only deletion/insertion test to compare the explanations. Explanation B produced the larger deletion probability drop, while Explanation A produced the larger insertion score. The combined support difference was only **-0.002746**, well below the **0.05** decision margin. Consequently, the correct experimental conclusion is **neither/inconclusive** rather than selecting A or B.

The main takeaway is that an explanation should be evaluated as an approximation of model behavior, not automatically accepted as ground-truth evidence. When reasonable explanations disagree, explicitly defined validation tests can reveal whether the model itself provides meaningful evidence for one explanation; in this experiment, it did not provide sufficiently strong evidence for either one.

## 15. Key Takeaways

- LIME can provide useful local visual explanations for image classifiers.
- The same prediction can yield substantially different explanations under different segmentation settings.
- Local fidelity is useful but is not equivalent to explanation correctness.
- Deletion and insertion tests can compare competing explanations using only model predictions.
- Conflicting validation signals should be reported rather than forced into a winner.
- For this experiment, the evidence supports an **inconclusive** conclusion between the two selected explanations.

---

**Student:** Raj Kalpesh Mathuria  
**UID:** 2023300139  
**Batch:** PE - C  
**Experiment:** 8 — LIME
