# Experiment 10 — LRP and LIT

**Objective:** Implement Layer-wise Relevance Propagation (LRP) from scratch for the provided small movie-sentiment neural network, compare it with Gradient × Input, study epsilon values, and complete the separate LIT interface activity from the handout.

## 1. Run the custom LRP experiment

Use Python in your existing `uv` project:

```bash
uv add numpy scikit-learn matplotlib
uv run lrp_starter.py
```

The script trains the 24-sentence / 8-unit model with a fixed random seed and prints:

- vocabulary size, hidden-unit count, training/test accuracy, and all parameter shapes;
- output logit, hidden activations, predicted sentiment, positive probability, and LRP-0 word relevances for `i love this movie`;
- LRP-0 vs Gradient × Input for a bias-free copy and for the original model with biases;
- relevance values for epsilon = 0, 0.1, 0.5, 1, and 2.

All artifacts are written to `results/`: `metrics.json`, CSV tables for word relevances / comparisons / epsilon sweep / test predictions, and two PNG attribution charts.

## 2. LRP implementation notes

The output relevance is initialized to the scalar output logit. Output-to-hidden redistribution uses `z_k = a_k * W2_k`; hidden-to-input redistribution uses `z_jk = x_j * W1_jk`. The denominator is the sum of the corresponding z values plus epsilon, and biases are excluded from redistribution denominators, matching the supplied handout. Zero denominators are handled safely to avoid NaN/inf values. Words absent from the sentence have input value zero and therefore receive zero relevance.

`Gradient × Input` is computed for the output logit as input value multiplied by the gradient. The script uses the same trained weights and explicitly zeros both bias vectors for the bias-free comparison; it does not retrain the model for that comparison.

## 3. Separate LIT (Language Interpretability Tool) task

The handout also asks you to run LIT on a real sentiment model, inspect predictions, edit words, and compare explanation methods. This is a separate interactive web UI activity and is not the same as the custom toy-model LRP calculation above. Do it in a separate environment so it does not interfere with this experiment's dependencies:

1. Create a separate folder/environment for the LIT demo.
2. Follow the official LIT installation and sentiment-model tutorial: <https://github.com/PAIR-code/lit> and <https://pair-code.github.io/lit/>. Use the current official setup instructions because LIT/model APIs and model download requirements can change.
3. Load a real pretrained sentiment model using the tutorial's supported model wrapper, then launch the LIT web app as instructed by the tutorial.
4. In the UI, open a positive and a negative example. Record the predicted label/score, inspect available explanation panels, edit or remove words, and compare how the prediction and explanations change.
5. Save screenshots showing (a) a prediction, (b) a changed sentence and its changed prediction, and (c) a comparison between explanation methods. Include a brief observation for each screenshot in your final report.

Do not claim LIT was run based only on this script; document it after you have actually opened the interface and captured your own results.

## 4. Files to send back for the report

After running, send the complete `results/` folder (zip it if convenient), plus your LIT screenshots and any notes from the interactive task. I can then create the experiment document with your actual outputs rather than guessed metrics.
