# Naive Bayes (Gaussian) — MNIST

Gaussian Naive Bayes classifier for MNIST digits. For each class, the model estimates per-feature (pixel) mean and variance and predicts using log-posterior scoring with class priors.

## Method
- Priors: P(y=c) from class frequencies
- Likelihood (Gaussian): x_j | y=c ~ N(μ_cj, σ²_cj)
- Prediction:
  argmax_c [ log P(y=c) + Σ_j log N(x_j; μ_cj, σ²_cj) ]

Uses log-probabilities for numerical stability.

## Results
Run on MNIST with an 80/20 stratified split:
- Overall accuracy: **0.6339**
- Per-class accuracy:
  - 0: 0.8943
  - 1: 0.9448
  - 2: 0.4521
  - 3: 0.5623
  - 4: 0.3070
  - 5: 0.1409
  - 6: 0.9207
  - 7: 0.4435
  - 8: 0.6813
  - 9: 0.9173
Note: performance varies by digit due to pixel-distribution overlap; Gaussian assumptions per pixel are a rough fit for MNIST.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run:
   ```bash
   python train.py
