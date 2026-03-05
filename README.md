# Naive Bayes (Gaussian) — MNIST

Gaussian Naive Bayes classifier for MNIST digits. For each class, the model estimates per-feature (pixel) mean and variance and predicts using log-posterior scoring with class priors.

## Method
- Priors: P(y=c) from class frequencies
- Likelihood (Gaussian): x_j | y=c ~ N(μ_cj, σ²_cj)
- Prediction:
  argmax_c [ log P(y=c) + Σ_j log N(x_j; μ_cj, σ²_cj) ]

Uses log-probabilities for numerical stability.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run:
   ```bash
   python train.py