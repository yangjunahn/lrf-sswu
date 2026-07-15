# Machine Learning for Seakeeping Control in Waves

This repository contains the research outputs developed by **Sungshin Women's University** as a joint research institution under the project supervised by **Seoul National University**, supported by the **Lloyd's Register Foundation**.

---

## Project Overview

- **Project Title:** Machine Learning for Seakeeping Control in Waves
- **Supported by:** Lloyd's Register Foundation
- **Lead Institution:** Seoul National University
- **Joint Research Institution:** Sungshin Women's University (School of AI Convergence)
- **Principal Investigator at SSWU:** Prof. Yangjun Ahn

---

## Research Scope & Key Contents

This research investigates artificial intelligence methods for ship motion time-series prediction and uncertainty quantification (UQ) to support safe seakeeping control under wave environments.

### 1. Theoretical Background
- Build an analysis foundation for ship motion in waves based on towing-tank model tests and numerical simulation data.
- Define analysis methods for nonlinear ship motion time-series data (including 6-DOF motions) in both frequency and time domains.

### 2. Deep Learning Models for Ship Motion Prediction
- Implement and compare modern deep learning architectures to achieve reliable short- and long-horizon prediction for complex ship motion time series.
- Main model components:
  - **Time-CNN:** A 2D convolutional neural network architecture designed to capture spatiotemporal features in multivariate time series.
  - **TimesNet:** A model that transforms one-dimensional time series into two-dimensional representations to learn multi-periodicity features.
  - **TCN (Temporal Convolutional Network):** A dilated-convolution model that extracts temporal patterns while preserving causality.

### 3. Uncertainty Quantification (UQ)
- Design methods to quantify aleatoric and epistemic uncertainty in AI model predictions for reliable and safety-critical control.
- Include studies on prediction intervals and confidence bands using Monte Carlo Dropout and Bayesian neural networks.

### 4. Data Sufficiency Analysis
- Analyze trade-offs between required data volume and prediction performance for wave-simulation analysis and model training.
- Examine minimum data requirements for reliable prediction under limited observation conditions.

---

## Repository Structure

```text
lrf-sswu/
├── README.md
├── ComrisonOfModels.ipynb   # Model comparison and analysis notebook
├── dataset/                 # Ship motion time-series data by sea condition
│   ├── 000_Tm_*_HS_*_V_*.csv|.out
│   ├── ...
│   └── Wave_180_Tm_*_HS_*_V_*.csv|.out
│       # Filename convention:
│       #   heading(deg)_Tm_(s)_HS_(m)_V_(knots)
│       #   heading: 0–180° (15° steps)
│       #   Tm: modal period, HS: significant wave height, V: ship speed
└── dataset_grid/            # Grid-based condition time-series data
    ├── 090_Tm_*_HS_*.csv|.out
    └── Wave_*_Tm_*_HS_*.csv|.out
```

### File / Folder Description

| Path | Description |
|------|-------------|
| `ComrisonOfModels.ipynb` | Notebook for comparing and analyzing ship motion prediction models (Time-CNN, TimesNet, TCN, etc.) |
| `dataset/` | 6-DOF motion time-series data by sea state (heading, Tm, Hs, V) in CSV/OUT format |
| `dataset_grid/` | Time-series data under grid-based simulation conditions in CSV/OUT format |
