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

## Public Materials and Documentation

The final documented release provides two Python training implementations for 6-DOF ship-motion forecasting, their theory explanation, and detailed user documentation. The public dissemination overview introduces the materials for readers of the research centre's website.

| Material | Online source / description | Downloadable formats |
|---|---|---|
| Public dissemination overview | A three-page English overview of purpose, contents, formats, access, use, and references | [PDF](LRF_SSWU_Public_Dissemination_Overview.pdf) · [Word](LRF_SSWU_Public_Dissemination_Overview.docx) |
| BiLSTM final training code | [py_BiLSTM_std.py](py_BiLSTM_std.py): bidirectional LSTM encoder with a single-shot forecast head | Python source |
| Causal ModernTCN final training code | [py_modernTCN_std.py](py_modernTCN_std.py): causal convolutional backbone with a temporal projection head | Python source |
| Theory and previous experiments | [LRF-SSWU-CodeDocumentation_Theory.pdf](LRF-SSWU-CodeDocumentation_Theory.pdf): 9-page explanation of architectures, preprocessing, optimization, metrics, and historical results | PDF |
| English code review and user manual | [Read the English manual](BiLSTM_ModernTCN_Code_Review_and_User_Manual.md) | [PDF](BiLSTM_ModernTCN_Code_Review_and_User_Manual.pdf) · [Word](BiLSTM_ModernTCN_Code_Review_and_User_Manual.docx) |
| Korean code review and user manual | [Read the Korean manual](BiLSTM_ModernTCN_Code_Review_and_User_Manual-kor.md) | [PDF](BiLSTM_ModernTCN_Code_Review_and_User_Manual-kor.pdf) |
| Additional ModernTCN analysis | [Causal_ModernTCN_6DOF_Ship_Motion_Report.md](Causal_ModernTCN_6DOF_Ship_Motion_Report.md) | Markdown |
| Comparison notebook | [ComrisonOfModels.ipynb](ComrisonOfModels.ipynb) | Jupyter notebook |
| Research data | [dataset/](dataset/) and [dataset_grid/](dataset_grid/) | CSV and OUT |

### Reading and Using the Release

1. Read the [public dissemination overview](LRF_SSWU_Public_Dissemination_Overview.pdf) for the purpose and organisation of the released materials.
2. Read the [theory PDF](LRF-SSWU-CodeDocumentation_Theory.pdf) for the forecasting formulation and model architectures.
3. Follow the [English manual](BiLSTM_ModernTCN_Code_Review_and_User_Manual.md#9-environment-and-data-preparation-manual) or [Korean manual](BiLSTM_ModernTCN_Code_Review_and_User_Manual-kor.md#9-환경-및-데이터-준비-매뉴얼) for installation, preflight checks, training, interruption/resumption, and saved-model inference.

The default task uses 128 observed samples to predict the next 128 samples of surge, sway, heave, roll, pitch, and yaw at an assumed 2 Hz. This corresponds to future lead times of 0.5–64 seconds after the last observation. Both models predict the complete future window in one pass. Optuna selects hyperparameters using validation loss; a newly initialized model is then trained on the original training split, with validation retained for checkpoint selection.

### Data Preparation and Execution

The final scripts expect motion CSV files under `csv_dataset/` by default, with `x1`–`x6` in the documented order and units (m for translations; deg for rotations). Prepare finite, chronologically ordered 2 Hz data and verify file-level split independence. Files beginning with `Wave_` are excluded. Each valid training file needs at least 256 rows, and the training split needs enough windows for the chosen batch sizes.

The published `dataset/` and `dataset_grid/` folders contain exported CSV/OUT files rather than a prepared `csv_dataset/` directory. Inspected example timestamps advance by 0.5 seconds in `dataset/` and 0.1 seconds in `dataset_grid/`; verify every source's timing and column mapping and resample where necessary before using a 2 Hz configuration. Exported headers can differ from the `x1`–`x6` names expected by the manual's stricter preflight check.

After completing the manual's environment and data preparation steps, run from the repository root:

```bash
MPLBACKEND=Agg python -u py_BiLSTM_std.py
MPLBACKEND=Agg python -u py_modernTCN_std.py
```

The scripts use CUDA when available and CPU otherwise. Settings are edited in the source; these training scripts do not provide command-line configuration flags. The default result directories are:

```text
experiments0917/BiLSTM_SingleShot_optuna_in128_out128_stride16/
experiments0904/CausalModernTCN_optuna_in128_out128_stride16/
```

Outputs include best weights, scaler and split lists, hyperparameters, an Optuna SQLite database, training history, test metrics, and plots. Trained weights are generated by running the pipeline and are not part of the documented release.

### Implementation Status and Reproducibility

The two final source scripts are preserved as submitted at code baseline [`31988e9`](https://github.com/yangjunahn/lrf-sswu/commit/31988e994fe6d35e1308760700ecbd2e4f3ec647). The [review findings](BiLSTM_ModernTCN_Code_Review_and_User_Manual.md#8-review-findings-and-improvement-priorities) describe limitations and proposed changes, including unequal-batch metric aggregation and ModernTCN's historical interruption handling. Those changes have not been applied to the final scripts.

Documentation checks used reduced synthetic runs. Historical results in the theory PDF are separate from those checks; the full real-data hyperparameter search was not newly reproduced during documentation preparation. The final scripts produce point forecasts. They do not implement uncertainty intervals or an operational ship-control system.

For reproducible use, preserve the code commit, data identification and preprocessing, actual train/validation/test file lists, scaler, settings, and software environment. Cite the institution, repository URL, document title, and the revision or access date used.

---

## Broader Research Scope and Context

The following describes the wider project context. The two final scripts linked above specifically implement BiLSTM and Causal ModernTCN forecasting; broader topics are not a claim that every method below is implemented in those scripts.

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
├── LRF_SSWU_Public_Dissemination_Overview.pdf
├── LRF_SSWU_Public_Dissemination_Overview.docx
├── LRF-SSWU-CodeDocumentation_Theory.pdf
├── py_BiLSTM_std.py
├── py_modernTCN_std.py
├── BiLSTM_ModernTCN_Code_Review_and_User_Manual.md
├── BiLSTM_ModernTCN_Code_Review_and_User_Manual-kor.md
├── BiLSTM_ModernTCN_Code_Review_and_User_Manual.pdf
├── BiLSTM_ModernTCN_Code_Review_and_User_Manual-kor.pdf
├── BiLSTM_ModernTCN_Code_Review_and_User_Manual.docx
├── Causal_ModernTCN_6DOF_Ship_Motion_Report.md
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
| `LRF_SSWU_Public_Dissemination_Overview.pdf` / `.docx` | Three-page English overview prepared for public dissemination |
| `LRF-SSWU-CodeDocumentation_Theory.pdf` | Theory, architectures, training/evaluation procedure, and previous experiments |
| `py_BiLSTM_std.py` / `py_modernTCN_std.py` | Final documented forecasting and hyperparameter-search implementations |
| `BiLSTM_ModernTCN_Code_Review_and_User_Manual*` | English/Korean code review and user manuals in the formats listed above |
| `Causal_ModernTCN_6DOF_Ship_Motion_Report.md` | Additional analysis of the convolutional forecasting implementation |
| `ComrisonOfModels.ipynb` | Notebook for comparing and analyzing ship motion prediction models (Time-CNN, TimesNet, TCN, etc.) |
| `dataset/` | 6-DOF motion time-series data by sea state (heading, Tm, Hs, V) in CSV/OUT format |
| `dataset_grid/` | Time-series data under grid-based simulation conditions in CSV/OUT format |
