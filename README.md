# IndPenSim Forecast UDA

**Predicting penicillin concentration across shifting industrial regimes — when your training data comes from a different world than your deployment.**

Industrial fermentation batches run under fundamentally different control strategies (recipe-driven, operator-controlled, advanced process control). A model trained on one regime fails on another. This project systematically explores *how far* classical ML, transformers, physics-informed models, and unsupervised domain adaptation can bridge that gap.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-f7931e) ![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2) ![uv](https://img.shields.io/badge/uv-package_manager-purple)

---

## The Problem

100 simulated penicillin fermentation batches ([IndPenSim](https://data.mendeley.com/datasets/pdnjz7zz5x)) operate under three control regimes. Models must predict penicillin concentration at arbitrary future times — but the source and target domains have different signal distributions.

| Batches | Control Mode | Role |
|---------|-------------|------|
| 1–30 | Recipe-driven | **Source** |
| 31–60 | Operator-controlled | Intermediate |
| 61–90 | APC / Raman-based | **Target** |

---

## Strategies Tested

A deliberate progression from simple to complex, each step answering a specific question:

- **EDA & Domain Analysis** — signal exploration, PCA/t-SNE visualization, domain classifier to quantify the shift
- **Classical ML Baselines** — Ridge regression and Random Forest with hyperparameter search; feature importance reveals substrate and acetic acid as dominant predictors
- **Gradient Boosted Trees** — XGBoost and LightGBM for regime classification, confirming the domain gap is real but moderate
- **PatchTST Transformer** — patch-based transformer encoder for time-series forecasting (~45k params)
- **Temporal Fusion Transformer** — probabilistic output (mu, sigma) with uncertainty quantification, variable selection networks, and interpretable temporal attention
- **Physics-Informed Modeling** — piecewise-log growth model (lag, logistic growth, linear decline) achieving R² > 0.999; hybrid PatchTST + PieceLog architecture
- **Unsupervised Domain Adaptation** — CORAL loss for covariance alignment of transformer feature representations across domains
- **Foundation Model Exploration** — Chronos for zero-shot time-series forecasting

---

## Project Structure

```
src/                        # Core modules
├── data_loader.py          # Batch loading and metadata
├── preprocessing.py        # Windowing, normalization, feature selection
├── feature_config.py       # 26 process signals configuration
├── domain_splits.py        # Source/target splits (control mode & k-means)
├── dataset.py              # PyTorch Dataset (fixed window)
├── tft_dataset.py          # TFT dataset (variable horizon)
├── baseline_model.py       # Ridge regression
├── rf_baseline.py          # Random Forest
├── transformer_model.py    # PatchTST encoder
├── tft_model.py            # Temporal Fusion Transformer
├── piecelog_model.py       # Parametric growth model
├── train.py / train_uda.py # Training loops (standard + CORAL)
├── coral_loss.py           # CORAL domain adaptation loss
└── domain_classifier.py    # Domain shift quantification

notebooks/                  # Step-by-step experiments (step0 → step5)
tests/                      # pytest suite (240+ tests)
```

---

## Quick Start

```bash
uv sync                     # Install dependencies
uv run pytest tests/        # Run tests
uv run jupyter notebook     # Launch notebooks
```

## References

- **IndPenSim**: Goldrick et al., *Journal of Biotechnology*, 2015. [Data](https://data.mendeley.com/datasets/pdnjz7zz5x)
- **Deep CORAL**: Sun & Saenko, *ECCV Workshops*, 2016
