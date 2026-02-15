# IndPenSim Forecast UDA

Time-series forecasting playground for unsupervised domain adaptation on industrial penicillin fermentation data.

## Motivation

In IndSimPen the fermentation processes operates under fundamentally different control strategies — recipe-driven, operator-controlled, and advanced process control with Raman spectro. Forecasting models trained on one regime would fail on another because the underlying signal distributions shift.

This project is to experiment the *whether and how* domain adaptation is feasible on this kind of data. The notebook progression deliberately builds from exploratory analysis through baselines to transformer-based UDA, with each step answering a specific physical question rather than chasing a leaderboard metric. Is there a domain shift? Is that confounded by model properly set/trained? Do simple models already generalize? How much alignment in feature space actually help?

Here I try to keep focus on physically meaningful insights for efficient experimentation: piecewise+log growth modeling, feature importance rankings, domain shift quantification — matter more than MAE and fancy training.

## Dataset

**IndPenSim** — a high-fidelity simulation of industrial penicillin fermentation.

- 100 batches, 25 process signals, ~230 hours each, 0.2h sampling interval
- Source: [Mendeley Data](https://data.mendeley.com/datasets/pdnjz7zz5x) (stored locally in `data/Mendeley_data/`)
- Target variable: penicillin concentration P (g/L)

**Control mode split:**

| Batches | Control Mode | Role |
|---------|-------------|------|
| 1-30 | Recipe-driven | Source |
| 31-60 | Operator-controlled | Intermediate |
| 61-90 | APC / Raman-based | Target |
| 91-100 | Fault batches | Excluded |

## Approach — Notebook Progression

1. **Step 0 — EDA** (`step0_basic_eda.ipynb`, `step0_eda_modeling.ipynb`): Understand signal shapes, distributions, and visible domain differences in raw data.

2. **Step 1 — Domain Splits** (`step1_domain_splits.ipynb`): Define source/target via control mode metadata and k-means clustering on batch statistics; compare overlap between the two split strategies.

3. **Step 2 — Classification** (`step2_classification.ipynb`): Test whether simple features can distinguish concentration regimes using RF, XGBoost, and LightGBM classifiers.

4. **Step 3 — Baselines** (`step3_ridge_regress.ipynb`, `step3_rf_regress.ipynb`): Ridge and Random Forest regression establish a performance floor and reveal the domain gap (Ridge MAE: 1.68 on source, 57.7 on target).

5. **Step 4 — Transformers & Domain Shift** (`step4_patchTST_regress.ipynb`, `step4_piecelog_patchtst_regress.ipynb`, `step4_domain_shift.ipynb`): PatchTST encoder, piece-log parametric model, and quantification of shift via domain classifier accuracy and covariance distance.

6. **Step 5 — UDA & Adaptation** (`step5_coral_adaptation.ipynb`, `step5_split2_models.ipynb`): CORAL loss for feature alignment, TFT with probabilistic output and interpretable attention, evaluation of cross-domain generalization.

## Some notes & ideas from these tests (they keep coming..)

- Domain shift is there but moderate and train-dependent (~67% domain classifier accuracy on the control-mode split).
- K-means split captures clearer distributional shift than the metadata-based split.
- Random Forest generalizes better than Ridge across domains, showing robustness to covariate shift.
- Physics-informed piece-log model provides interpretable growth parameters with remarkable R².
- CORAL reduces target-domain error when applied to transformer feature representations.


## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/

# Launch notebooks
uv run jupyter notebook
```

## References

- **IndPenSim**: Goldrick et al., "The development of an industrial-scale fed-batch fermentation simulation," *Journal of Biotechnology*, 2015. Data available on [Mendeley Data](https://data.mendeley.com/datasets/pdnjz7zz5x).
- **Deep CORAL**: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation," *ECCV Workshops*, 2016.


## Project Structure

```
src/
├── data_loader.py          # Batch loading and metadata
├── preprocessing.py        # Windowing, normalization, feature selection
├── feature_config.py       # Feature groups and signal type metadata (26 features)
├── domain_splits.py        # Source/target assignment (control mode & clustering)
├── dataset.py              # PyTorch Dataset for fixed early-window prediction
├── tft_dataset.py          # TFT dataset with variable (T, D) sampling
├── baseline_model.py       # Ridge regression baseline
├── rf_baseline.py          # Random Forest baseline
├── rf_analysis.py          # RF feature importance analysis
├── transformer_model.py    # PatchTST-style encoder
├── tft_model.py            # Temporal Fusion Transformer (probabilistic)
├── piecelog_model.py       # Piece-log parametric growth model
├── piecelog_patchtst.py    # PatchTST + piece-log hybrid
├── train.py                # Standard training loop with early stopping
├── train_uda.py            # UDA training loop with CORAL loss
├── coral_loss.py           # CORAL loss for domain adaptation
└── domain_classifier.py    # Domain shift validation

notebooks/
├── step0_basic_eda.ipynb           # Exploratory data analysis
├── step1_domain_splits.ipynb       # Split strategy comparison
├── step2_classification.ipynb      # Regime classification
├── step3_ridge_regress.ipynb       # Ridge baseline
├── step3_rf_regress.ipynb          # Random Forest baseline
├── step4_patchTST_regress.ipynb    # PatchTST experiments
├── step4_domain_shift.ipynb        # Domain shift quantification
├── step5_coral_adaptation.ipynb    # CORAL UDA experiments
└── step5_split2_models.ipynb       # K-means split evaluation
```
