# ICU Intubation ATE – TMLE Pipeline

Estimate the **Average Treatment Effect (ATE)** of **invasive intubation** vs **non‑invasive ventilation** for ICU patients using a Targeted Maximum Likelihood Estimation (TMLE) workflow. The outcome is binary (e.g., mortality/complication), with **train/test split**, **downsampling**, **overlap weighting**, and **calibrated propensity (g) models** to improve robustness and generalization.

---

## Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Data Format](#data-format)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage (Python API)](#usage-python-api)
- [Configuration](#configuration)
- [Features](#features)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [License](#license)

---

## Overview
This project implements a TMLE-based causal inference pipeline tailored to ICU data where treatment \(A\) is **invasive intubation (1)** or **non‑invasive (0)** and the binary outcome \(Y\) captures an adverse event or mortality. It:
- Splits data into **train/test** sets (stratified by treatment).
- Builds flexible **Super Learner** models for the outcome (Q) and propensity (g).
- **Calibrates** the g-model (Platt or Isotonic).
- Applies **overlap weighting** and trimming to stabilize estimates.
- Computes **ATE** (and ATT when available), standard errors, confidence intervals, and p-values.
- Produces rich **diagnostics** (AUC/F1/Brier, calibration curves, overfitting checks).

---

## Methodology
- **Q-model (Outcome regression)**: A stacked ensemble (`StackingClassifier`) with diverse base learners (regularized logistic regression, tree ensembles, SVM, MLP, kNN, Naive Bayes, etc.).  
- **g-model (Propensity)**: Learns \(P(A=1\mid W)\) with class balancing (downsampling), then **calibration** via:
  - `platt`: sigmoid scaling
  - `isotonic`: monotonic isotonic regression
- **Overlap weighting**: Uses \( \omega = g(1-g) \) with propensity trimming (e.g., keep 0.1–0.9 region) to mitigate extreme weights.
- **TMLE update**: Fluctuates initial Q using clever covariates derived from g to target ATE.
- **Diagnostics**: Train vs test metrics (AUC, F1, precision/recall, Brier), calibration curves and error, improvement after calibration, and generalization/overfitting assessment.

> Core implementation lives in `tmle.py` with primary entry point `tmle_project(...)`.

---

## Data Format
Provide a **CSV** with at least:
- `Y`: binary outcome (0/1)
- `A`: treatment indicator (1=invasive, 0=non‑invasive)
- `W...`: any number of baseline covariates (demographics, labs, vitals, scores, comorbidities, etc.)

Example header:
```text
Y,A,age,sex,sofa,lactate,spo2,rr,creatinine,prior_niv,...
```

---

## Installation
Python 3.9+ recommended.

```bash
# create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install dependencies
pip install -U pandas numpy statsmodels scikit-learn scipy tqdm matplotlib
pip install xgboost lightgbm
```

---

## Quick Start
```python
from tmle import tmle_project

results = tmle_project(
    file_path="data/tmle_data.csv",
    test_size=0.30,
    random_state=42,
    calibration_method="platt"   # or "isotonic"
)
```

> The script also contains an `if __name__ == "__main__":` example that runs `tmle_project(...)` with a CSV path; adapt it to your dataset.

---

## Usage (Python API)

### `tmle_project(file_path, test_size=0.3, random_state=42, calibration_method='platt') -> dict`
End‑to‑end run:
1. **Load & split** data (stratified by `A`).
2. **Preprocess W** (median impute + standardize on train; apply to test).
3. **Fit Super Learner** for Q and g (with **downsampling** for g).
4. **Calibrate g** with Platt/Isotonic; plot calibration curves.
5. **Compute overlap weights** and **TMLE ATE** (plus ATT if applicable).
6. **Print diagnostics** and return a results dictionary.

### Useful internal helpers (for reference)
- `get_base_learners(n_features)` – base models for the Super Learner.
- `fit_superlearner(X, y)` – stack/fit ensemble.
- `estimate_g_with_calibration(...)` – fit + calibrate propensity, return metrics/plots.
- `predict_Q_models(...)` – predict \(Q_0, Q_1, Q_A\).
- `compute_tmle(...)`, `update_Q(...)`, `estimate_fluctuation_param(...)`.
- `evaluate_calibration(...)` – calibration curve, calibration error, and Brier score.
- `print_train_test_comparison(...)`, `print_results(...)`, `diagnostic_checks(...)`.

---

## Configuration

Parameter | Description | Default
---|---|---
`file_path` | Path to CSV with `Y`, `A`, and covariates `W*`. | —
`test_size` | Test split fraction. | `0.30`
`random_state` | Seed for reproducibility. | `42`
`calibration_method` | `platt` or `isotonic` for g-model calibration. | `platt`

**Assumptions**
- `A` is the **last column** in the `W_A` design used for Q predictions in some internals.
- Downsampling is applied to the g-model training to address class imbalance.
- Propensity trimming for stability (e.g., keep 0.1–0.9 region) before overlap weighting.

---

## Features
- TMLE **ATE** (and ATT when available) with inference (SE, 95% CI, p-value).
- **Super Learner** with a diverse model zoo (LogisticCV L1/L2, Ridge/ElasticNet, RandomForest, ExtraTrees, GBDT, XGBoost, LightGBM, SVM, MLP, kNN, Naive Bayes, Decision Tree).
- **g-model calibration**: Platt/Isotonic + **calibration curves** and **Brier score** improvements.
- **Overlap weighting** and **propensity trimming**; detailed overlap diagnostics.
- **Train vs Test**: AUC/F1/precision/recall/Brier comparisons and **overfitting assessment**.

---

## Outputs

The run prints a rich report and returns a dict like (keys may vary slightly):

```python
{
  "ate_tmle": float,            # TMLE ATE
  "se": float,
  "ci_low": float,
  "ci_high": float,
  "p_value": float,
  "att": float or np.nan,
  "raw_ate": float,             # g-computation / un-targeted estimate
  "train_post_diagnostics": {   # AUC, F1, precision, recall, Brier, calibration info
     "q_auc": float, "g_auc": float, ...
  },
  "test_post_diagnostics": {...},
  "calibration_info": {
     "method": "platt"|"isotonic",
     "pre_calibration_metrics": {"calibration_error": ..., "brier_score": ...},
     "post_calibration_metrics": {"calibration_error": ..., "brier_score": ...}
  },
  "test_ratio": 0.30,
  "calibration_method": "platt"
}
```

Artifacts (printed/plots):
- Calibration curves (pre vs post) for the **downsampled training** g-model.
- Overlap summaries (min/max/mean propensities, fraction with 0.1–0.9 overlap).
- Train/Test performance tables and **overfitting flags**.

---

## Troubleshooting
- **Import errors (xgboost/lightgbm)**: Install `xgboost` and `lightgbm` as shown above or remove them from `get_base_learners`.
- **Columns not found**: Ensure CSV has `Y` and `A`. All other columns are treated as `W`.
- **Non-binary A or Y**: Convert to `0/1`.
- **Class imbalance**: The g-model uses downsampling internally; still consider reviewing treatment prevalence.
- **Poor calibration**: Try `calibration_method='isotonic'`. Inspect pre/post calibration error and Brier score.
- **Extreme propensities**: Check trimming thresholds and overlap diagnostics; widen inclusion or enrich covariates.

---

## Project Structure
```
.
├─ tmle.py        # End-to-end TMLE pipeline with SL, g-calibration, overlap weighting & diagnostics
└─ data/
   └─ tmle_data.csv  # Your ICU dataset (not included)
```

---

## Contributors
- [Your Name / Team / Lab]
- Clinical guidance: [Optional]
- Methodology review: [Optional]

---

## License
Add a license (e.g., MIT, Apache-2.0) to clarify reuse.

---

### Notes for Your ICU Study
- Map **A=1** to **invasive intubation** and **A=0** to **non‑invasive** consistently.
- Define the binary **Y** explicitly (e.g., in-hospital mortality, ICU mortality, ventilator-associated event) and the time horizon.
- Confirm inclusion/exclusion criteria and baseline window for covariates \(W\) (e.g., first 24h labs/vitals).
- Consider sensitivity analyses (different trimming windows, alternative learners, and calibration methods).
