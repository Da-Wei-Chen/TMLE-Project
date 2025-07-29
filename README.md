# TMLE Project

A Python implementation of **Targeted Maximum Likelihood Estimation (TMLE)** for binary outcomes, leveraging a **Super Learner ensemble framework** to estimate causal effects.

##  Overview

This project evaluates **invasive mechanical ventilation** by estimating:

- **Average Treatment Effect (ATE)**  
- **Average Treatment Effect on the Treated (ATT)**

Key methods include:

- Ensemble modeling (`Super Learner`) for outcome (`Q`) and propensity score (`g`) models  
- Data preprocessing (imputation, standardization)  
- Diagnostic checks before and after TMLE updating  
- Downsampling for balanced propensity estimation  
- Comprehensive performance and influence-function diagnostics

##  Features

- Flexible **Super Learner** with Logistic Regression, Random Forest, Gradient Boosting, SVM, MLP, and XGBoost  
- Built-in parameter tuning for stable convergence  
- Propensity score downsampling for robustness  
- Automated ATE and ATT estimation with diagnostics  

##  Installation

```bash
git clone <repository-url>
cd tmle-project
pip install -r requirements.txt
```

##  Usage

```python
from tmle_project import tmle_project

tmle_project("data.csv")
```

Input data must include:

- `Y`: binary outcome  
- `A`: binary treatment indicator  
- Covariates: any number of predictive features

##  Dependencies

- pandas, numpy, statsmodels  
- scikit-learn, xgboost  
- scipy, tqdm

##  Contributors

- Da-Wei Chen

