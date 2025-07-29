import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline  
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import StackingClassifier 
from scipy.stats import norm
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings
from tqdm import tqdm
import time
from sklearn.utils import resample

############################################################################
# Base Models
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
############################################################################

def tmle_project(file_path):
    def data_loading(file_path):
        df = pd.read_csv(file_path)
        Y = df["Y"].values
        A = df["A"].values
        W = df.drop(columns=["Y", "A"])
        W_A = df.drop(columns=["Y"])
        return Y, A, W, W_A

    def data_preprocessing(W):
        '''data preprocessing function'''

        print(f"<< shape of raw data>>: {W.shape}")
        print(f"<< missing values >>: {W.isnull().sum().sum()}")
        W_clean = W.fillna(W.median())
        scaler = StandardScaler()
        W_standardized = scaler.fit_transform(W_clean)
        return pd.DataFrame(W_standardized, columns=W.columns), scaler

    def get_base_learners(n_features):
        '''base learners with improved parameters'''
        base_learners = [
            # Linear models - increase max_iter for convergence
            ('logistic', LogisticRegressionCV(cv=5, max_iter=10000, random_state=42, 
                                            solver='lbfgs')),  
            ('logistic_l1', LogisticRegressionCV(cv=5, max_iter=10000, penalty='l1', 
                                               solver='liblinear', random_state=42, 
                                               tol=1e-3)),  

            # Tree-based models - adjust parameters to avoid overfitting
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, # decrease depth
                                        min_samples_split=20,
                                        min_samples_leaf=10,
                                        random_state=42)),
            ('gbm', GradientBoostingClassifier(n_estimators=100, max_depth=4,  # decrease depth
                                             learning_rate=0.05,
                                             subsample=0.8,
                                             random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=4, 
                                 learning_rate=0.05,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 random_state=42, 
                                 eval_metric='logloss',
                                 verbosity=0)),  # decrease verbosity
            
            # non-linear models - adjust parameters for better generalization
            ('svm', make_pipeline(StandardScaler(), 
                                SVC(probability=True, kernel='rbf', 
                                   C=1.0, gamma='scale',  # use default parameters
                                   random_state=42))),
            ('mlp', make_pipeline(StandardScaler(),
                                MLPClassifier(hidden_layer_sizes=(30, 15),  # reduce network size
                                            max_iter=2000,  # increase iterations
                                            alpha=0.01,     # increase regularization
                                            random_state=42)))
        ]
        return base_learners

    def fit_superlearner(X, Y, base_learners, model_name="SuperLearner"):
        print(f"\n << Fitting {model_name} >>")

        
        pbar = tqdm(total=len(base_learners) + 2, desc=f"<< Training {model_name} >>", leave=False)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            meta_learner = LogisticRegressionCV(cv=3, max_iter=5000, 
                                              random_state=42, solver='lbfgs')
            
            pbar.set_description(f"Building {model_name}")
            pbar.update(1)
            
            sl = StackingClassifier(
                estimators=base_learners,
                cv=3,
                stack_method='predict_proba',
                final_estimator=meta_learner,
                n_jobs=-1  # use all available cores    
            )
            
            try:
                pbar.set_description(f"<< Fitting {model_name} >>   ")
                sl.fit(X, Y)
                pbar.update(1)

                # Evaluate model performance
                pbar.set_description(f"<< Evaluating {model_name} >>")
                cv_scores = cross_val_score(sl, X, Y, cv=3, scoring='roc_auc')
                print(f"{model_name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Warning: {model_name} fitting failed: {str(e)}")
                print("Using simplified model...")
                # downgrade to simple model
                from sklearn.linear_model import LogisticRegression
                sl = LogisticRegression(max_iter=5000, random_state=42)
                sl.fit(X, Y)
            
            finally:
                pbar.close()
        
        return sl

    def predict_Q_models(sl, W_A_standardized, A):
        """Predict Q models with improved stability"""
        try:
            Q_A = sl.predict_proba(W_A_standardized)[:, 1]
        except:
            Q_A = sl.predict_proba(W_A_standardized)[:, 1]
        
        W_A1 = W_A_standardized.copy()
        # Predict Q1 and Q0
        W_A1['A'] = 1
        Q_1 = sl.predict_proba(W_A1)[:, 1]
        W_A0 = W_A_standardized.copy()
        W_A0['A'] = 0
        Q_0 = sl.predict_proba(W_A0)[:, 1]
        return Q_A, Q_1, Q_0

    def estimate_g(A, W_standardized, base_learners):
        """Estimation of propensity scores"""
        print("\n <<Estimating propensity scores >>")
        # ===== 🔻 1. Downsampling  =====
        treated_idx = A == 1
        control_idx = A == 0
        W_treated = W_standardized[treated_idx]
        W_control = W_standardized[control_idx]
        A_treated = A[treated_idx]
        A_control = A[control_idx]

        if len(W_treated) > len(W_control):
            W_treated_down = resample(W_treated, replace=False, n_samples=len(W_control), random_state=42)
            A_treated_down = resample(A_treated, replace=False, n_samples=len(W_control), random_state=42)
            W_down = np.vstack([W_treated_down, W_control])
            A_down = np.concatenate([A_treated_down, A_control])
        else:
            W_control_down = resample(W_control, replace=False, n_samples=len(W_treated), random_state=42)
            A_control_down = resample(A_control, replace=False, n_samples=len(W_treated), random_state=42)
            W_down = np.vstack([W_treated, W_control_down])
            A_down = np.concatenate([A_treated, A_control_down])

        print(f"g-model training samples (downsampled): {W_down.shape}, A=1 proportion: {np.mean(A_down):.2f}")

        # ===== 🔺 2. Train g-model with downsampled data =====
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            g_model = fit_superlearner(W_down, A_down, base_learners, "Propensity Score Model")

        # ===== 🔁 3. Predict propensity scores for all samples =====
        g_w = g_model.predict_proba(W_standardized)[:, 1]

        # Trim propensity scores to avoid extreme values
        g_w_trimmed = np.clip(g_w, 0.025, 0.975)

        # Compute clever covariates
        H_1 = A / g_w_trimmed
        H_0 = (1 - A) / (1 - g_w_trimmed)

        # Print diagnostics
        print(f"Propensity Score: min={g_w.min():.4f}, max={g_w.max():.4f}, mean={g_w.mean():.4f}")
        print(f"Propensity Score (trimmed): min={g_w_trimmed.min():.4f}, max={g_w_trimmed.max():.4f}")

        return g_model, g_w_trimmed, H_1, H_0

    def estimate_fluctuation_param(Y, Q_A, H_1, H_0, A):
        """Estimation of fluctuation parameters"""
        # Ensuring Q_A is a valid probability vector
        Q_A_clipped = np.clip(Q_A, 1e-6, 1 - 1e-6)
        logit_QA = np.log(Q_A_clipped / (1 - Q_A_clipped))

        # Construct clever covariate
        H_A = A * H_1 - (1 - A) * H_0
        H_A = H_A.reshape(-1, 1) if H_A.ndim == 1 else H_A

        # Using GLM to estimate fluctuation parameters
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = sm.GLM(Y, H_A, offset=logit_QA, family=sm.families.Binomial()).fit()
                eps = model.params[0]
                print(f"Fluctuation parameter(epsilon): {eps:.6f}")
        except Exception as e:
            print(f"GLM fitting failed, using fallback method: {str(e)}")
            eps = 0.0
            
        return eps

    def update_Q(Q_base, H, eps):
        """Update Q values using the fluctuation parameter"""
        Q_clipped = np.clip(Q_base, 1e-6, 1 - 1e-6)
        logit_Q = np.log(Q_clipped / (1 - Q_clipped))
        updated_Q = 1 / (1 + np.exp(-(logit_Q + eps * H)))
        return np.clip(updated_Q, 1e-6, 1 - 1e-6)

    def compute_tmle(Y, A, Q_A_update, Q_1_update, Q_0_update, H_1, H_0):
        """Compute TMLE estimates and diagnostics"""
        # Computing ATE
        ate = np.mean(Q_1_update - Q_0_update)
        
        # ATT = E[Q1 - Q0 | A = 1]
        att = np.mean((Q_1_update - Q_0_update)[A == 1]) if np.any(A == 1) else np.nan

        # Computing influence function
        H_A = A * H_1 - (1 - A) * H_0
        infl_fn = H_A * (Y - Q_A_update) + (Q_1_update - Q_0_update) - ate
        
        # Standard error and confidence intervals
        se = np.sqrt(np.var(infl_fn) / len(Y))
        ci_low = ate - 1.96 * se
        ci_high = ate + 1.96 * se
        p_value = 2 * (1 - norm.cdf(abs(ate / se))) if se > 0 else 1.0
        
        return ate, se, ci_low, ci_high, p_value, infl_fn, att

    def diagnostic_checks(Y, A, W, Q_A, Q_1, Q_0, g_w, stage="First"):
        """Diagnostic checks"""
        print(f"\n=== {stage} Diagnostic Checks ===")
        
        # 1. Data balance
        treatment_prop = A.mean()
        print(f"Treatment group proportion: {treatment_prop:.4f}")

        # 2. Propensity score distribution
        g_treated = g_w[A==1].mean()
        g_control = g_w[A==0].mean()
        g_overlap = np.minimum(g_w, 1-g_w).mean()

        print(f"Propensity Score - Treated Mean: {g_treated:.4f}")
        print(f"Propensity Score - Control Mean: {g_control:.4f}")
        print(f"Overlap Measure: {g_overlap:.4f} (Higher is better)")

        # 3. Q Model Performance
        y_pred = (Q_A >= 0.5).astype(int)
        auc = roc_auc_score(Y, Q_A)
        precision = precision_score(Y, y_pred, zero_division=0)
        recall = recall_score(Y, y_pred, zero_division=0)
        
        print(f"Q model AUC: {auc:.4f}")
        print(f"Q model Precision: {precision:.4f}")
        print(f"Q model Recall: {recall:.4f}")
        print(f"Q model F1 Score: {2 * (precision * recall) / (precision + recall):.4f}")

        # 4. Prediction distribution check
        print(f"Q_A distribution: min={Q_A.min():.4f}, max={Q_A.max():.4f}, mean={Q_A.mean():.4f}")
        print(f"Q_1 distribution: min={Q_1.min():.4f}, max={Q_1.max():.4f}, mean={Q_1.mean():.4f}")
        print(f"Q_0 distribution: min={Q_0.min():.4f}, max={Q_0.max():.4f}, mean={Q_0.mean():.4f}")

        # 5. Raw ATE estimate
        raw_ate = np.mean(Q_1 - Q_0)
        print(f"{stage} ATE estimate: {raw_ate:.6f}")

        # 6. Numerical stability check
        extreme_ps = np.sum((g_w < 0.05) | (g_w > 0.95))
        print(f"Extreme propensity score samples: {extreme_ps} ({extreme_ps/len(g_w)*100:.2f}%)")

        return {
            'treatment_prop': treatment_prop,
            'g_treated': g_treated,
            'g_control': g_control,
            'g_overlap': g_overlap,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'raw_ate': raw_ate,
            'extreme_ps_count': extreme_ps
        }

    def print_results(ate, se, ci_low, ci_high, p_value, raw_ate, pre_diagnostics, post_diagnostics, att):
        """Print results and diagnostic comparisons"""
        print("\n" + "="*60)
        print("                   TMLE Final Results")
        print("="*60)
        print(f"TMLE ATE Estimate:    {ate:.6f}")
        print(f"TMLE ATT Estimate:    {att:.6f}")
        print(f"Standard Error:        {se:.6f}")
        print(f"95% Confidence Interval:     [{ci_low:.6f}, {ci_high:.6f}]")
        print(f"P-value:         {p_value:.6f}")
        print(f"Statistical Significance:      {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        print("\n" + "-"*40)
        print("           Estimate Comparison")
        print("-"*40)
        print(f"Raw ATE (Pre-update):  {pre_diagnostics['raw_ate']:.6f}")
        print(f"TMLE ATE (Post-update):  {ate:.6f}")
        print(f"Adjustment Magnitude:          {abs(ate - pre_diagnostics['raw_ate']):.6f}")
        print(f"Relative Change:          {abs(ate - pre_diagnostics['raw_ate'])/abs(pre_diagnostics['raw_ate'])*100:.2f}%")

        print("\n" + "-"*40)
        print("        Model Performance Comparison")
        print("-"*40)
        print(f"{'Indicator':<15} {'Before ':<12} {'After update':<12} {'Change':<10}")
        print("-"*50)

        # AUC comparison 
        post_auc = post_diagnostics.get('auc', pre_diagnostics['auc'])
        auc_change = post_auc - pre_diagnostics['auc']
        print(f"{'AUC':<15} {pre_diagnostics['auc']:<12.4f} {post_auc:<12.4f} {auc_change:+.4f}")
        
        # Overlap indicator comparison
        overlap_change = post_diagnostics['g_overlap'] - pre_diagnostics['g_overlap']
        print(f"{'Overlap Indicator':<15} {pre_diagnostics['g_overlap']:<12.4f} {post_diagnostics['g_overlap']:<12.4f} {overlap_change:+.4f}")

        # Extreme PS samples comparison
        ps_change = post_diagnostics['extreme_ps_count'] - pre_diagnostics['extreme_ps_count']
        print(f"{'Extreme PS Samples':<15} {pre_diagnostics['extreme_ps_count']:<12} {post_diagnostics['extreme_ps_count']:<12} {ps_change:+}")

        print("\n" + "="*60)
        
        # Results summary
        if abs(ate) < 0.01:
            effect_size = "Tiny"
        elif abs(ate) < 0.05:
            effect_size = "Small"
        elif abs(ate) < 0.1:
            effect_size = "Medium"
        else:
            effect_size = "Big"

        direction = "Positive" if ate > 0 else "Negative"
        significance = "Statistically Significant" if p_value < 0.05 else "Not Statistically Significant"

        print(f"Results Summary: Found {effect_size} {direction} Treatment Effect, {significance}.")

    #############################################################################
    # Main Process
    print("***** Start TMLE Analysis *****")
    print("="*60)

    # Main Progress Bar
    with tqdm(total=8, desc="TMLE Progress", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as main_pbar:

        try:
            # 1. Data Loading and Preprocessing
            main_pbar.set_description("***** Loading and Preprocessing Data *****")
            Y, A, W, W_A = data_loading(file_path)
            W_standardized, scaler = data_preprocessing(W)
            W_A_standardized = pd.concat([W_standardized, pd.DataFrame(A, columns=['A'])], axis=1)
            main_pbar.update(1)
            time.sleep(0.1)  

            # 2. Set up base learners
            main_pbar.set_description("***** Setting Up Base Learners *****")
            n_features = W.shape[1]
            base_learners = get_base_learners(n_features)
            main_pbar.update(1)
            time.sleep(0.1)

            # 3. Fit Q models
            main_pbar.set_description("> Step 1: Fit Outcome Models (Q)")
            print("\nStep 1: Fit Outcome Models (Q)")
            sl = fit_superlearner(W_A_standardized, Y, base_learners, "Outcome Model")
            Q_A, Q_1, Q_0 = predict_Q_models(sl, W_A_standardized, A)
            main_pbar.update(1)

            # 4. Estimate propensity scores
            main_pbar.set_description("> Step 2: Estimate Propensity Scores (g)")
            print("\nStep 2: Estimate Propensity Scores (g)")
            g_model, g_w, H_1, H_0 = estimate_g(A, W_standardized, base_learners)
            main_pbar.update(1)

            # 5. Pre-update diagnostic checks
            main_pbar.set_description("***** Diagnostic Checks - Pre-update *****")
            pre_diagnostics = diagnostic_checks(Y, A, W, Q_A, Q_1, Q_0, g_w, "Pre-update")
            main_pbar.update(1)

            # 6. Estimate and update fluctuation parameters
            main_pbar.set_description("> Step 3: TMLE Update")
            print("\nStep 3: TMLE Update")
            eps = estimate_fluctuation_param(Y, Q_A, H_1, H_0, A)

            Q_A_update = update_Q(Q_A, A * H_1 - (1 - A) * H_0, eps)
            Q_1_update = update_Q(Q_1, H_1, eps)
            Q_0_update = update_Q(Q_0, -H_0, eps)
            main_pbar.update(1)

            # 7. Post-update diagnostic checks
            main_pbar.set_description("***** Diagnostic Checks - Post-update *****")
            post_diagnostics = diagnostic_checks(Y, A, W, Q_A_update, Q_1_update, Q_0_update, g_w, "Post-update")
            main_pbar.update(1)

            # 8. Compute final results
            main_pbar.set_description("***** Compute Final Results *****")
            ate, se, ci_low, ci_high, p_value, infl_fn, att = compute_tmle(
                Y, A, Q_A_update, Q_1_update, Q_0_update, H_1, H_0)
            
            raw_ate = pre_diagnostics['raw_ate']
            main_pbar.update(1)
            
            # 9. 打印結果
            print_results(ate, se, ci_low, ci_high, p_value, raw_ate, pre_diagnostics, post_diagnostics, att)
            
            return {
                'ate': ate,'att': att ,'se': se, 'ci_low': ci_low, 'ci_high': ci_high, 
                'p_value': p_value, 'raw_ate': raw_ate, 'influence_function': infl_fn,
                'pre_diagnostics': pre_diagnostics, 'post_diagnostics': post_diagnostics
            }
        
        except Exception as e:
            main_pbar.set_description("<< Analysis Failed >>")
            print(f"An error occurred during the analysis: {str(e)}")
            print("Please check the data format and path")
            return None


results = tmle_project('/Users/chendawei/Desktop/Task 2 /yasmeen tmle/tmle_data_s0.csv')