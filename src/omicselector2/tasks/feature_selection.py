"""Feature selection Celery tasks.

This module provides Celery tasks for running feature selection jobs.
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    from omicselector2.tasks import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None  # type: ignore

try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV, LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold as SKVarianceThreshold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - feature selection will not work")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not available - XGBoost feature selection will not work")

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.warning("lifelines not available - Cox PH feature selection will not work")


def run_lasso_feature_selection(X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100) -> tuple[list[str], dict]:
    """Run Lasso feature selection with cross-validation.

    Args:
        X: Feature matrix (samples x features)
        y: Target variable
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run LassoCV to find optimal alpha
    lasso_cv = LassoCV(cv=cv, random_state=42, max_iter=10000, n_jobs=-1)
    lasso_cv.fit(X_scaled, y)

    # Get feature coefficients
    coef = np.abs(lasso_cv.coef_)

    # Select top n features by absolute coefficient
    top_indices = np.argsort(coef)[::-1][:n_features]
    top_indices = top_indices[coef[top_indices] > 0]  # Only non-zero coefficients

    selected_features = [X.columns[i] for i in top_indices]

    # Calculate metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        metrics = {
            "method": "lasso",
            "n_features_selected": len(selected_features),
            "optimal_alpha": float(lasso_cv.alpha_),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
        }
    else:
        metrics = {
            "method": "lasso",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_randomforest_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run Random Forest feature selection with variable importance.

    Args:
        X: Feature matrix (samples x features)
        y: Target variable
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    # Train Random Forest to get feature importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_

    # Select top n features by importance
    top_indices = np.argsort(importances)[::-1][:n_features]
    selected_features = [X.columns[i] for i in top_indices]

    # Calculate metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        # Store feature importances for top features
        feature_importance_dict = {
            X.columns[i]: float(importances[i]) for i in top_indices
        }

        metrics = {
            "method": "random_forest",
            "n_features_selected": len(selected_features),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
            "feature_importances": feature_importance_dict,
        }
    else:
        metrics = {
            "method": "random_forest",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_elasticnet_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run Elastic Net feature selection with cross-validation.

    Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization,
    which is particularly useful for handling correlated features.

    Args:
        X: Feature matrix (samples x features)
        y: Target variable
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run ElasticNetCV to find optimal alpha and l1_ratio
    elasticnet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],  # Test different L1/L2 ratios
        cv=cv,
        random_state=42,
        max_iter=10000,
        n_jobs=-1
    )
    elasticnet_cv.fit(X_scaled, y)

    # Get feature coefficients
    coef = np.abs(elasticnet_cv.coef_)

    # Select top n features by absolute coefficient
    top_indices = np.argsort(coef)[::-1][:n_features]
    top_indices = top_indices[coef[top_indices] > 0]  # Only non-zero coefficients

    selected_features = [X.columns[i] for i in top_indices]

    # Calculate metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        metrics = {
            "method": "elastic_net",
            "n_features_selected": len(selected_features),
            "optimal_alpha": float(elasticnet_cv.alpha_),
            "optimal_l1_ratio": float(elasticnet_cv.l1_ratio_),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
        }
    else:
        metrics = {
            "method": "elastic_net",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_xgboost_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run XGBoost feature selection with feature importance.

    XGBoost uses gradient boosting to train an ensemble of decision trees,
    providing feature importances based on gain, weight, or cover.

    Args:
        X: Feature matrix (samples x features)
        y: Target variable
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is required for XGBoost feature selection")
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    # Train XGBoost to get feature importances
    xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb.fit(X, y)

    # Get feature importances (based on gain by default)
    importances = xgb.feature_importances_

    # Select top n features by importance
    top_indices = np.argsort(importances)[::-1][:n_features]
    selected_features = [X.columns[i] for i in top_indices]

    # Calculate metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        # Store feature importances for top features
        feature_importance_dict = {
            X.columns[i]: float(importances[i]) for i in top_indices
        }

        metrics = {
            "method": "xgboost",
            "n_features_selected": len(selected_features),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
            "feature_importances": feature_importance_dict,
        }
    else:
        metrics = {
            "method": "xgboost",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_variance_threshold_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run Variance Threshold feature selection.

    This is a simple filter method that removes features with low variance.
    Useful as a preprocessing step to remove uninformative features.

    Args:
        X: Feature matrix (samples x features)
        y: Target variable
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    # Calculate variance for each feature
    variances = X.var()

    # Filter out zero-variance features (constants)
    non_zero_variance_mask = variances > 0
    non_zero_indices = np.where(non_zero_variance_mask)[0]
    non_zero_variances = variances[non_zero_variance_mask]

    if len(non_zero_variances) == 0:
        # All features have zero variance
        selected_features = []
        threshold = 0.0
    else:
        # Sort features by variance (descending)
        sorted_relative_indices = np.argsort(non_zero_variances.values)[::-1]
        sorted_indices = non_zero_indices[sorted_relative_indices]

        # Select top n features by variance
        # Make sure we don't request more features than available
        n_available = min(n_features, len(sorted_indices))
        top_indices = sorted_indices[:n_available]
        selected_features = [X.columns[i] for i in top_indices]

        # Determine threshold used (minimum variance of selected features)
        threshold = float(variances.iloc[sorted_indices[n_available - 1]])

    # Calculate metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        metrics = {
            "method": "variance_threshold",
            "n_features_selected": len(selected_features),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
            "variance_threshold": threshold,
        }
    else:
        metrics = {
            "method": "variance_threshold",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_ttest_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run t-test feature selection.

    Uses independent t-test to identify features with significant
    difference between classes. Common for gene expression analysis.

    Args:
        X: Feature matrix (samples x features)
        y: Target variable (must be binary)
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    from scipy.stats import ttest_ind

    # Check if binary classification
    unique_classes = y.unique()
    if len(unique_classes) != 2:
        raise ValueError(f"t-test requires binary classification, found {len(unique_classes)} classes")

    # Split data by class
    class0_mask = y == unique_classes[0]
    class1_mask = y == unique_classes[1]

    # Calculate t-test for each feature
    p_values = []
    t_statistics = []

    for col in X.columns:
        class0_vals = X.loc[class0_mask, col]
        class1_vals = X.loc[class1_mask, col]

        t_stat, p_val = ttest_ind(class0_vals, class1_vals, equal_var=False)  # Welch's t-test
        t_statistics.append(abs(t_stat))  # Use absolute value
        p_values.append(p_val)

    # Sort by t-statistic (or equivalently, by p-value ascending)
    sorted_indices = np.argsort(p_values)[:n_features]
    selected_features = [X.columns[i] for i in sorted_indices]

    # Create p-value dictionary for selected features
    p_value_dict = {
        X.columns[i]: float(p_values[i]) for i in sorted_indices
    }

    # Calculate metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        metrics = {
            "method": "ttest",
            "n_features_selected": len(selected_features),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
            "p_values": p_value_dict,
        }
    else:
        metrics = {
            "method": "ttest",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_l1svm_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run L1-SVM feature selection.

    Uses Linear SVM with L1 penalty for embedded feature selection.
    L1 regularization drives feature weights to zero, effectively selecting features.

    Args:
        X: Feature matrix (samples x features)
        y: Target variable
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal C parameter using cross-validation
    from sklearn.model_selection import GridSearchCV

    # Try different C values
    C_range = [0.001, 0.01, 0.1, 1.0, 10.0]
    svm = LinearSVC(penalty='l1', dual=False, random_state=42, max_iter=5000)

    grid_search = GridSearchCV(
        svm,
        param_grid={'C': C_range},
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_scaled, y)

    # Get feature coefficients from best model
    best_svm = grid_search.best_estimator_
    coef = np.abs(best_svm.coef_[0])

    # Select top n features by absolute coefficient
    top_indices = np.argsort(coef)[::-1][:n_features]
    top_indices = top_indices[coef[top_indices] > 0]  # Only non-zero coefficients

    selected_features = [X.columns[i] for i in top_indices]

    # Calculate metrics
    from sklearn.model_selection import cross_val_score

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        metrics = {
            "method": "l1_svm",
            "n_features_selected": len(selected_features),
            "optimal_C": float(grid_search.best_params_['C']),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
        }
    else:
        metrics = {
            "method": "l1_svm",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_ridge_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run Ridge Regression feature selection.

    Ridge uses L2 regularization which shrinks coefficients but doesn't zero them out.
    Features are ranked by absolute coefficient magnitude.

    Args:
        X: Feature matrix (samples x features)
        y: Target variable
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run RidgeCV to find optimal alpha
    from sklearn.linear_model import RidgeClassifierCV

    ridge_cv = RidgeClassifierCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=cv)
    ridge_cv.fit(X_scaled, y)

    # Get feature coefficients
    coef = np.abs(ridge_cv.coef_[0])

    # Select top n features by absolute coefficient
    top_indices = np.argsort(coef)[::-1][:n_features]
    selected_features = [X.columns[i] for i in top_indices]

    # Calculate metrics
    from sklearn.model_selection import cross_val_score

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        metrics = {
            "method": "ridge",
            "n_features_selected": len(selected_features),
            "optimal_alpha": float(ridge_cv.alpha_),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
        }
    else:
        metrics = {
            "method": "ridge",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_coxph_feature_selection(
    X: "pd.DataFrame", y: "pd.DataFrame", n_features: int = 100
) -> tuple[list[str], dict]:
    """Run Cox Proportional Hazards feature selection.

    Uses univariate Cox PH models for each feature to identify genes
    associated with survival. Essential for cancer survival analysis.

    Args:
        X: Feature matrix (samples x features)
        y: Survival data DataFrame with 'time' and 'event' columns
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines is required for Cox PH feature selection")

    if not isinstance(y, pd.DataFrame):
        raise ValueError("y must be a DataFrame with 'time' and 'event' columns for Cox PH")

    if 'time' not in y.columns or 'event' not in y.columns:
        raise ValueError("y must have 'time' and 'event' columns for survival analysis")

    # Calculate univariate Cox PH hazard ratios and p-values for each feature
    hazard_ratios = []
    p_values = []
    c_indices = []

    for col in X.columns:
        try:
            # Create dataframe with single feature + survival data
            df_temp = pd.DataFrame({
                'feature': X[col],
                'time': y['time'],
                'event': y['event']
            })

            # Fit univariate Cox model
            cph = CoxPHFitter()
            cph.fit(df_temp, duration_col='time', event_col='event')

            # Get hazard ratio and p-value
            hr = cph.hazard_ratios_['feature']
            p_val = cph.summary['p']['feature']

            # Calculate concordance index
            c_index = concordance_index(y['time'], -cph.predict_partial_hazard(df_temp[['feature']]).values.flatten(), y['event'])

            hazard_ratios.append(abs(np.log(hr)))  # Use log hazard ratio magnitude
            p_values.append(p_val)
            c_indices.append(c_index)

        except Exception as e:
            # If fitting fails for a feature, assign worst values
            logger.warning(f"Cox PH failed for feature {col}: {str(e)}")
            hazard_ratios.append(0.0)
            p_values.append(1.0)
            c_indices.append(0.5)

    # Select top n features by log hazard ratio magnitude (most associated with survival)
    sorted_indices = np.argsort(hazard_ratios)[::-1][:n_features]
    selected_features = [X.columns[i] for i in sorted_indices]

    # Create hazard ratio dictionary for selected features
    hr_dict = {
        X.columns[i]: float(hazard_ratios[i]) for i in sorted_indices
    }

    # Calculate average C-index for selected features
    if len(selected_features) > 0:
        selected_c_indices = [c_indices[i] for i in sorted_indices]

        metrics = {
            "method": "cox_ph",
            "n_features_selected": len(selected_features),
            "c_index_mean": float(np.mean(selected_c_indices)),
            "c_index_std": float(np.std(selected_c_indices)),
            "hazard_ratios": hr_dict,
        }
    else:
        metrics = {
            "method": "cox_ph",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_mrmr_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run mRMR (Minimum Redundancy Maximum Relevance) feature selection.

    mRMR selects features that maximize relevance to target while minimizing
    redundancy among selected features. Uses mutual information.

    Args:
        X: Feature matrix (samples x features)
        y: Target variable
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    # Calculate mutual information between each feature and target (relevance)
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # mRMR greedy selection
    selected_indices = []
    remaining_indices = list(range(X.shape[1]))

    # Select first feature with highest relevance
    if len(remaining_indices) > 0:
        first_idx = np.argmax(mi_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

    # Iteratively select features that maximize relevance - redundancy
    while len(selected_indices) < min(n_features, X.shape[1]) and remaining_indices:
        mrmr_scores = []

        for idx in remaining_indices:
            # Relevance: MI with target
            relevance = mi_scores[idx]

            # Redundancy: average MI with already selected features
            if len(selected_indices) > 0:
                redundancy = 0
                for selected_idx in selected_indices:
                    # Compute MI between two features (treat one as continuous target)
                    mi_between = mutual_info_regression(
                        X.iloc[:, [idx]].values,
                        X.iloc[:, selected_idx].values.ravel(),
                        random_state=42
                    )[0]
                    redundancy += mi_between
                redundancy /= len(selected_indices)
            else:
                redundancy = 0

            # mRMR score = relevance - redundancy
            mrmr_score = relevance - redundancy
            mrmr_scores.append(mrmr_score)

        # Select feature with highest mRMR score
        best_idx_in_remaining = np.argmax(mrmr_scores)
        best_idx = remaining_indices[best_idx_in_remaining]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    selected_features = [X.columns[i] for i in selected_indices]

    # Create MI scores dictionary for selected features
    mi_dict = {X.columns[i]: float(mi_scores[i]) for i in selected_indices}

    # Calculate metrics
    from sklearn.model_selection import cross_val_score

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        metrics = {
            "method": "mrmr",
            "n_features_selected": len(selected_features),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
            "mutual_information_scores": mi_dict,
        }
    else:
        metrics = {
            "method": "mrmr",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_boruta_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run Boruta-like feature selection.

    Boruta is a wrapper method around Random Forest that compares feature
    importances against shadow features (randomized copies).

    Args:
        X: Feature matrix (samples x features)
        y: Target variable
        cv: Number of cross-validation folds
        n_features: Maximum number of features to select

    Returns:
        Tuple of (selected_feature_names, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    # Create shadow features (randomized copies)
    np.random.seed(42)
    X_shadow = X.apply(lambda col: np.random.permutation(col.values), axis=0)
    X_shadow.columns = [f"shadow_{col}" for col in X.columns]

    # Combine original and shadow features
    X_combined = pd.concat([X, X_shadow], axis=1)

    # Train Random Forest on combined dataset
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_combined, y)

    # Get feature importances
    importances = rf.feature_importances_

    # Split importances into original and shadow
    n_features_original = X.shape[1]
    original_importances = importances[:n_features_original]
    shadow_importances = importances[n_features_original:]

    # Statistical test: compare each feature to max shadow importance
    max_shadow_importance = np.max(shadow_importances)

    # Select features with importance greater than max shadow
    significant_mask = original_importances > max_shadow_importance
    significant_indices = np.where(significant_mask)[0]

    # If we have more significant features than requested, select top n
    if len(significant_indices) > n_features:
        top_n_indices = np.argsort(original_importances[significant_indices])[::-1][:n_features]
        selected_indices = significant_indices[top_n_indices]
    else:
        selected_indices = significant_indices

    selected_features = [X.columns[i] for i in selected_indices]

    # Create importance scores dictionary for selected features
    importance_dict = {
        X.columns[i]: float(original_importances[i]) for i in selected_indices
    }

    # Calculate metrics
    from sklearn.model_selection import cross_val_score

    # Train a simple logistic regression on selected features for evaluation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        metrics = {
            "method": "boruta",
            "n_features_selected": len(selected_features),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
            "importance_scores": importance_dict,
            "max_shadow_importance": float(max_shadow_importance),
        }
    else:
        metrics = {
            "method": "boruta",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


def run_relieff_feature_selection(
    X: "pd.DataFrame", y: "pd.Series", cv: int = 5, n_features: int = 100
) -> tuple[list[str], dict]:
    """Run ReliefF feature selection.

    ReliefF is an instance-based feature selection algorithm that evaluates
    features based on their ability to distinguish between near hits (same class
    neighbors) and near misses (different class neighbors).

    Algorithm:
    1. For m iterations:
       - Randomly sample an instance
       - Find k nearest neighbors from same class (near hits)
       - Find k nearest neighbors from different classes (near misses)
       - Update feature weights:
         * Decrease if feature differs from near hits
         * Increase if feature differs from near misses
    2. Select top features by final weights

    Args:
        X: Feature matrix (samples x features) as DataFrame
        y: Target variable (samples,) as Series
        cv: Number of cross-validation folds for evaluation
        n_features: Number of features to select

    Returns:
        Tuple of (selected_features, metrics)
        - selected_features: List of selected feature names
        - metrics: Dictionary containing method metadata and performance

    Raises:
        ImportError: If scikit-learn is not installed
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature selection")

    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import cross_val_score

    # Standardize features for distance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ReliefF parameters
    n_iterations = min(100, len(X))  # Number of random samples
    k_neighbors = min(10, len(X) // 10)  # Number of neighbors

    # Initialize feature weights
    weights = np.zeros(X.shape[1])

    # Get unique classes
    classes = np.unique(y)
    n_classes = len(classes)

    np.random.seed(42)

    for iteration in range(n_iterations):
        # Randomly sample an instance
        idx = np.random.randint(0, len(X))
        instance = X_scaled[idx]
        instance_class = y.iloc[idx]

        # Find k nearest neighbors from same class (near hits)
        same_class_mask = y == instance_class
        if same_class_mask.sum() > 1:  # Need at least one other instance
            X_same = X_scaled[same_class_mask]
            nn_same = NearestNeighbors(n_neighbors=min(k_neighbors + 1, X_same.shape[0]))
            nn_same.fit(X_same)
            distances_same, indices_same = nn_same.kneighbors([instance])

            # Exclude the instance itself (first neighbor)
            indices_same = indices_same[0][1:]
            same_class_indices = np.where(same_class_mask)[0][indices_same]
            near_hits = X_scaled[same_class_indices]
        else:
            near_hits = np.array([])

        # Find k nearest neighbors from different classes (near misses)
        near_misses_by_class = []
        for other_class in classes:
            if other_class == instance_class:
                continue

            other_class_mask = y == other_class
            if other_class_mask.sum() > 0:
                X_other = X_scaled[other_class_mask]
                nn_other = NearestNeighbors(n_neighbors=min(k_neighbors, X_other.shape[0]))
                nn_other.fit(X_other)
                distances_other, indices_other = nn_other.kneighbors([instance])

                other_class_indices = np.where(other_class_mask)[0][indices_other[0]]
                near_misses_by_class.append(X_scaled[other_class_indices])

        # Update feature weights
        if len(near_hits) > 0:
            # Decrease weight for features that differ from near hits
            diff_hits = np.abs(instance - near_hits)
            weights -= diff_hits.mean(axis=0) / n_iterations

        if near_misses_by_class:
            # Increase weight for features that differ from near misses
            for near_misses in near_misses_by_class:
                diff_misses = np.abs(instance - near_misses)
                weights += diff_misses.mean(axis=0) / (n_iterations * (n_classes - 1))

    # Select top n features by weight
    top_indices = np.argsort(weights)[::-1][:n_features]
    selected_features = [X.columns[i] for i in top_indices]

    # Create weights dictionary for selected features
    relief_scores = {X.columns[i]: float(weights[i]) for i in top_indices}

    # Evaluate with cross-validation
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        lr = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(lr, X_selected, y, cv=cv, scoring='roc_auc')

        metrics = {
            "method": "relieff",
            "n_features_selected": len(selected_features),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "cv_folds": cv,
            "relief_scores": relief_scores,
            "n_iterations": n_iterations,
            "k_neighbors": k_neighbors,
        }
    else:
        metrics = {
            "method": "relieff",
            "n_features_selected": 0,
            "error": "No features selected",
        }

    return selected_features, metrics


if CELERY_AVAILABLE and celery_app:

    @celery_app.task(name="omicselector2.feature_selection", bind=True)
    def feature_selection_task(self, job_id: str, dataset_id: str, config: dict):
        """Run feature selection analysis.

        Args:
            self: Celery task instance (for updating state)
            job_id: Job UUID
            dataset_id: Dataset UUID
            config: Job configuration containing:
                - methods: List of feature selection methods
                - n_features: Number of features to select
                - cv_folds: Cross-validation folds

        Returns:
            dict: Result summary
        """
        logger.info(f"Starting feature selection job {job_id} for dataset {dataset_id}")
        logger.info(f"Configuration: {config}")

        try:
            # Update job status to RUNNING
            from omicselector2.db import get_db, Job, JobStatus, Result, Dataset
            from omicselector2.utils.storage import get_storage_client

            db = next(get_db())

            try:
                job = db.query(Job).filter(Job.id == UUID(job_id)).first()
                if not job:
                    raise ValueError(f"Job {job_id} not found")

                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()
                db.commit()

                # Update task state
                self.update_state(state='PROGRESS', meta={'status': 'Loading data'})

                # Get dataset
                dataset = db.query(Dataset).filter(Dataset.id == UUID(dataset_id)).first()
                if not dataset:
                    raise ValueError(f"Dataset {dataset_id} not found")

                # Load data from S3
                logger.info(f"Loading dataset from {dataset.file_path}")
                storage_client = get_storage_client()

                # Parse S3 URI
                from urllib.parse import urlparse
                parsed = urlparse(dataset.file_path)
                object_name = parsed.path.lstrip('/')

                # Download and read data
                file_obj = storage_client.download_file(object_name)

                # Assume CSV format for now
                if dataset.file_path.endswith('.csv'):
                    df = pd.read_csv(file_obj)
                elif dataset.file_path.endswith('.h5ad'):
                    # For h5ad files, we'd need scanpy
                    raise NotImplementedError("h5ad format not yet supported")
                else:
                    raise ValueError(f"Unsupported file format: {dataset.file_path}")

                logger.info(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} features")

                # Get configuration
                methods = config.get('methods', ['lasso'])
                n_features = config.get('n_features', 100)
                cv_folds = config.get('cv_folds', 5)

                # Determine target columns
                target_columns: list[str] = []
                target_config = config.get('target')
                if isinstance(target_config, str):
                    target_columns = [target_config]
                elif isinstance(target_config, dict):
                    if 'column' in target_config:
                        target_columns = [target_config['column']]
                    elif 'columns' in target_config and isinstance(
                        target_config['columns'], (list, tuple)
                    ):
                        target_columns = list(target_config['columns'])
                    elif {
                        'time_column',
                        'event_column',
                    }.issubset(target_config.keys()):
                        target_columns = [
                            target_config['time_column'],
                            target_config['event_column'],
                        ]

                if not target_columns:
                    explicit_target_column = config.get('target_column')
                    if isinstance(explicit_target_column, str):
                        target_columns = [explicit_target_column]
                    explicit_target_columns = config.get('target_columns')
                    if isinstance(explicit_target_columns, (list, tuple)):
                        target_columns = list(explicit_target_columns)

                if not target_columns:
                    # Fallback: assume last column is target, rest are features
                    target_columns = [df.columns[-1]]

                missing_columns = [col for col in target_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(
                        f"Target column(s) {missing_columns} not found in dataset"
                    )

                # Ensure Cox PH receives both survival columns
                if 'cox_ph' in methods and len(target_columns) < 2:
                    raise ValueError(
                        "Cox PH feature selection requires both time and event columns. "
                        "Provide them using config['target'] or config['target_columns']."
                    )

                feature_columns = [col for col in df.columns if col not in target_columns]
                if not feature_columns:
                    raise ValueError("No feature columns available after removing target columns")

                X = df[feature_columns]
                if len(target_columns) == 1:
                    y = df[target_columns[0]]
                else:
                    y = df[target_columns]

                # Get stability and ensemble config
                stability_config = config.get('stability', None)
                ensemble_config = config.get('ensemble', None)

                # Update task state
                self.update_state(state='PROGRESS', meta={'status': 'Running feature selection'})

                # Map method names to functions
                method_functions = {
                    'lasso': run_lasso_feature_selection,
                    'elastic_net': run_elasticnet_feature_selection,
                    'random_forest': run_randomforest_feature_selection,
                    'xgboost': run_xgboost_feature_selection,
                    'variance_threshold': run_variance_threshold_feature_selection,
                    'ttest': run_ttest_feature_selection,
                    'l1_svm': run_l1svm_feature_selection,
                    'ridge': run_ridge_feature_selection,
                    'cox_ph': run_coxph_feature_selection,
                    'mrmr': run_mrmr_feature_selection,
                    'boruta': run_boruta_feature_selection,
                    'relieff': run_relieff_feature_selection,
                }

                # Determine execution mode
                if ensemble_config and len(methods) > 1:
                    # ENSEMBLE MODE: Run multiple methods and combine
                    logger.info(f"Running ensemble of {len(methods)} methods")

                    from omicselector2.features.ensemble import EnsembleFeatureSelector

                    # Get method functions
                    selector_funcs = []
                    for method_name in methods:
                        if method_name not in method_functions:
                            logger.warning(f"Skipping unknown method: {method_name}")
                            continue
                        selector_funcs.append(method_functions[method_name])

                    if len(selector_funcs) < 2:
                        raise ValueError("Ensemble requires at least 2 valid methods")

                    # Create ensemble selector
                    ensemble_method = ensemble_config.get('method', 'majority_vote')
                    min_votes = ensemble_config.get('min_votes', 2)
                    ensemble_n_features = ensemble_config.get('n_features', n_features)

                    ensemble_selector = EnsembleFeatureSelector(
                        base_selectors=selector_funcs,
                        ensemble_method=ensemble_method,
                        min_votes=min_votes,
                        n_features=ensemble_n_features,
                        verbose=True
                    )

                    selected_features, metrics = ensemble_selector.select_features(
                        X, y, n_features=n_features, cv=cv_folds
                    )

                    logger.info(f"Ensemble selected {len(selected_features)} features")

                elif stability_config:
                    # STABILITY MODE: Wrap method with stability selection
                    method_name = methods[0] if isinstance(methods, list) else methods

                    if method_name not in method_functions:
                        available_methods = ', '.join(method_functions.keys())
                        raise ValueError(
                            f"Method '{method_name}' not implemented. "
                            f"Available methods: {available_methods}"
                        )

                    logger.info(f"Running {method_name} with stability selection")

                    from omicselector2.features.stability import StabilitySelector

                    # Create stability selector
                    n_bootstraps = stability_config.get('n_bootstraps', 100)
                    threshold = stability_config.get('threshold', 0.6)
                    sample_fraction = stability_config.get('sample_fraction', 0.8)

                    method_func = method_functions[method_name]
                    stability_selector = StabilitySelector(
                        base_selector=method_func,
                        n_bootstraps=n_bootstraps,
                        threshold=threshold,
                        sample_fraction=sample_fraction,
                        random_state=42,
                        verbose=True
                    )

                    selected_features, stability_scores = stability_selector.select_stable_features(
                        X, y, n_features=n_features, cv=cv_folds
                    )

                    # Create metrics dict
                    metrics = {
                        'method': f"{method_name}_stability",
                        'n_features_selected': len(selected_features),
                        'n_bootstraps': n_bootstraps,
                        'threshold': threshold,
                        'stability_scores': {f: stability_scores[f] for f in selected_features}
                    }

                    logger.info(
                        f"Stability selection selected {len(selected_features)} features "
                        f"(threshold={threshold})"
                    )

                else:
                    # SINGLE METHOD MODE: Run one method
                    method_name = methods[0] if isinstance(methods, list) else methods

                    if method_name not in method_functions:
                        available_methods = ', '.join(method_functions.keys())
                        raise ValueError(
                            f"Method '{method_name}' not implemented. "
                            f"Available methods: {available_methods}"
                        )

                    method_func = method_functions[method_name]
                    selected_features, metrics = method_func(
                        X, y, cv=cv_folds, n_features=n_features
                    )

                    logger.info(f"Selected {len(selected_features)} features")

                # Create result record
                result = Result(
                    job_id=UUID(job_id),
                    selected_features=selected_features,
                    metrics=metrics,
                    artifacts_path=None,  # TODO: Upload detailed results to S3
                )

                db.add(result)
                db.commit()
                db.refresh(result)

                # Update job status to COMPLETED
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.result_id = result.id
                db.commit()

                logger.info(f"Feature selection job {job_id} completed successfully")

                return {
                    "job_id": job_id,
                    "status": "completed",
                    "n_features_selected": len(selected_features),
                    "metrics": metrics,
                }

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Feature selection job {job_id} failed: {str(e)}", exc_info=True)

            # Update job status to FAILED
            try:
                db = next(get_db())
                try:
                    job = db.query(Job).filter(Job.id == UUID(job_id)).first()
                    if job:
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.utcnow()
                        job.error_message = str(e)
                        db.commit()
                finally:
                    db.close()
            except Exception as db_error:
                logger.error(f"Failed to update job status: {str(db_error)}")

            # Re-raise to mark Celery task as failed
            raise

else:
    # Stub function when Celery not available
    def feature_selection_task(job_id: str, dataset_id: str, config: dict):
        """Stub feature selection task."""
        raise ImportError("Celery is required for background tasks")


__all__ = [
    "feature_selection_task",
    "run_lasso_feature_selection",
    "run_randomforest_feature_selection",
    "run_elasticnet_feature_selection",
    "run_xgboost_feature_selection",
    "run_variance_threshold_feature_selection",
    "run_ttest_feature_selection",
    "run_l1svm_feature_selection",
    "run_ridge_feature_selection",
    "run_coxph_feature_selection",
    "run_mrmr_feature_selection",
    "run_boruta_feature_selection",
    "run_relieff_feature_selection",
]
