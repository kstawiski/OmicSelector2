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
    from sklearn.linear_model import LassoCV, ElasticNetCV
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

                # Extract features and target
                # Assume last column is target, rest are features
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                # Get configuration
                methods = config.get('methods', ['lasso'])
                n_features = config.get('n_features', 100)
                cv_folds = config.get('cv_folds', 5)

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
                }

                # Run feature selection with first specified method
                # TODO: Support running multiple methods and aggregating results
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
]
