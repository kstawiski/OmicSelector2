"""Model training Celery tasks.

This module provides Celery tasks for running model training jobs.
"""

import io
import logging
import pickle
from datetime import datetime, timezone
from urllib.parse import urlparse
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

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available - model training will not work")

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, SVR

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - model training will not work")

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not available - XGBoost models will not work")


def get_model_class(model_type: str, task_type: str = "classification"):
    """Get model class based on model type and task type.

    Args:
        model_type: Type of model (random_forest, xgboost, logistic_regression, svm)
        task_type: Type of task (classification or regression)

    Returns:
        Model class

    Raises:
        ValueError: If model type is not supported
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for model training")

    model_map = {
        "classification": {
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "svm": SVC,
        },
        "regression": {
            "random_forest": RandomForestRegressor,
            "svm": SVR,
        },
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        model_map["classification"]["xgboost"] = XGBClassifier
        model_map["regression"]["xgboost"] = XGBRegressor

    if task_type not in model_map:
        raise ValueError(f"Task type '{task_type}' not supported. Must be 'classification' or 'regression'")

    if model_type not in model_map[task_type]:
        available = ", ".join(model_map[task_type].keys())
        raise ValueError(f"Model type '{model_type}' not supported for {task_type}. Available: {available}")

    return model_map[task_type][model_type]


if CELERY_AVAILABLE and celery_app:

    @celery_app.task(name="omicselector2.model_training", bind=True)
    def model_training_task(self, job_id: str, dataset_id: str, config: dict):
        """Run model training.

        Args:
            self: Celery task instance (for updating state)
            job_id: Job UUID
            dataset_id: Dataset UUID
            config: Job configuration containing:
                - model_type: Type of model (random_forest, xgboost, logistic_regression, svm)
                - task_type: classification or regression (default: classification)
                - hyperparameters: Model hyperparameters (optional)
                - optimize_hyperparameters: Whether to run Optuna optimization (default: False)
                - cv_folds: Cross-validation folds (default: 5)
                - target_column: Target column name (optional, defaults to last column)
                - selected_features: List of features to use (optional, uses all if not specified)
                - feature_selection_job_id: Previous feature selection job to get features from (optional)

        Returns:
            dict: Result summary
        """
        logger.info(f"Starting model training job {job_id} for dataset {dataset_id}")
        logger.info(f"Configuration: {config}")

        try:
            # Update job status to RUNNING
            from omicselector2.db import Dataset, Job, JobStatus, Result, get_db
            from omicselector2.utils.redis_pubsub import get_publisher
            from omicselector2.utils.storage import get_storage_client

            db = next(get_db())
            publisher = get_publisher()

            try:
                # Connect to Redis for publishing updates
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(publisher.connect())

                # Get job
                job = db.query(Job).filter(Job.id == UUID(job_id)).first()
                if not job:
                    raise ValueError(f"Job {job_id} not found")

                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc)
                db.commit()

                # Publish status update
                loop.run_until_complete(
                    publisher.publish_job_update(
                        job_id=job_id,
                        status="running",
                        message="Job started - loading dataset",
                    )
                )

                # Update task state
                self.update_state(state="PROGRESS", meta={"status": "Loading data"})

                # Get dataset
                dataset = db.query(Dataset).filter(Dataset.id == UUID(dataset_id)).first()
                if not dataset:
                    raise ValueError(f"Dataset {dataset_id} not found")

                # Load data from S3
                logger.info(f"Loading dataset from {dataset.file_path}")
                storage_client = get_storage_client()

                parsed = urlparse(dataset.file_path)
                object_name = parsed.path.lstrip("/")

                # Download and read data
                file_obj = storage_client.download_file(object_name)

                # Load data based on format
                if dataset.file_path.endswith(".csv"):
                    df = pd.read_csv(file_obj)
                elif dataset.file_path.endswith(".h5ad"):
                    raise NotImplementedError("h5ad format not yet supported in model training")
                else:
                    raise ValueError(f"Unsupported file format: {dataset.file_path}")

                logger.info(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} features")

                # Publish progress update
                loop.run_until_complete(
                    publisher.publish_job_update(
                        job_id=job_id,
                        status="running",
                        message=f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features",
                        metadata={"n_samples": df.shape[0], "n_features": df.shape[1]},
                    )
                )

                # Get configuration
                model_type = config.get("model_type", "random_forest")
                task_type = config.get("task_type", "classification")
                cv_folds = config.get("cv_folds", 5)
                optimize_hyperparameters = config.get("optimize_hyperparameters", False)
                hyperparameters = config.get("hyperparameters", {})

                # Determine target column
                target_column = config.get("target_column")
                if not target_column:
                    # Default to last column
                    target_column = df.columns[-1]

                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset")

                # Get selected features
                selected_features = config.get("selected_features")

                # If not provided, check if there's a feature selection job to get features from
                if not selected_features:
                    feature_selection_job_id = config.get("feature_selection_job_id")
                    if feature_selection_job_id:
                        # Load features from previous feature selection job
                        from omicselector2.db import Result as ResultModel

                        fs_result = (
                            db.query(ResultModel)
                            .join(Job)
                            .filter(Job.id == UUID(feature_selection_job_id))
                            .first()
                        )

                        if fs_result and fs_result.selected_features:
                            selected_features = fs_result.selected_features
                            logger.info(
                                f"Using {len(selected_features)} features from feature selection job {feature_selection_job_id}"
                            )
                        else:
                            logger.warning(
                                f"Feature selection job {feature_selection_job_id} has no results, using all features"
                            )

                # Filter features
                if selected_features:
                    # Verify all selected features exist
                    missing_features = [f for f in selected_features if f not in df.columns]
                    if missing_features:
                        logger.warning(f"Missing features: {missing_features[:10]}")
                        selected_features = [f for f in selected_features if f in df.columns]

                    if not selected_features:
                        raise ValueError("No valid features found after filtering")

                    X = df[selected_features]
                    logger.info(f"Using {len(selected_features)} selected features")
                else:
                    # Use all features except target
                    feature_cols = [col for col in df.columns if col != target_column]
                    X = df[feature_cols]
                    selected_features = feature_cols
                    logger.info(f"Using all {len(feature_cols)} features")

                y = df[target_column]

                # Publish progress
                loop.run_until_complete(
                    publisher.publish_job_update(
                        job_id=job_id,
                        status="running",
                        message=f"Prepared data: {X.shape[1]} features, task: {task_type}",
                    )
                )

                # Update task state
                self.update_state(
                    state="PROGRESS", meta={"status": f"Training {model_type} model"}
                )

                # Get model class
                ModelClass = get_model_class(model_type, task_type)

                # Train model with or without hyperparameter optimization
                if optimize_hyperparameters:
                    logger.info("Running hyperparameter optimization with Optuna")

                    from omicselector2.training.hyperparameter import HyperparameterOptimizer

                    optimizer = HyperparameterOptimizer(
                        model_type=model_type,
                        task_type=task_type,
                        n_trials=config.get("n_trials", 50),
                        cv_folds=cv_folds,
                        timeout=config.get("optimization_timeout", 3600),
                    )

                    # Publish progress
                    loop.run_until_complete(
                        publisher.publish_job_update(
                            job_id=job_id,
                            status="running",
                            message=f"Running hyperparameter optimization ({config.get('n_trials', 50)} trials)",
                        )
                    )

                    best_params, best_score, study = optimizer.optimize(X, y)
                    logger.info(f"Best parameters: {best_params}, Best score: {best_score}")

                    # Use optimized hyperparameters
                    model = ModelClass(**best_params)

                    optimization_results = {
                        "best_params": best_params,
                        "best_score": float(best_score),
                        "n_trials": len(study.trials),
                    }
                else:
                    # Use provided hyperparameters or defaults
                    if hyperparameters:
                        logger.info(f"Using provided hyperparameters: {hyperparameters}")
                        model = ModelClass(**hyperparameters)
                    else:
                        logger.info("Using default hyperparameters")
                        model = ModelClass(random_state=42)

                    optimization_results = None

                # Train model with cross-validation
                logger.info(f"Training {model_type} with {cv_folds}-fold cross-validation")

                from omicselector2.training.cross_validation import (
                    CrossValidator,
                    StratifiedKFoldSplitter,
                )
                from omicselector2.training.evaluator import ClassificationEvaluator

                splitter = StratifiedKFoldSplitter(n_splits=cv_folds, shuffle=True, random_state=42)
                evaluator = ClassificationEvaluator()
                cv = CrossValidator(splitter=splitter, evaluator=evaluator)

                # Publish progress
                loop.run_until_complete(
                    publisher.publish_job_update(
                        job_id=job_id,
                        status="running",
                        message=f"Running {cv_folds}-fold cross-validation",
                    )
                )

                cv_results = cv.cross_validate(model, X, y)

                logger.info(f"Cross-validation complete. Mean AUC: {cv_results['test_auc'].mean():.3f}")

                # Train final model on full dataset
                logger.info("Training final model on full dataset")
                model.fit(X, y)

                # Publish progress
                loop.run_until_complete(
                    publisher.publish_job_update(
                        job_id=job_id,
                        status="running",
                        message="Saving trained model",
                    )
                )

                # Save model to S3
                model_filename = f"models/{job_id}/model.pkl"

                # Serialize model
                model_bytes = pickle.dumps(model)
                model_file_obj = io.BytesIO(model_bytes)

                # Upload to S3
                model_s3_path = storage_client.upload_file(
                    file_obj=model_file_obj,
                    object_name=model_filename,
                    metadata={
                        "job_id": job_id,
                        "model_type": model_type,
                        "task_type": task_type,
                        "n_features": X.shape[1],
                    },
                )

                logger.info(f"Model saved to {model_s3_path}")

                # Prepare metrics
                metrics = {
                    "model_type": model_type,
                    "task_type": task_type,
                    "n_features": int(X.shape[1]),
                    "n_samples": int(X.shape[0]),
                    "cv_folds": cv_folds,
                    "cv_results": {
                        "mean_auc": float(cv_results["test_auc"].mean()),
                        "std_auc": float(cv_results["test_auc"].std()),
                        "mean_accuracy": float(cv_results["test_accuracy"].mean()),
                        "std_accuracy": float(cv_results["test_accuracy"].std()),
                        "mean_f1": float(cv_results["test_f1"].mean()),
                        "std_f1": float(cv_results["test_f1"].std()),
                    },
                }

                if optimization_results:
                    metrics["hyperparameter_optimization"] = optimization_results

                # Create result record
                result = Result(
                    job_id=UUID(job_id),
                    selected_features=selected_features if len(selected_features) < 10000 else None,
                    metrics=metrics,
                    artifacts_path=model_s3_path,
                )

                db.add(result)
                db.commit()
                db.refresh(result)

                # Update job status to COMPLETED
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                job.result_id = result.id
                db.commit()

                # Publish final status update
                loop.run_until_complete(
                    publisher.publish_job_update(
                        job_id=job_id,
                        status="completed",
                        message="Model training completed successfully",
                        metadata={
                            "result_id": str(result.id),
                            "mean_auc": metrics["cv_results"]["mean_auc"],
                            "model_path": model_s3_path,
                        },
                    )
                )

                logger.info(f"Model training job {job_id} completed successfully")

                return {
                    "job_id": job_id,
                    "status": "completed",
                    "result_id": str(result.id),
                    "metrics": metrics,
                }

            finally:
                # Clean up
                loop.run_until_complete(publisher.disconnect())
                loop.close()
                db.close()

        except Exception as e:
            logger.error(f"Model training job {job_id} failed: {str(e)}", exc_info=True)

            # Update job status to FAILED
            try:
                db = next(get_db())
                publisher = get_publisher()

                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(publisher.connect())

                    job = db.query(Job).filter(Job.id == UUID(job_id)).first()
                    if job:
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.now(timezone.utc)
                        job.error_message = str(e)
                        db.commit()

                        # Publish failure update
                        loop.run_until_complete(
                            publisher.publish_job_update(
                                job_id=job_id,
                                status="failed",
                                message=f"Model training failed: {str(e)}",
                            )
                        )
                finally:
                    loop.run_until_complete(publisher.disconnect())
                    loop.close()
                    db.close()
            except Exception as db_error:
                logger.error(f"Failed to update job status: {str(db_error)}")

            # Re-raise to mark Celery task as failed
            raise

else:
    # Stub function when Celery not available
    def model_training_task(job_id: str, dataset_id: str, config: dict):
        """Stub model training task."""
        raise ImportError("Celery is required for background tasks")


__all__ = ["model_training_task", "get_model_class"]
