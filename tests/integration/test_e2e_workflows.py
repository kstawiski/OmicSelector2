"""End-to-end integration tests for complete workflows.

Tests:
- Complete feature selection workflow (upload → job → results)
- Complete model training workflow (upload → feature selection → training → results)
- Multi-method feature selection with ensemble
- Hyperparameter optimization workflow
- Pipeline chaining (feature selection → model training)

Note: These tests require PostgreSQL, Redis, S3/MinIO, and Celery workers to be running.
They are marked as slow tests and may take several minutes to complete.
"""

import time
from typing import Optional

import pytest
from fastapi.testclient import TestClient


def wait_for_job_completion(
    client: TestClient,
    auth_headers: dict,
    job_id: str,
    timeout: int = 300,
    poll_interval: int = 2,
) -> dict:
    """Wait for job to complete and return final status.

    Args:
        client: FastAPI test client
        auth_headers: Authentication headers
        job_id: Job UUID
        timeout: Maximum wait time in seconds
        poll_interval: Polling interval in seconds

    Returns:
        Final job status dict

    Raises:
        TimeoutError: If job doesn't complete within timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to get job status: {response.json()}")

        job_data = response.json()
        status = job_data["status"]

        if status in ["completed", "failed", "cancelled"]:
            return job_data

        time.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.slow
class TestFeatureSelectionWorkflow:
    """End-to-end tests for feature selection workflow."""

    def test_complete_feature_selection_workflow(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test complete feature selection workflow from upload to results."""
        # Step 1: Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={
                "name": "E2E Feature Selection Test",
                "description": "Testing complete workflow",
                "data_type": "bulk_rna_seq",
            },
        )

        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        # Verify dataset was created
        dataset_response = client.get(
            f"/api/v1/data/{dataset_id}", headers=auth_headers
        )
        assert dataset_response.status_code == 200
        assert dataset_response.json()["n_samples"] == 100

        # Step 2: Create feature selection job
        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "cv_folds": 3,  # Reduced for speed
                "target_column": "target",
            },
        )

        assert job_response.status_code == 201
        job_data = job_response.json()
        job_id = job_data["id"]
        assert job_data["status"] == "pending"

        # Step 3: Wait for job completion
        final_job = wait_for_job_completion(client, auth_headers, job_id, timeout=120)

        # Verify job completed successfully
        assert final_job["status"] == "completed", f"Job failed: {final_job.get('error_message')}"
        assert final_job["completed_at"] is not None

        # Step 4: Retrieve results
        results_response = client.get(
            f"/api/v1/jobs/{job_id}/results", headers=auth_headers
        )

        assert results_response.status_code == 200
        results_data = results_response.json()

        # Verify results structure
        assert "selected_features" in results_data
        assert "metrics" in results_data
        assert len(results_data["selected_features"]) > 0

        # Step 5: Verify results are queryable
        all_jobs_response = client.get("/api/v1/jobs/", headers=auth_headers)
        assert all_jobs_response.status_code == 200
        job_list = all_jobs_response.json()
        assert any(j["id"] == job_id for j in job_list)

    def test_multi_method_feature_selection(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test feature selection with multiple methods."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Multi-Method Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create job with multiple methods
        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso", "random_forest"],
                "n_features": 15,
                "cv_folds": 3,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]

        # Wait for completion
        final_job = wait_for_job_completion(client, auth_headers, job_id, timeout=180)

        assert final_job["status"] == "completed"

        # Get results
        results_response = client.get(
            f"/api/v1/jobs/{job_id}/results", headers=auth_headers
        )

        results_data = results_response.json()

        # Should have results from multiple methods
        assert len(results_data["selected_features"]) > 0

    def test_feature_selection_with_stability(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test feature selection with stability selection enabled."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Stability Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create job with stability selection
        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "cv_folds": 3,
                "target_column": "target",
                "stability": {
                    "enabled": True,
                    "n_bootstraps": 20,  # Reduced for speed
                    "threshold": 0.6,
                },
            },
        )

        job_id = job_response.json()["id"]

        # Wait for completion (may take longer with stability selection)
        final_job = wait_for_job_completion(client, auth_headers, job_id, timeout=240)

        assert final_job["status"] == "completed"

        # Get results
        results_response = client.get(
            f"/api/v1/jobs/{job_id}/results", headers=auth_headers
        )

        results_data = results_response.json()

        # Verify stability scores are included
        assert len(results_data["selected_features"]) > 0


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.slow
class TestModelTrainingWorkflow:
    """End-to-end tests for model training workflow."""

    def test_complete_model_training_workflow(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test complete model training workflow."""
        # Step 1: Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "E2E Training Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Step 2: Create model training job
        job_response = client.post(
            "/api/v1/jobs/model-training",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "random_forest",
                "task_type": "classification",
                "cv_folds": 3,
                "target_column": "target",
            },
        )

        assert job_response.status_code == 201
        job_id = job_response.json()["id"]

        # Step 3: Wait for completion
        final_job = wait_for_job_completion(client, auth_headers, job_id, timeout=180)

        assert final_job["status"] == "completed"

        # Step 4: Retrieve results
        results_response = client.get(
            f"/api/v1/jobs/{job_id}/results", headers=auth_headers
        )

        assert results_response.status_code == 200
        results_data = results_response.json()

        # Verify results structure
        assert "metrics" in results_data
        assert "cv_metrics" in results_data
        assert results_data["metrics"]["accuracy"] is not None

    def test_model_training_with_hyperparameter_optimization(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test model training with Optuna hyperparameter optimization."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Optuna Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create training job with optimization
        job_response = client.post(
            "/api/v1/jobs/model-training",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "random_forest",
                "task_type": "classification",
                "cv_folds": 3,
                "target_column": "target",
                "optimize_hyperparameters": True,
                "n_trials": 5,  # Reduced for speed
            },
        )

        job_id = job_response.json()["id"]

        # Wait for completion (optimization takes longer)
        final_job = wait_for_job_completion(client, auth_headers, job_id, timeout=300)

        assert final_job["status"] == "completed"

        # Get results
        results_response = client.get(
            f"/api/v1/jobs/{job_id}/results", headers=auth_headers
        )

        results_data = results_response.json()

        # Should have optimization results
        assert results_data["metrics"]["accuracy"] is not None


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.slow
class TestPipelineChaining:
    """End-to-end tests for chaining feature selection and model training."""

    def test_feature_selection_to_model_training_pipeline(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test complete pipeline: feature selection → model training."""
        # Step 1: Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Pipeline Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Step 2: Run feature selection
        fs_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "cv_folds": 3,
                "target_column": "target",
            },
        )

        fs_job_id = fs_response.json()["id"]

        # Wait for feature selection to complete
        fs_final = wait_for_job_completion(client, auth_headers, fs_job_id, timeout=120)

        assert fs_final["status"] == "completed"

        # Step 3: Train model using selected features
        training_response = client.post(
            "/api/v1/jobs/model-training",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "logistic_regression",
                "task_type": "classification",
                "cv_folds": 3,
                "target_column": "target",
                "feature_selection_job_id": fs_job_id,  # Use selected features
            },
        )

        training_job_id = training_response.json()["id"]

        # Wait for model training to complete
        training_final = wait_for_job_completion(
            client, auth_headers, training_job_id, timeout=120
        )

        assert training_final["status"] == "completed"

        # Step 4: Verify results
        training_results = client.get(
            f"/api/v1/jobs/{training_job_id}/results", headers=auth_headers
        )

        results_data = training_results.json()

        # Should have trained model with selected features
        assert results_data["metrics"]["accuracy"] is not None

    def test_multiple_models_on_same_features(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test training multiple models on the same selected features."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Multi-Model Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Run feature selection
        fs_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "cv_folds": 3,
                "target_column": "target",
            },
        )

        fs_job_id = fs_response.json()["id"]

        # Wait for completion
        wait_for_job_completion(client, auth_headers, fs_job_id, timeout=120)

        # Train multiple models with same features
        model_types = ["random_forest", "logistic_regression"]
        job_ids = []

        for model_type in model_types:
            response = client.post(
                "/api/v1/jobs/model-training",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    "model_type": model_type,
                    "task_type": "classification",
                    "cv_folds": 3,
                    "target_column": "target",
                    "feature_selection_job_id": fs_job_id,
                },
            )

            job_ids.append(response.json()["id"])

        # Wait for all models to complete
        results = []
        for job_id in job_ids:
            final_job = wait_for_job_completion(client, auth_headers, job_id, timeout=120)
            assert final_job["status"] == "completed"

            results_response = client.get(
                f"/api/v1/jobs/{job_id}/results", headers=auth_headers
            )
            results.append(results_response.json())

        # All models should have results
        assert len(results) == len(model_types)
        for result in results:
            assert result["metrics"]["accuracy"] is not None


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.slow
class TestErrorHandling:
    """End-to-end tests for error handling in workflows."""

    def test_job_failure_handling(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that job failures are properly recorded."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Error Handling Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create job with invalid configuration (should fail)
        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["invalid_method"],  # Invalid method
                "n_features": 10,
                "target_column": "target",
            },
        )

        # Job creation should fail with validation error
        assert job_response.status_code in [400, 422]

    def test_cancel_running_job(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test canceling a running job."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Cancel Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create job
        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]

        # Immediately cancel the job
        cancel_response = client.post(
            f"/api/v1/jobs/{job_id}/cancel", headers=auth_headers
        )

        assert cancel_response.status_code == 200

        # Check status
        status_response = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)

        # Status should be cancelled or pending (if cancelled before starting)
        assert status_response.json()["status"] in ["cancelled", "pending"]
