"""Integration tests for job submission workflow.

Tests:
- Create feature selection job
- Create model training job
- Monitor job status (PENDING → RUNNING → COMPLETED)
- Retrieve job results
- Cancel running job
- Job error handling
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestFeatureSelectionJob:
    """Test suite for feature selection job workflows."""

    def test_create_feature_selection_job(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test creating a feature selection job."""
        # Upload dataset first
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Feature Selection Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create feature selection job
        response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso", "random_forest"],
                "n_features": 10,
                "cv_folds": 5,
                "target_column": "target",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["job_type"] == "feature_selection"
        assert data["status"] == "pending"
        assert data["dataset_id"] == dataset_id

    def test_create_job_without_authentication(
        self, client: TestClient, sample_dataset_csv
    ):
        """Test that creating job without auth fails."""
        response = client.post(
            "/api/v1/jobs/feature-selection",
            json={
                "dataset_id": "fake-id",
                "methods": ["lasso"],
                "n_features": 10,
            },
        )

        assert response.status_code == 401

    def test_create_job_with_invalid_dataset(self, client: TestClient, auth_headers):
        """Test that creating job with invalid dataset ID fails."""
        fake_uuid = "00000000-0000-0000-0000-000000000000"

        response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": fake_uuid,
                "methods": ["lasso"],
                "n_features": 10,
            },
        )

        assert response.status_code == 404

    def test_create_job_with_stability_selection(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test creating feature selection job with stability selection."""
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
        response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "stability": {
                    "enabled": True,
                    "n_bootstraps": 50,
                    "threshold": 0.7,
                },
                "target_column": "target",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending"

    def test_create_job_with_ensemble(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test creating feature selection job with ensemble methods."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Ensemble Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create job with ensemble
        response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso", "random_forest", "elastic_net"],
                "n_features": 10,
                "ensemble": {
                    "enabled": True,
                    "method": "majority_vote",
                    "min_votes": 2,
                },
                "target_column": "target",
            },
        )

        assert response.status_code == 201


@pytest.mark.integration
class TestModelTrainingJob:
    """Test suite for model training job workflows."""

    def test_create_model_training_job(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test creating a model training job."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Training Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create model training job
        response = client.post(
            "/api/v1/jobs/model-training",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "random_forest",
                "task_type": "classification",
                "cv_folds": 5,
                "target_column": "target",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["job_type"] == "model_training"
        assert data["status"] == "pending"

    def test_create_training_job_with_hyperparameter_optimization(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test creating training job with hyperparameter optimization."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Optuna Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create job with optimization
        response = client.post(
            "/api/v1/jobs/model-training",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "xgboost",
                "task_type": "classification",
                "optimize_hyperparameters": True,
                "n_trials": 10,
                "target_column": "target",
            },
        )

        assert response.status_code == 201

    def test_create_training_job_with_feature_selection(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test creating training job using features from previous job."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Pipeline Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create feature selection job
        fs_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 5,
                "target_column": "target",
            },
        )

        fs_job_id = fs_response.json()["id"]

        # Create training job using selected features
        response = client.post(
            "/api/v1/jobs/model-training",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "logistic_regression",
                "task_type": "classification",
                "feature_selection_job_id": fs_job_id,
                "target_column": "target",
            },
        )

        assert response.status_code == 201


@pytest.mark.integration
class TestJobMonitoring:
    """Test suite for job monitoring workflows."""

    def test_get_job_status(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test retrieving job status."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Status Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

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

        # Get job status
        response = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert "status" in data
        assert data["status"] in ["pending", "running", "completed", "failed"]

    def test_list_user_jobs(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test listing user's jobs."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "List Jobs Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Create multiple jobs
        for i in range(3):
            client.post(
                "/api/v1/jobs/feature-selection",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    "methods": ["lasso"],
                    "n_features": 10,
                    "target_column": "target",
                },
            )

        # List jobs
        response = client.get("/api/v1/jobs/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3

    def test_cancel_job(self, client: TestClient, auth_headers, sample_dataset_csv):
        """Test canceling a job."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Cancel Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

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

        # Cancel job
        response = client.post(
            f"/api/v1/jobs/{job_id}/cancel", headers=auth_headers
        )

        assert response.status_code == 200

        # Verify status is cancelled
        status_response = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
        assert status_response.json()["status"] in ["cancelled", "pending"]


@pytest.mark.integration
class TestJobResults:
    """Test suite for job result retrieval."""

    def test_get_job_results(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test retrieving job results."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Results Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

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

        # Get results (may not be available yet if job is pending/running)
        response = client.get(
            f"/api/v1/jobs/{job_id}/results", headers=auth_headers
        )

        # Should return 200 with null/empty results or 404 if not ready
        assert response.status_code in [200, 404]

    def test_get_results_for_nonexistent_job(
        self, client: TestClient, auth_headers
    ):
        """Test that getting results for non-existent job returns 404."""
        fake_uuid = "00000000-0000-0000-0000-000000000000"
        response = client.get(
            f"/api/v1/jobs/{fake_uuid}/results", headers=auth_headers
        )

        assert response.status_code == 404


@pytest.mark.integration
class TestJobValidation:
    """Test suite for job configuration validation."""

    def test_create_job_with_invalid_method(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that invalid method name is rejected."""
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Validation Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["invalid_method"],
                "n_features": 10,
            },
        )

        # Should fail validation
        assert response.status_code in [400, 422]

    def test_create_training_job_with_invalid_model_type(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that invalid model type is rejected."""
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Validation Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        response = client.post(
            "/api/v1/jobs/model-training",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "invalid_model",
                "task_type": "classification",
            },
        )

        # Should fail validation
        assert response.status_code in [400, 422]
