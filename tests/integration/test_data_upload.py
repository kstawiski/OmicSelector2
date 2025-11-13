"""Integration tests for data upload workflow.

Tests:
- Upload datasets (CSV)
- Retrieve dataset by ID
- List user datasets with pagination
- Delete dataset (S3 + DB cleanup)
- Upload validation (file format, size)
"""

import io

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestDataUpload:
    """Test suite for data upload workflows."""

    def test_upload_csv_dataset(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test uploading a CSV dataset."""
        # Reset file pointer
        sample_dataset_csv.seek(0)

        response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={
                "name": "Test Dataset",
                "description": "A test dataset for integration testing",
                "data_type": "bulk_rna_seq",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Dataset"
        assert data["description"] == "A test dataset for integration testing"
        assert data["data_type"] == "bulk_rna_seq"
        assert "id" in data
        assert "file_path" in data
        assert data["n_samples"] == 100
        assert data["n_features"] == 21  # 20 features + 1 target

    def test_upload_without_authentication(self, client: TestClient, sample_dataset_csv):
        """Test that upload without authentication fails."""
        sample_dataset_csv.seek(0)

        response = client.post(
            "/api/v1/data/upload",
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Test Dataset", "data_type": "bulk_rna_seq"},
        )

        assert response.status_code == 401

    def test_upload_invalid_file_format(self, client: TestClient, auth_headers):
        """Test that uploading invalid file format fails."""
        # Create invalid file
        invalid_file = io.BytesIO(b"not a valid csv file")

        response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test.txt", invalid_file, "text/plain")},
            data={"name": "Invalid Dataset", "data_type": "bulk_rna_seq"},
        )

        # Should fail validation
        assert response.status_code in [400, 422]

    def test_upload_missing_required_fields(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that upload with missing required fields fails."""
        sample_dataset_csv.seek(0)

        response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={
                # Missing name and data_type
                "description": "Missing required fields"
            },
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.integration
class TestDataRetrieval:
    """Test suite for data retrieval workflows."""

    def test_get_dataset_by_id(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test retrieving a dataset by ID."""
        # First, upload a dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Test Dataset", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Retrieve the dataset
        response = client.get(f"/api/v1/data/{dataset_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == "Test Dataset"

    def test_get_nonexistent_dataset(self, client: TestClient, auth_headers):
        """Test that retrieving non-existent dataset returns 404."""
        fake_uuid = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/data/{fake_uuid}", headers=auth_headers)

        assert response.status_code == 404

    def test_get_dataset_without_authentication(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that retrieving dataset without auth fails."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Test Dataset", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Try to retrieve without auth
        response = client.get(f"/api/v1/data/{dataset_id}")

        assert response.status_code == 401

    def test_list_user_datasets(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test listing user's datasets."""
        # Upload multiple datasets
        for i in range(3):
            sample_dataset_csv.seek(0)
            client.post(
                "/api/v1/data/upload",
                headers=auth_headers,
                files={"file": (f"dataset_{i}.csv", sample_dataset_csv, "text/csv")},
                data={"name": f"Dataset {i}", "data_type": "bulk_rna_seq"},
            )

        # List datasets
        response = client.get("/api/v1/data/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3  # At least the 3 we just uploaded

    def test_list_datasets_pagination(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test dataset listing with pagination."""
        # Upload 5 datasets
        for i in range(5):
            sample_dataset_csv.seek(0)
            client.post(
                "/api/v1/data/upload",
                headers=auth_headers,
                files={"file": (f"dataset_{i}.csv", sample_dataset_csv, "text/csv")},
                data={"name": f"Dataset {i}", "data_type": "bulk_rna_seq"},
            )

        # Get first page (limit 3)
        response = client.get("/api/v1/data/?skip=0&limit=3", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 3

        # Get second page
        response = client.get("/api/v1/data/?skip=3&limit=3", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2  # At least 2 more


@pytest.mark.integration
class TestDataDeletion:
    """Test suite for data deletion workflows."""

    def test_delete_dataset(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test deleting a dataset."""
        # Upload dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Dataset to Delete", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Delete dataset
        response = client.delete(f"/api/v1/data/{dataset_id}", headers=auth_headers)

        assert response.status_code == 204

        # Verify deletion - should return 404
        get_response = client.get(f"/api/v1/data/{dataset_id}", headers=auth_headers)
        assert get_response.status_code == 404

    def test_delete_nonexistent_dataset(self, client: TestClient, auth_headers):
        """Test that deleting non-existent dataset returns 404."""
        fake_uuid = "00000000-0000-0000-0000-000000000000"
        response = client.delete(f"/api/v1/data/{fake_uuid}", headers=auth_headers)

        assert response.status_code == 404

    def test_delete_dataset_unauthorized(
        self, client: TestClient, auth_headers, researcher_auth_headers, sample_dataset_csv
    ):
        """Test that user cannot delete another user's dataset."""
        # User uploads dataset
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "User Dataset", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        # Researcher tries to delete user's dataset
        response = client.delete(
            f"/api/v1/data/{dataset_id}", headers=researcher_auth_headers
        )

        # Should be forbidden
        assert response.status_code in [403, 404]


@pytest.mark.integration
class TestDataValidation:
    """Test suite for data validation."""

    def test_upload_validates_data_type(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that data_type is validated."""
        sample_dataset_csv.seek(0)

        response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={
                "name": "Test Dataset",
                "data_type": "invalid_type",  # Invalid
            },
        )

        assert response.status_code == 422  # Validation error

    def test_upload_extracts_metadata(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that upload extracts dataset metadata (n_samples, n_features)."""
        sample_dataset_csv.seek(0)

        response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Metadata Test", "data_type": "bulk_rna_seq"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["n_samples"] is not None
        assert data["n_features"] is not None
        assert data["n_samples"] > 0
        assert data["n_features"] > 0
