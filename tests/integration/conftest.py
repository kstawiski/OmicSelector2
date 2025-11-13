"""Fixtures for integration tests.

This module provides shared fixtures for API integration tests including:
- Test database setup/teardown
- Test FastAPI client
- Authentication helpers
- Sample datasets
"""

import io
import os
import tempfile
from typing import Generator
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from omicselector2.api.main import app
from omicselector2.db import Base, User, UserRole, get_db
from omicselector2.utils.security import hash_password


# Test database URL (SQLite for fast in-memory tests)
TEST_DATABASE_URL = "sqlite:///./test.db"


@pytest.fixture(scope="function")
def test_db() -> Generator[Session, None, None]:
    """Create test database and session.

    Yields:
        Database session for testing

    Notes:
        - Creates tables before each test
        - Drops tables after each test
        - Uses SQLite for speed
    """
    # Create test database engine
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db: Session) -> TestClient:
    """Create test FastAPI client with test database.

    Args:
        test_db: Test database session

    Returns:
        TestClient instance
    """

    # Override database dependency
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Create test client
    with TestClient(app) as test_client:
        yield test_client

    # Clear overrides
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_user(test_db: Session) -> User:
    """Create test user in database.

    Args:
        test_db: Test database session

    Returns:
        Test user instance
    """
    user = User(
        id=uuid4(),
        email="test@example.com",
        username="testuser",
        hashed_password=hash_password("testpassword"),
        full_name="Test User",
        role=UserRole.USER,
        is_active=True,
    )

    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)

    return user


@pytest.fixture(scope="function")
def test_researcher(test_db: Session) -> User:
    """Create test researcher user in database.

    Args:
        test_db: Test database session

    Returns:
        Test researcher user instance
    """
    user = User(
        id=uuid4(),
        email="researcher@example.com",
        username="researcher",
        hashed_password=hash_password("researcherpassword"),
        full_name="Test Researcher",
        role=UserRole.RESEARCHER,
        is_active=True,
    )

    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)

    return user


@pytest.fixture(scope="function")
def test_admin(test_db: Session) -> User:
    """Create test admin user in database.

    Args:
        test_db: Test database session

    Returns:
        Test admin user instance
    """
    user = User(
        id=uuid4(),
        email="admin@example.com",
        username="admin",
        hashed_password=hash_password("adminpassword"),
        full_name="Test Admin",
        role=UserRole.ADMIN,
        is_active=True,
    )

    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)

    return user


@pytest.fixture(scope="function")
def auth_headers(client: TestClient, test_user: User) -> dict:
    """Get authentication headers with valid JWT token.

    Args:
        client: Test client
        test_user: Test user

    Returns:
        Dict with Authorization header
    """
    # Login to get token
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "testuser", "password": "testpassword"},
    )

    assert response.status_code == 200
    token = response.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def researcher_auth_headers(client: TestClient, test_researcher: User) -> dict:
    """Get authentication headers for researcher user.

    Args:
        client: Test client
        test_researcher: Test researcher user

    Returns:
        Dict with Authorization header
    """
    # Login to get token
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "researcher", "password": "researcherpassword"},
    )

    assert response.status_code == 200
    token = response.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def admin_auth_headers(client: TestClient, test_admin: User) -> dict:
    """Get authentication headers for admin user.

    Args:
        client: Test client
        test_admin: Test admin user

    Returns:
        Dict with Authorization header
    """
    # Login to get token
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "adminpassword"},
    )

    assert response.status_code == 200
    token = response.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def sample_dataset_csv() -> io.BytesIO:
    """Create sample CSV dataset for testing.

    Returns:
        BytesIO object containing CSV data

    Notes:
        Dataset has 100 samples and 20 features + 1 target
    """
    import numpy as np

    np.random.seed(42)

    # Generate synthetic data
    n_samples = 100
    n_features = 20

    # Features
    X = np.random.randn(n_samples, n_features)

    # Target (binary classification)
    y = np.random.binomial(1, 0.5, n_samples)

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Convert to CSV in memory
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return csv_buffer


@pytest.fixture(scope="function")
def sample_regression_dataset() -> io.BytesIO:
    """Create sample regression dataset for testing.

    Returns:
        BytesIO object containing CSV data
    """
    import numpy as np

    np.random.seed(42)

    # Generate synthetic data
    n_samples = 100
    n_features = 15

    # Features
    X = np.random.randn(n_samples, n_features)

    # Target (continuous)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Convert to CSV in memory
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return csv_buffer


@pytest.fixture(scope="function")
def temp_storage_dir() -> Generator[str, None, None]:
    """Create temporary directory for file storage during tests.

    Yields:
        Path to temporary directory

    Notes:
        - Directory is created before test
        - Directory and contents are removed after test
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_redis: mark test as requiring Redis")
    config.addinivalue_line("markers", "requires_s3: mark test as requiring S3/MinIO")
