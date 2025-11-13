"""
Pytest configuration and fixtures for OmicSelector2 tests.

This module provides shared fixtures and configuration for all tests,
including mocked settings, database sessions, and test data.
"""

import os
import pytest
from typing import Generator
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    """
    Set up test environment variables.

    This fixture runs once per test session and sets required environment
    variables for testing (SECRET_KEY, DATABASE_URL, etc.).
    """
    # Set test environment variables
    os.environ["SECRET_KEY"] = "test-secret-key-for-jwt-tokens-minimum-32-characters-required"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test_omicselector2"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    os.environ["MINIO_ENDPOINT"] = "localhost:9000"
    os.environ["MINIO_ACCESS_KEY"] = "test_access_key"
    os.environ["MINIO_SECRET_KEY"] = "test_secret_key"
    os.environ["ENVIRONMENT"] = "development"  # Must be 'development', 'staging', or 'production'

    yield

    # Cleanup (optional)
    # Remove test environment variables if needed


@pytest.fixture
def mock_settings():
    """
    Mock the settings object for testing.

    Returns a mock settings object with test values.
    """
    from omicselector2.utils.config import Settings

    # Create a test settings object
    settings = Settings(
        SECRET_KEY="test-secret-key-for-jwt-tokens-minimum-32-characters-required",
        DATABASE_URL="postgresql://test:test@localhost/test_omicselector2",
        REDIS_URL="redis://localhost:6379/0",
        MINIO_ENDPOINT="localhost:9000",
        MINIO_ACCESS_KEY="test_access_key",
        MINIO_SECRET_KEY="test_secret_key",
        ENVIRONMENT="development",
    )

    return settings


@pytest.fixture
def mock_get_settings(mock_settings):
    """
    Patch get_settings() to return mock settings.

    This fixture patches the get_settings function to return test settings.
    """
    with patch("omicselector2.utils.config.get_settings", return_value=mock_settings):
        yield mock_settings
