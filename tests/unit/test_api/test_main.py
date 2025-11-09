"""Tests for main FastAPI application."""

import os

import pytest


@pytest.fixture(autouse=True)
def set_test_env_vars() -> None:
    """Set required environment variables for testing."""
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"


@pytest.mark.unit
def test_app_can_be_imported() -> None:
    """Test that FastAPI app can be imported."""
    from omicselector2.api.main import app

    assert app is not None
    assert app.title == "OmicSelector2 API"


@pytest.mark.unit
def test_app_has_correct_version() -> None:
    """Test that app version matches package version."""
    from omicselector2 import __version__
    from omicselector2.api.main import app

    assert app.version == __version__


@pytest.mark.unit
def test_app_has_docs_endpoints() -> None:
    """Test that app has documentation endpoints configured."""
    from omicselector2.api.main import app

    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"
    assert app.openapi_url == "/openapi.json"
