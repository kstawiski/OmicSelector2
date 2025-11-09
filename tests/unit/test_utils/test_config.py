"""Tests for configuration management."""

import os

import pytest


@pytest.mark.unit
def test_settings_can_be_imported() -> None:
    """Test that Settings class can be imported."""
    from omicselector2.utils.config import Settings

    assert Settings is not None


@pytest.mark.unit
def test_settings_has_required_fields() -> None:
    """Test that Settings has all required configuration fields."""
    from omicselector2.utils.config import Settings

    # Create settings with minimal required env vars
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-12345678"
    os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost/test"

    settings = Settings()

    # Verify core fields exist
    assert hasattr(settings, "SECRET_KEY")
    assert hasattr(settings, "DATABASE_URL")
    assert hasattr(settings, "ENVIRONMENT")
    assert hasattr(settings, "DEBUG")
    assert hasattr(settings, "API_HOST")
    assert hasattr(settings, "API_PORT")


@pytest.mark.unit
def test_settings_loads_from_env() -> None:
    """Test that Settings loads values from environment variables."""
    from omicselector2.utils.config import Settings

    # Set test environment variables
    os.environ["SECRET_KEY"] = "my-test-secret-key-16chars"
    os.environ["DATABASE_URL"] = "postgresql://testuser:testpass@testhost/testdb"
    os.environ["ENVIRONMENT"] = "development"

    settings = Settings()

    assert settings.SECRET_KEY == "my-test-secret-key-16chars"
    assert settings.DATABASE_URL == "postgresql://testuser:testpass@testhost/testdb"
    assert settings.ENVIRONMENT == "development"


@pytest.mark.unit
def test_settings_has_sensible_defaults() -> None:
    """Test that Settings provides sensible default values."""
    from omicselector2.utils.config import Settings

    os.environ["SECRET_KEY"] = "test-key-with-16-characters"
    os.environ["DATABASE_URL"] = "postgresql://localhost/test"
    # Don't set ENVIRONMENT so it uses default

    settings = Settings()

    # Check defaults
    assert settings.API_HOST == "0.0.0.0"
    assert settings.API_PORT == 8000
    assert settings.ENVIRONMENT in ["development", "staging", "production"]
    assert isinstance(settings.DEBUG, bool)


@pytest.mark.unit
def test_get_settings_returns_singleton() -> None:
    """Test that get_settings returns the same instance."""
    from omicselector2.utils.config import get_settings

    os.environ["SECRET_KEY"] = "test-key-singleton-16chars"
    os.environ["DATABASE_URL"] = "postgresql://localhost/test"

    # Clear the lru_cache to ensure fresh instances
    get_settings.cache_clear()

    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2
