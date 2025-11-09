"""Configuration management using Pydantic Settings.

This module provides centralized configuration management for OmicSelector2
using environment variables and .env files.

Examples:
    >>> from omicselector2.utils.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.API_HOST)
    '0.0.0.0'
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    Required settings must be provided via environment variables.

    Attributes:
        SECRET_KEY: Secret key for JWT token signing (required)
        DATABASE_URL: PostgreSQL connection string (required)
        REDIS_URL: Redis connection string for Celery
        ENVIRONMENT: Deployment environment (development/staging/production)
        DEBUG: Enable debug mode
        API_HOST: API server host address
        API_PORT: API server port
        CORS_ORIGINS: Allowed CORS origins
        LOG_LEVEL: Logging level
        MAX_UPLOAD_SIZE_MB: Maximum file upload size in MB
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Security (Required)
    SECRET_KEY: str = Field(
        ...,
        description="Secret key for JWT token signing",
        min_length=16,
    )

    # Database (Required)
    DATABASE_URL: str = Field(
        ...,
        description="PostgreSQL connection string",
    )

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL",
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend URL",
    )

    # Environment
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # API Configuration
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    API_PORT: int = Field(
        default=8000,
        description="API server port",
        ge=1,
        le=65535,
    )
    API_WORKERS: int = Field(
        default=4,
        description="Number of API workers",
        ge=1,
    )

    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Comma-separated list of allowed CORS origins",
    )

    # S3/MinIO
    AWS_ACCESS_KEY_ID: str = Field(
        default="minioadmin",
        description="AWS/MinIO access key",
    )
    AWS_SECRET_ACCESS_KEY: str = Field(
        default="minioadmin",
        description="AWS/MinIO secret key",
    )
    AWS_ENDPOINT_URL: str = Field(
        default="http://localhost:9000",
        description="S3/MinIO endpoint URL",
    )
    S3_BUCKET_NAME: str = Field(
        default="omicselector2",
        description="S3 bucket name",
    )

    # MLflow
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    LOG_FORMAT: Literal["json", "text"] = Field(
        default="json",
        description="Log output format",
    )

    # Resource Limits
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=5000,
        description="Maximum file upload size in MB",
        ge=1,
    )
    MAX_CONCURRENT_JOBS_PER_USER: int = Field(
        default=5,
        description="Maximum concurrent jobs per user",
        ge=1,
    )
    MAX_FEATURES_IN_DATASET: int = Field(
        default=100000,
        description="Maximum number of features in dataset",
        ge=1,
    )

    # Feature Selection Defaults
    DEFAULT_CV_FOLDS: int = Field(
        default=5,
        description="Default number of cross-validation folds",
        ge=2,
        le=10,
    )
    DEFAULT_N_FEATURES: int = Field(
        default=100,
        description="Default number of features to select",
        ge=1,
    )
    DEFAULT_STABILITY_THRESHOLD: float = Field(
        default=0.7,
        description="Default stability selection threshold",
        ge=0.0,
        le=1.0,
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list.

        Returns:
            List of allowed CORS origin URLs.
        """
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment.

        Returns:
            True if ENVIRONMENT is 'production', False otherwise.
        """
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment.

        Returns:
            True if ENVIRONMENT is 'development', False otherwise.
        """
        return self.ENVIRONMENT == "development"


@lru_cache
def get_settings() -> Settings:
    """Get application settings singleton.

    Uses lru_cache to ensure only one Settings instance is created.

    Returns:
        Settings instance with loaded configuration.

    Examples:
        >>> settings = get_settings()
        >>> print(settings.DATABASE_URL)
        'postgresql://...'
    """
    return Settings()
