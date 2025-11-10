"""FastAPI application for OmicSelector2.

This module defines the main FastAPI application with routes, middleware,
and configuration for the OmicSelector2 API.

Examples:
    Run the application:
    >>> uvicorn omicselector2.api.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from omicselector2 import __version__
from omicselector2.utils.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application lifespan events.

    Args:
        app: FastAPI application instance.

    Yields:
        None during application runtime.
    """
    # Startup
    settings = get_settings()
    print(f"🚀 Starting OmicSelector2 v{__version__}")
    print(f"📊 Environment: {settings.ENVIRONMENT}")
    print(f"🔧 Debug mode: {settings.DEBUG}")

    yield

    # Shutdown
    print("👋 Shutting down OmicSelector2")


# Create FastAPI application
app = FastAPI(
    title="OmicSelector2 API",
    description="Next-generation platform for multi-omic biomarker discovery in oncology",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Get settings
settings = get_settings()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
from omicselector2.api.routes import auth, data

app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["authentication"],
)
app.include_router(
    data.router,
    prefix="/api/v1/data",
    tags=["data"],
)


@app.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    """Root endpoint - returns API status.

    Returns:
        Dictionary with API status and version information.

    Examples:
        >>> # GET /
        >>> {"status": "healthy", "version": "0.1.0", "message": "OmicSelector2 API"}
    """
    return {
        "status": "healthy",
        "version": __version__,
        "message": "OmicSelector2 API - Next-generation multi-omic biomarker discovery",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> JSONResponse:
    """Health check endpoint for monitoring.

    Returns:
        JSON response with health status.

    Examples:
        >>> # GET /health
        >>> {"status": "healthy", "environment": "development"}
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
            "version": __version__,
        },
    )


@app.get("/api/v1/info", tags=["Info"])
async def api_info() -> dict[str, object]:
    """Get API information and configuration.

    Returns:
        Dictionary with API configuration information.

    Examples:
        >>> # GET /api/v1/info
        >>> {
        ...     "version": "0.1.0",
        ...     "environment": "development",
        ...     "features": {...}
        ... }
    """
    return {
        "version": __version__,
        "environment": settings.ENVIRONMENT,
        "features": {
            "feature_selection": True,
            "multi_omics": True,
            "graph_neural_networks": True,
            "single_cell": True,
        },
        "limits": {
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
            "max_concurrent_jobs": settings.MAX_CONCURRENT_JOBS_PER_USER,
            "max_features": settings.MAX_FEATURES_IN_DATASET,
        },
        "defaults": {
            "cv_folds": settings.DEFAULT_CV_FOLDS,
            "n_features": settings.DEFAULT_N_FEATURES,
            "stability_threshold": settings.DEFAULT_STABILITY_THRESHOLD,
        },
    }


# Health check for Docker/K8s
@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Kubernetes-style health check endpoint.

    Returns:
        Simple OK response for health probes.
    """
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz() -> dict[str, str]:
    """Kubernetes-style readiness check endpoint.

    Returns:
        Simple OK response for readiness probes.
    """
    return {"status": "ok"}
