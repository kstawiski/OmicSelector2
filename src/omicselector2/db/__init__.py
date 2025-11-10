"""Database models and utilities for OmicSelector2.

This module exports all database models and provides database initialization.
"""

# Export database utilities
from omicselector2.db.database import Base, get_db, get_engine, init_db

# Export enums
from omicselector2.db.dataset import DataType
from omicselector2.db.job import JobStatus, JobType
from omicselector2.db.user import UserRole

# Export models
from omicselector2.db.dataset import Dataset
from omicselector2.db.job import Job
from omicselector2.db.result import Result
from omicselector2.db.user import User

__all__ = [
    # Database utilities
    "Base",
    "get_db",
    "get_engine",
    "init_db",
    # Enums
    "UserRole",
    "DataType",
    "JobType",
    "JobStatus",
    # Models
    "User",
    "Dataset",
    "Job",
    "Result",
]
