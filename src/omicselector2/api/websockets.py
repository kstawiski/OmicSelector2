"""WebSocket support for real-time job updates.

This module provides WebSocket endpoints and connection management for
real-time job status updates.
"""

import asyncio
import logging
from typing import Dict, Set
from uuid import UUID

try:
    from fastapi import WebSocket, WebSocketDisconnect, status

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    WebSocket = None  # type: ignore
    WebSocketDisconnect = Exception  # type: ignore
    status = None  # type: ignore

try:
    from sqlalchemy.orm import Session

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Session = None  # type: ignore

from omicselector2.db import Job, User, UserRole, get_db
from omicselector2.utils.redis_pubsub import get_subscriber
from omicselector2.utils.security import decode_access_token

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manager for WebSocket connections.

    This class manages active WebSocket connections and broadcasts updates
    to connected clients.

    Attributes:
        active_connections: Dict mapping job_id to set of WebSocket connections
    """

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """Accept WebSocket connection and add to active connections.

        Args:
            websocket: WebSocket connection
            job_id: Job UUID
        """
        await websocket.accept()

        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()

        self.active_connections[job_id].add(websocket)
        logger.info(f"WebSocket connected for job {job_id}. Total connections: {len(self.active_connections[job_id])}")

    def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        """Remove WebSocket connection from active connections.

        Args:
            websocket: WebSocket connection
            job_id: Job UUID
        """
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)

            # Clean up empty sets
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

        logger.info(f"WebSocket disconnected for job {job_id}. Remaining connections: {len(self.active_connections.get(job_id, []))}")

    async def broadcast_to_job(self, job_id: str, message: dict) -> None:
        """Broadcast message to all connections for a specific job.

        Args:
            job_id: Job UUID
            message: Message dict to broadcast
        """
        if job_id not in self.active_connections:
            return

        # Create a copy to avoid modification during iteration
        connections = list(self.active_connections[job_id])

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                # Remove dead connection
                self.disconnect(connection, job_id)

    def get_connection_count(self, job_id: str) -> int:
        """Get number of active connections for a job.

        Args:
            job_id: Job UUID

        Returns:
            Number of active connections
        """
        return len(self.active_connections.get(job_id, set()))


# Global connection manager instance
manager = ConnectionManager()


async def verify_websocket_auth(websocket: WebSocket, token: str, db: Session) -> User:
    """Verify JWT token for WebSocket connection.

    Args:
        websocket: WebSocket connection
        token: JWT token from query parameter
        db: Database session

    Returns:
        Authenticated user

    Raises:
        WebSocketDisconnect: If authentication fails
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for WebSocket support")

    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is required for database access")

    # Decode JWT token
    payload = decode_access_token(token)

    if payload is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)

    # Extract user ID from token
    user_id = payload.get("sub")
    if user_id is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)

    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()

    if user is None or not user.is_active:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)

    return user


async def verify_job_access(user: User, job_id: str, db: Session) -> Job:
    """Verify user has access to the specified job.

    Args:
        user: Authenticated user
        job_id: Job UUID
        db: Database session

    Returns:
        Job instance

    Raises:
        WebSocketDisconnect: If job not found or access denied
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for WebSocket support")

    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is required for database access")

    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise WebSocketDisconnect(code=status.WS_1003_UNSUPPORTED_DATA)

    # Get job
    job = db.query(Job).filter(Job.id == job_uuid).first()

    if not job:
        raise WebSocketDisconnect(code=status.WS_1003_UNSUPPORTED_DATA)

    # Check access (owner or admin)
    if job.user_id != user.id and user.role != UserRole.ADMIN:
        raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)

    return job


async def handle_job_updates(job_id: str) -> None:
    """Subscribe to Redis and broadcast job updates to WebSocket clients.

    Args:
        job_id: Job UUID
    """
    subscriber = get_subscriber()

    try:
        await subscriber.connect()
        await subscriber.subscribe_to_job(job_id)

        logger.info(f"Listening for updates on job {job_id}")

        async for update in subscriber.listen():
            logger.debug(f"Received update for job {job_id}: {update}")
            await manager.broadcast_to_job(job_id, update)

    except asyncio.CancelledError:
        logger.info(f"Update listener cancelled for job {job_id}")
    except Exception as e:
        logger.error(f"Error in update listener for job {job_id}: {e}")
    finally:
        await subscriber.disconnect()


__all__ = [
    "ConnectionManager",
    "manager",
    "verify_websocket_auth",
    "verify_job_access",
    "handle_job_updates",
]
