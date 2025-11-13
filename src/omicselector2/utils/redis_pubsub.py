"""Redis Pub/Sub utilities for real-time job updates.

This module provides utilities for publishing and subscribing to job status updates
via Redis pub/sub channels.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore

from omicselector2.utils.config import get_settings

logger = logging.getLogger(__name__)


class RedisPublisher:
    """Publisher for job status updates via Redis pub/sub.

    Attributes:
        redis_client: Redis async client instance
    """

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis publisher.

        Args:
            redis_url: Redis connection URL (optional, defaults from settings)

        Raises:
            ImportError: If redis is not installed
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis is required for pub/sub. " "Install with: pip install redis"
            )

        settings = get_settings()
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Connect to Redis."""
        if self.redis_client is None:
            self.redis_client = await redis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

    async def publish_job_update(
        self, job_id: str, status: str, message: Optional[str] = None, metadata: Optional[dict] = None
    ) -> None:
        """Publish a job status update.

        Args:
            job_id: Job UUID
            status: Job status (pending, running, completed, failed, cancelled)
            message: Optional status message
            metadata: Optional metadata dict

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        channel = f"job_updates:{job_id}"

        payload = {
            "job_id": job_id,
            "status": status,
            "message": message,
            "metadata": metadata or {},
        }

        try:
            await self.redis_client.publish(channel, json.dumps(payload))
            logger.debug(f"Published update to {channel}: {payload}")
        except Exception as e:
            logger.error(f"Failed to publish job update: {e}")
            raise


class RedisSubscriber:
    """Subscriber for job status updates via Redis pub/sub.

    Attributes:
        redis_client: Redis async client instance
        pubsub: Redis pub/sub instance
    """

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis subscriber.

        Args:
            redis_url: Redis connection URL (optional, defaults from settings)

        Raises:
            ImportError: If redis is not installed
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis is required for pub/sub. " "Install with: pip install redis"
            )

        settings = get_settings()
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None

    async def connect(self) -> None:
        """Connect to Redis and create pub/sub instance."""
        if self.redis_client is None:
            self.redis_client = await redis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
            self.pubsub = self.redis_client.pubsub()

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
            self.pubsub = None
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

    async def subscribe_to_job(self, job_id: str) -> None:
        """Subscribe to job updates for a specific job.

        Args:
            job_id: Job UUID

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self.pubsub:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        channel = f"job_updates:{job_id}"
        await self.pubsub.subscribe(channel)
        logger.debug(f"Subscribed to {channel}")

    async def listen(self) -> AsyncGenerator[dict[str, Any], None]:
        """Listen for job updates and yield them.

        Yields:
            Job update messages as dictionaries

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self.pubsub:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        yield data
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode message: {e}")
                        continue
        except asyncio.CancelledError:
            logger.info("Subscription cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in subscription: {e}")
            raise


def get_publisher() -> RedisPublisher:
    """Get a Redis publisher instance.

    Returns:
        RedisPublisher instance

    Raises:
        ImportError: If redis is not installed
    """
    return RedisPublisher()


def get_subscriber() -> RedisSubscriber:
    """Get a Redis subscriber instance.

    Returns:
        RedisSubscriber instance

    Raises:
        ImportError: If redis is not installed
    """
    return RedisSubscriber()


__all__ = [
    "RedisPublisher",
    "RedisSubscriber",
    "get_publisher",
    "get_subscriber",
]
