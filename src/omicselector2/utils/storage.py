"""S3/MinIO storage utilities for file management.

This module provides utilities for uploading, downloading, and managing files
in S3-compatible object storage (AWS S3 or MinIO).
"""

import io
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore

from omicselector2.utils.config import get_settings


class StorageClient:
    """Client for S3/MinIO object storage operations.

    This class provides methods for uploading, downloading, and managing files
    in S3-compatible storage.

    Attributes:
        client: boto3 S3 client
        bucket_name: Default bucket name
    """

    def __init__(self, bucket_name: Optional[str] = None):
        """Initialize storage client.

        Args:
            bucket_name: S3 bucket name (optional, defaults from settings)

        Raises:
            ImportError: If boto3 is not installed
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for storage operations. "
                "Install with: pip install boto3"
            )

        settings = get_settings()

        self.client = boto3.client(
            "s3",
            endpoint_url=settings.AWS_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME

        # Create bucket if it doesn't exist
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Ensure the bucket exists, create if not."""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            # Bucket doesn't exist, create it
            try:
                self.client.create_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                # Ignore if bucket already exists (race condition)
                if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                    raise

    def upload_file(
        self, file_obj: io.BytesIO, object_name: str, metadata: Optional[dict] = None
    ) -> str:
        """Upload a file to S3.

        Args:
            file_obj: File-like object to upload
            object_name: S3 object key (path in bucket)
            metadata: Optional metadata dict

        Returns:
            S3 URL of uploaded file

        Raises:
            ClientError: If upload fails
        """
        extra_args = {}
        if metadata:
            extra_args["Metadata"] = metadata

        self.client.upload_fileobj(
            file_obj, self.bucket_name, object_name, ExtraArgs=extra_args
        )

        return f"s3://{self.bucket_name}/{object_name}"

    def download_file(self, object_name: str) -> io.BytesIO:
        """Download a file from S3.

        Args:
            object_name: S3 object key

        Returns:
            File content as BytesIO

        Raises:
            ClientError: If download fails
        """
        file_obj = io.BytesIO()
        self.client.download_fileobj(self.bucket_name, object_name, file_obj)
        file_obj.seek(0)  # Reset to beginning
        return file_obj

    def delete_file(self, object_name: str) -> None:
        """Delete a file from S3.

        Args:
            object_name: S3 object key

        Raises:
            ClientError: If deletion fails
        """
        self.client.delete_object(Bucket=self.bucket_name, Key=object_name)

    def file_exists(self, object_name: str) -> bool:
        """Check if a file exists in S3.

        Args:
            object_name: S3 object key

        Returns:
            True if file exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except ClientError:
            return False

    def get_file_metadata(self, object_name: str) -> Optional[dict]:
        """Get metadata for a file in S3.

        Args:
            object_name: S3 object key

        Returns:
            Metadata dict or None if file doesn't exist
        """
        try:
            response = self.client.head_object(Bucket=self.bucket_name, Key=object_name)
            return response.get("Metadata", {})
        except ClientError:
            return None

    def list_files(self, prefix: str = "") -> list[str]:
        """List files in bucket with optional prefix.

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of object keys
        """
        response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

        if "Contents" not in response:
            return []

        return [obj["Key"] for obj in response["Contents"]]

    def get_presigned_url(
        self, object_name: str, expiration: int = 3600
    ) -> Optional[str]:
        """Generate a presigned URL for downloading a file.

        Args:
            object_name: S3 object key
            expiration: URL expiration time in seconds (default 1 hour)

        Returns:
            Presigned URL or None if generation fails
        """
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_name},
                ExpiresIn=expiration,
            )
            return url
        except ClientError:
            return None


def get_storage_client(bucket_name: Optional[str] = None) -> StorageClient:
    """Get a storage client instance.

    Args:
        bucket_name: Optional bucket name

    Returns:
        StorageClient instance

    Raises:
        ImportError: If boto3 is not installed
    """
    return StorageClient(bucket_name=bucket_name)


__all__ = [
    "StorageClient",
    "get_storage_client",
]
