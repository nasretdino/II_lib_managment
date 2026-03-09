import io
import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from minio import Minio
from minio.error import S3Error

from src.core import get_logger
from src.core.config import settings


logger = get_logger(module="documents", component="storage")


class ObjectStorageError(RuntimeError):
    """Raised when object storage operation fails."""


class ObjectStorage(Protocol):
    """Интерфейс для хранилища файлов."""

    async def ensure_bucket(self) -> None: ...
    async def upload_bytes(
        self, user_id: int, filename: str, content: bytes, content_type: str,
    ) -> str: ...
    async def delete_by_uri(self, uri: str) -> None: ...


class LocalStorage:
    """Хранилище файлов на локальном диске (удобно для dev)."""

    def __init__(self, base_path: str | None = None) -> None:
        self._base = Path(base_path or settings.storage.local_path)

    async def ensure_bucket(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        return Path(filename).name or "file.bin"

    async def upload_bytes(
        self,
        user_id: int,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        import asyncio

        await self.ensure_bucket()
        safe_name = self._sanitize_filename(filename)
        user_dir = self._base / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        dest = user_dir / f"{uuid4().hex}_{safe_name}"

        def _write() -> None:
            dest.write_bytes(content)

        await asyncio.to_thread(_write)
        uri = f"file://{dest}"
        logger.info("Uploaded file locally: {}", uri)
        return uri

    async def delete_by_uri(self, uri: str) -> None:
        if not uri.startswith("file://"):
            return
        path = Path(uri[len("file://"):])
        if path.exists():
            os.remove(path)
            logger.info("Deleted local file: {}", path)


class MinioStorage:
    """Thin async wrapper around MinIO client used by DocumentService."""

    def __init__(self) -> None:
        self._client = Minio(
            endpoint=settings.minio.endpoint,
            access_key=settings.minio.access_key,
            secret_key=settings.minio.secret_key.get_secret_value(),
            secure=settings.minio.use_ssl,
        )
        self._bucket = settings.minio.bucket_name
        self._bucket_checked = False

    async def ensure_bucket(self) -> None:
        if self._bucket_checked:
            return

        import asyncio

        def _ensure() -> None:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)

        await asyncio.to_thread(_ensure)
        self._bucket_checked = True

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        return Path(filename).name or "file.bin"

    def _build_object_key(self, user_id: int, filename: str) -> str:
        safe_name = self._sanitize_filename(filename)
        return f"users/{user_id}/{uuid4().hex}_{safe_name}"

    async def upload_bytes(
        self,
        user_id: int,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        import asyncio

        await self.ensure_bucket()
        object_key = self._build_object_key(user_id, filename)

        def _upload() -> None:
            self._client.put_object(
                bucket_name=self._bucket,
                object_name=object_key,
                data=io.BytesIO(content),
                length=len(content),
                content_type=content_type,
            )

        try:
            await asyncio.to_thread(_upload)
        except S3Error as exc:
            raise ObjectStorageError(f"Failed to upload object to MinIO: {exc}") from exc

        uri = f"s3://{self._bucket}/{object_key}"
        logger.info("Uploaded file to MinIO: {}", uri)
        return uri

    async def delete_by_uri(self, uri: str) -> None:
        import asyncio

        if not uri.startswith("s3://"):
            return

        no_scheme = uri[len("s3://") :]
        bucket, _, object_key = no_scheme.partition("/")
        if not bucket or not object_key:
            return

        def _delete() -> None:
            self._client.remove_object(bucket, object_key)

        try:
            await asyncio.to_thread(_delete)
        except S3Error as exc:
            logger.warning("Failed to delete object from MinIO {}: {}", uri, exc)


_storage: MinioStorage | LocalStorage | None = None


def get_object_storage() -> MinioStorage | LocalStorage:
    global _storage
    if _storage is None:
        if settings.storage.mode == "local":
            _storage = LocalStorage()
        else:
            _storage = MinioStorage()
    return _storage
