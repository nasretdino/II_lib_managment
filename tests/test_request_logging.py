import io

import pytest
from httpx import AsyncClient
from loguru import logger

from src.core import settings, setup_logging


pytestmark = pytest.mark.asyncio


class TestRequestIdMiddleware:
    async def _create_user(self, client: AsyncClient) -> int:
        resp = await client.post("/users/", json={"name": "TestUser"})
        return resp.json()["id"]

    async def test_generates_request_id_header(self, client: AsyncClient):
        resp = await client.get("/documents/", params={"user_id": 999})
        assert resp.status_code == 200
        assert "x-request-id" in resp.headers
        assert resp.headers["x-request-id"]

    async def test_propagates_request_id_header(self, client: AsyncClient):
        sent_request_id = "custom-request-id-123"
        resp = await client.get(
            "/documents/",
            params={"user_id": 999},
            headers={"X-Request-ID": sent_request_id},
        )
        assert resp.status_code == 200
        assert resp.headers["x-request-id"] == sent_request_id

    async def test_error_response_contains_request_id(self, client: AsyncClient):
        user_id = await self._create_user(client)
        resp = await client.post(
            "/documents/upload",
            params={"user_id": user_id},
            files={"file": ("img.png", io.BytesIO(b"\x89PNG"), "image/png")},
        )
        assert resp.status_code == 422
        assert "x-request-id" in resp.headers
        assert resp.headers["x-request-id"]

    async def test_request_id_is_present_in_application_logs(self, client: AsyncClient):
        sent_request_id = "rid-log-123"
        setup_logging("dev")

        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message} | rid={extra[request_id]}")
        try:
            resp = await client.post(
                "/users/",
                json={"name": "LogCheckUser"},
                headers={"X-Request-ID": sent_request_id},
            )
            assert resp.status_code == 201
        finally:
            logger.remove(sink_id)
            setup_logging(settings.env)

        assert any(
            "Creating user name=LogCheckUser" in msg and f"rid={sent_request_id}" in msg
            for msg in messages
        )
