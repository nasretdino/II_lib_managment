import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


class TestUsersAPI:
    # ── Create ────────────────────────────────────────────

    async def test_create_user(self, client: AsyncClient):
        resp = await client.post("/users/", json={"name": "Alice"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Alice"
        assert "id" in data
        assert "created_at" in data

    async def test_create_user_invalid_empty_name(self, client: AsyncClient):
        resp = await client.post("/users/", json={"name": ""})
        assert resp.status_code == 422

    async def test_create_user_missing_name(self, client: AsyncClient):
        resp = await client.post("/users/", json={})
        assert resp.status_code == 422

    async def test_create_user_whitespace_only_name(self, client: AsyncClient):
        resp = await client.post("/users/", json={"name": "   "})
        assert resp.status_code == 422

    async def test_create_user_strips_whitespace(self, client: AsyncClient):
        resp = await client.post("/users/", json={"name": "  Bob  "})
        assert resp.status_code == 201
        assert resp.json()["name"] == "Bob"

    async def test_create_user_too_long_name(self, client: AsyncClient):
        resp = await client.post("/users/", json={"name": "A" * 256})
        assert resp.status_code == 422

    async def test_create_user_max_length_name(self, client: AsyncClient):
        resp = await client.post("/users/", json={"name": "A" * 255})
        assert resp.status_code == 201

    # ── Get list ──────────────────────────────────────────

    async def test_get_users(self, client: AsyncClient):
        await client.post("/users/", json={"name": "Alice"})
        await client.post("/users/", json={"name": "Bob"})

        resp = await client.get("/users/")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_get_users_empty(self, client: AsyncClient):
        resp = await client.get("/users/")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_users_with_filter(self, client: AsyncClient):
        await client.post("/users/", json={"name": "Alice"})
        await client.post("/users/", json={"name": "Bob"})

        resp = await client.get("/users/", params={"name": "alice"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Alice"

    async def test_pagination(self, client: AsyncClient):
        for i in range(5):
            await client.post("/users/", json={"name": f"User{i}"})

        resp = await client.get("/users/", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        assert len(resp.json()) == 2

        resp = await client.get("/users/", params={"limit": 2, "offset": 4})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_pagination_invalid_limit(self, client: AsyncClient):
        resp = await client.get("/users/", params={"limit": 0})
        assert resp.status_code == 422

    async def test_pagination_negative_offset(self, client: AsyncClient):
        resp = await client.get("/users/", params={"offset": -1})
        assert resp.status_code == 422

    # ── Get by ID ─────────────────────────────────────────

    async def test_get_user_by_id(self, client: AsyncClient):
        create_resp = await client.post("/users/", json={"name": "Alice"})
        user_id = create_resp.json()["id"]

        resp = await client.get(f"/users/{user_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Alice"

    async def test_get_user_not_found(self, client: AsyncClient):
        resp = await client.get("/users/9999")
        assert resp.status_code == 404

    async def test_get_user_invalid_id(self, client: AsyncClient):
        resp = await client.get("/users/abc")
        assert resp.status_code == 422

    async def test_get_user_zero_id(self, client: AsyncClient):
        resp = await client.get("/users/0")
        assert resp.status_code == 422

    async def test_get_user_negative_id(self, client: AsyncClient):
        resp = await client.get("/users/-1")
        assert resp.status_code == 422

    # ── Update ────────────────────────────────────────────

    async def test_update_user(self, client: AsyncClient):
        create_resp = await client.post("/users/", json={"name": "Alice"})
        user_id = create_resp.json()["id"]

        resp = await client.patch(f"/users/{user_id}", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    async def test_update_user_not_found(self, client: AsyncClient):
        resp = await client.patch("/users/9999", json={"name": "Ghost"})
        assert resp.status_code == 404

    async def test_update_user_empty_body(self, client: AsyncClient):
        """PATCH без полей → возвращает текущие данные."""
        create_resp = await client.post("/users/", json={"name": "Alice"})
        user_id = create_resp.json()["id"]

        resp = await client.patch(f"/users/{user_id}", json={})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Alice"

    async def test_update_user_invalid_name(self, client: AsyncClient):
        create_resp = await client.post("/users/", json={"name": "Alice"})
        user_id = create_resp.json()["id"]

        resp = await client.patch(f"/users/{user_id}", json={"name": ""})
        assert resp.status_code == 422

    async def test_update_user_zero_id(self, client: AsyncClient):
        resp = await client.patch("/users/0", json={"name": "X"})
        assert resp.status_code == 422

    # ── Delete ────────────────────────────────────────────

    async def test_delete_user(self, client: AsyncClient):
        create_resp = await client.post("/users/", json={"name": "Alice"})
        user_id = create_resp.json()["id"]

        resp = await client.delete(f"/users/{user_id}")
        assert resp.status_code == 204

        get_resp = await client.get(f"/users/{user_id}")
        assert get_resp.status_code == 404

    async def test_delete_user_not_found(self, client: AsyncClient):
        resp = await client.delete("/users/9999")
        assert resp.status_code == 404

    async def test_delete_user_zero_id(self, client: AsyncClient):
        resp = await client.delete("/users/0")
        assert resp.status_code == 422

    # ── Response shape ────────────────────────────────────

    async def test_response_contains_all_fields(self, client: AsyncClient):
        resp = await client.post("/users/", json={"name": "Full"})
        data = resp.json()
        assert "id" in data
        assert "name" in data
        assert "created_at" in data
        assert "updated_at" in data
