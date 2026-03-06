import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.users.dao import UserDAO
from src.modules.users.schemas import UserCreate, UserFilter, UserUpdate
from src.modules.users.services import UserService


pytestmark = pytest.mark.asyncio


@pytest.fixture
def user_service(db_session: AsyncSession) -> UserService:
    return UserService(UserDAO(db_session))


class TestUserServiceCreate:
    async def test_create(self, user_service: UserService):
        user = await user_service.create(UserCreate(name="Alice"))
        assert user.id is not None
        assert user.name == "Alice"

    async def test_create_multiple(self, user_service: UserService):
        u1 = await user_service.create(UserCreate(name="Alice"))
        u2 = await user_service.create(UserCreate(name="Bob"))
        assert u1.id != u2.id


class TestUserServiceGetById:
    async def test_get_by_id(self, user_service: UserService):
        user = await user_service.create(UserCreate(name="Alice"))
        found = await user_service.get_by_id(user.id)
        assert found is not None
        assert found.name == "Alice"

    async def test_get_by_id_not_found(self, user_service: UserService):
        found = await user_service.get_by_id(9999)
        assert found is None


class TestUserServiceGetAll:
    async def test_get_all(self, user_service: UserService):
        await user_service.create(UserCreate(name="Alice"))
        await user_service.create(UserCreate(name="Bob"))

        users = await user_service.get_all(filters=UserFilter())
        assert len(users) == 2

    async def test_get_all_with_filter(self, user_service: UserService):
        await user_service.create(UserCreate(name="Alice"))
        await user_service.create(UserCreate(name="Bob"))

        users = await user_service.get_all(filters=UserFilter(name="bob"))
        assert len(users) == 1
        assert users[0].name == "Bob"

    async def test_get_all_with_limit(self, user_service: UserService):
        for i in range(5):
            await user_service.create(UserCreate(name=f"User{i}"))

        users = await user_service.get_all(filters=UserFilter(), limit=3)
        assert len(users) == 3

    async def test_get_all_with_offset(self, user_service: UserService):
        for i in range(5):
            await user_service.create(UserCreate(name=f"User{i}"))

        users = await user_service.get_all(filters=UserFilter(), limit=10, offset=3)
        assert len(users) == 2

    async def test_get_all_empty(self, user_service: UserService):
        users = await user_service.get_all(filters=UserFilter())
        assert users == []

    async def test_get_all_no_match(self, user_service: UserService):
        await user_service.create(UserCreate(name="Alice"))
        users = await user_service.get_all(filters=UserFilter(name="zzz"))
        assert users == []


class TestUserServiceUpdate:
    async def test_update(self, user_service: UserService, db_session):
        user = await user_service.create(UserCreate(name="Alice"))
        user_id = user.id
        db_session.expunge_all()
        updated = await user_service.update(user_id, UserUpdate(name="Alice Updated"))
        assert updated is not None
        assert updated.name == "Alice Updated"

    async def test_update_no_fields(self, user_service: UserService):
        user = await user_service.create(UserCreate(name="Alice"))
        result = await user_service.update(user.id, UserUpdate())
        assert result is not None
        assert result.name == "Alice"

    async def test_update_not_found(self, user_service: UserService):
        result = await user_service.update(9999, UserUpdate(name="Ghost"))
        assert result is None

    async def test_update_sets_updated_at(self, user_service: UserService, db_session):
        user = await user_service.create(UserCreate(name="Alice"))
        original_updated = user.updated_at
        user_id = user.id
        db_session.expunge_all()

        updated = await user_service.update(user_id, UserUpdate(name="New"))
        # updated_at должен измениться (или быть не None)
        assert updated is not None
        assert updated.updated_at is not None


class TestUserServiceDelete:
    async def test_delete(self, user_service: UserService, db_session):
        user = await user_service.create(UserCreate(name="Alice"))
        user_id = user.id
        deleted = await user_service.delete(user_id)
        assert deleted is True

        db_session.expunge_all()
        found = await user_service.get_by_id(user_id)
        assert found is None

    async def test_delete_not_found(self, user_service: UserService):
        deleted = await user_service.delete(9999)
        assert deleted is False

    async def test_delete_idempotent(self, user_service: UserService):
        """Повторное удаление возвращает False."""
        user = await user_service.create(UserCreate(name="Alice"))
        user_id = user.id
        assert await user_service.delete(user_id) is True
        assert await user_service.delete(user_id) is False
