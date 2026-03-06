import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.users.dao import UserDAO
from src.modules.users.schemas import UserCreate, UserFilter


pytestmark = pytest.mark.asyncio


class TestUserDAOAdd:
    async def test_add_and_find_by_id(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        user = await dao.add(UserCreate(name="Alice"), flush=True)
        assert user.id is not None
        assert user.name == "Alice"

        found = await dao.find_one_or_none_by_id(user.id)
        assert found is not None
        assert found.name == "Alice"

    async def test_find_by_id_not_found(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        found = await dao.find_one_or_none_by_id(9999)
        assert found is None

    async def test_add_multiple(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        u1 = await dao.add(UserCreate(name="Alice"), flush=True)
        u2 = await dao.add(UserCreate(name="Bob"), flush=True)
        assert u1.id != u2.id


class TestUserDAOFindFiltered:
    async def test_find_filtered_by_name(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        await dao.add(UserCreate(name="Alice"), flush=True)
        await dao.add(UserCreate(name="Bob"), flush=True)
        await dao.add(UserCreate(name="Alicia"), flush=True)

        results = await dao.find_filtered(UserFilter(name="ali"))
        names = [u.name for u in results]
        assert "Alice" in names
        assert "Alicia" in names
        assert "Bob" not in names

    async def test_find_filtered_no_filter(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        await dao.add(UserCreate(name="Alice"), flush=True)
        await dao.add(UserCreate(name="Bob"), flush=True)

        results = await dao.find_filtered(UserFilter())
        assert len(results) == 2

    async def test_find_filtered_case_insensitive(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        await dao.add(UserCreate(name="Alice"), flush=True)

        results = await dao.find_filtered(UserFilter(name="ALICE"))
        assert len(results) == 1

    async def test_find_filtered_no_match(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        await dao.add(UserCreate(name="Alice"), flush=True)

        results = await dao.find_filtered(UserFilter(name="zzz"))
        assert len(results) == 0

    async def test_find_filtered_with_limit_offset(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        for i in range(5):
            await dao.add(UserCreate(name=f"User{i}"), flush=True)

        results = await dao.find_filtered(UserFilter(), limit=2, offset=3)
        assert len(results) == 2


class TestUserDAOUpdate:
    async def test_update(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        user = await dao.add(UserCreate(name="Alice"), flush=True)

        updated = await dao.update(
            filters={"id": user.id},
            values={"name": "Alice Updated"},
        )
        assert updated == 1

    async def test_update_nonexistent(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        updated = await dao.update(
            filters={"id": 9999},
            values={"name": "Ghost"},
        )
        assert updated == 0

    async def test_update_empty_filters_raises(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        with pytest.raises(ValueError, match="filter criteria"):
            await dao.update(filters={}, values={"name": "X"})


class TestUserDAODelete:
    async def test_delete(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        user = await dao.add(UserCreate(name="Alice"), flush=True)
        user_id = user.id

        deleted = await dao.delete(filters={"id": user_id})
        assert deleted == 1

        db_session.expunge_all()
        found = await dao.find_one_or_none_by_id(user_id)
        assert found is None

    async def test_delete_nonexistent(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        deleted = await dao.delete(filters={"id": 9999})
        assert deleted == 0

    async def test_delete_empty_filters_raises(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        with pytest.raises(ValueError, match="filter criteria"):
            await dao.delete(filters={})
