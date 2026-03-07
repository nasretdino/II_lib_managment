"""
Тесты для BaseDAO и Base — покрытие функциональности,
которая не тестируется через конкретные DAO (UserDAO, DocumentDAO).
"""

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.base_dao import BaseDAO
from src.db.base_model import Base
from src.modules.users.dao import UserDAO
from src.modules.users.schemas import UserCreate


pytestmark = pytest.mark.asyncio


# ── BaseDAO.__init__ ─────────────────────────────────────


class TestBaseDAOInit:
    async def test_model_none_raises_value_error(self, db_session: AsyncSession):
        """BaseDAO без указания model выбрасывает ValueError."""

        class BadDAO(BaseDAO):
            model = None

        with pytest.raises(ValueError, match="must be explicitly defined"):
            BadDAO(db_session)

    async def test_model_set_works(self, db_session: AsyncSession):
        """BaseDAO с указанной моделью работает."""
        dao = UserDAO(db_session)
        assert dao.model is not None


# ── BaseDAO.add_many ──────────────────────────────────────


class TestBaseDAOAddMany:
    async def test_add_many_empty_list(self, db_session: AsyncSession):
        """Пустой список → None, без вызова insert."""
        dao = UserDAO(db_session)
        result = await dao.add_many([])
        assert result is None

    async def test_add_many_without_return(self, db_session: AsyncSession):
        """add_many без return_objects → None, записи в БД."""
        dao = UserDAO(db_session)
        result = await dao.add_many(
            [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}],
            return_objects=False,
        )
        assert result is None

        # Записи должны быть в БД
        all_users = await dao.find_all()
        assert len(all_users) == 3

    async def test_add_many_with_return(self, db_session: AsyncSession):
        """add_many с return_objects=True → список объектов."""
        dao = UserDAO(db_session)
        result = await dao.add_many(
            [{"name": "Alice"}, {"name": "Bob"}],
            return_objects=True,
        )
        assert result is not None
        assert len(result) == 2
        names = {u.name for u in result}
        assert "Alice" in names
        assert "Bob" in names

    async def test_add_many_with_pydantic_models(self, db_session: AsyncSession):
        """add_many принимает Pydantic-модели."""
        dao = UserDAO(db_session)
        result = await dao.add_many(
            [UserCreate(name="Alice"), UserCreate(name="Bob")],
            return_objects=False,
        )
        assert result is None

        all_users = await dao.find_all()
        assert len(all_users) == 2


# ── BaseDAO.find_all (расширенные тесты) ─────────────────


class TestBaseDAOFindAll:
    async def test_find_all_with_limit(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        for i in range(5):
            await dao.add(UserCreate(name=f"User{i}"), flush=True)

        result = await dao.find_all(limit=3)
        assert len(result) == 3

    async def test_find_all_with_offset(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        for i in range(5):
            await dao.add(UserCreate(name=f"User{i}"), flush=True)

        result = await dao.find_all(offset=3)
        assert len(result) == 2

    async def test_find_all_with_pydantic_filter(self, db_session: AsyncSession):
        """find_all с Pydantic-моделью в качестве фильтра."""
        dao = UserDAO(db_session)
        await dao.add(UserCreate(name="Alice"), flush=True)
        await dao.add(UserCreate(name="Bob"), flush=True)

        result = await dao.find_all(filters=UserCreate(name="Alice"))
        assert len(result) == 1
        assert result[0].name == "Alice"

    async def test_find_all_with_dict_filter(self, db_session: AsyncSession):
        """find_all с dict-фильтром."""
        dao = UserDAO(db_session)
        await dao.add(UserCreate(name="Alice"), flush=True)

        result = await dao.find_all(filters={"name": "Alice"})
        assert len(result) == 1

    async def test_find_all_empty_table(self, db_session: AsyncSession):
        dao = UserDAO(db_session)
        result = await dao.find_all()
        assert len(result) == 0


# ── BaseDAO.update (расширенные тесты) ───────────────────


class TestBaseDAOUpdate:
    async def test_update_with_pydantic_filter(self, db_session: AsyncSession):
        """update с Pydantic-моделью в фильтре."""
        dao = UserDAO(db_session)
        await dao.add(UserCreate(name="Alice"), flush=True)

        count = await dao.update(
            filters=UserCreate(name="Alice"),
            values={"name": "Alice Updated"},
        )
        assert count == 1

    async def test_update_with_pydantic_values(self, db_session: AsyncSession):
        """update с Pydantic-моделью в values."""
        dao = UserDAO(db_session)
        user = await dao.add(UserCreate(name="Alice"), flush=True)

        count = await dao.update(
            filters={"id": user.id},
            values=UserCreate(name="New Name"),
        )
        assert count == 1


# ── Base.__repr__ ─────────────────────────────────────────


class TestBaseRepr:
    async def test_repr_contains_class_name(self, db_session: AsyncSession):
        """__repr__ содержит имя класса."""
        dao = UserDAO(db_session)
        user = await dao.add(UserCreate(name="Alice"), flush=True)
        r = repr(user)
        assert "User" in r

    async def test_repr_contains_first_columns(self, db_session: AsyncSession):
        """__repr__ содержит первые repr_cols_num столбцов."""
        dao = UserDAO(db_session)
        user = await dao.add(UserCreate(name="Alice"), flush=True)
        r = repr(user)
        assert "name=" in r
        assert "Alice" in r

    async def test_repr_does_not_crash(self, db_session: AsyncSession):
        """__repr__ не падает ни на одной модели."""
        from src.modules.documents.dao import DocumentDAO
        from src.modules.documents.models import Document

        user_dao = UserDAO(db_session)
        user = await user_dao.add(UserCreate(name="Test"), flush=True)

        doc_dao = DocumentDAO(db_session)
        doc = await doc_dao.add(
            {
                "user_id": user.id,
                "filename": "test.txt",
                "content_type": "text/plain",
                "file_path": "/tmp/test.txt",
                "text_content": "",
                "status": "ready",
            },
            flush=True,
        )
        r = repr(doc)
        assert "Document" in r
