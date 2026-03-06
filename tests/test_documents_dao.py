import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.documents.dao import DocumentDAO
from src.modules.users.dao import UserDAO
from src.modules.users.schemas import UserCreate


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def user(db_session: AsyncSession):
    """Создаёт пользователя для тестов документов."""
    dao = UserDAO(db_session)
    return await dao.add(UserCreate(name="TestUser"), flush=True)


def _doc_dict(user_id: int, **overrides) -> dict:
    """Фабрика для создания словаря документа."""
    data = {
        "user_id": user_id,
        "filename": "test.txt",
        "content_type": "text/plain",
        "file_path": "/tmp/test.txt",
        "text_content": "",
        "status": "ready",
    }
    data.update(overrides)
    return data


class TestDocumentDAOAdd:
    async def test_add_and_find_by_id(self, db_session: AsyncSession, user):
        dao = DocumentDAO(db_session)
        doc = await dao.add(_doc_dict(user.id, text_content="hello"), flush=True)
        assert doc.id is not None
        assert doc.filename == "test.txt"

        found = await dao.find_one_or_none_by_id(doc.id)
        assert found is not None
        assert found.filename == "test.txt"

    async def test_add_sets_defaults(self, db_session: AsyncSession, user):
        dao = DocumentDAO(db_session)
        doc = await dao.add(_doc_dict(user.id), flush=True)
        assert doc.status == "ready"
        assert doc.text_content == ""

    async def test_add_multiple_documents(self, db_session: AsyncSession, user):
        dao = DocumentDAO(db_session)
        for i in range(3):
            await dao.add(_doc_dict(user.id, filename=f"doc_{i}.txt"), flush=True)
        docs = await dao.find_by_user(user.id)
        assert len(docs) == 3


class TestDocumentDAOFindByUser:
    async def test_find_by_user(self, db_session: AsyncSession, user):
        dao = DocumentDAO(db_session)
        await dao.add(_doc_dict(user.id, filename="a.txt"), flush=True)
        await dao.add(_doc_dict(user.id, filename="b.txt"), flush=True)

        docs = await dao.find_by_user(user.id)
        assert len(docs) == 2

    async def test_find_by_user_empty(self, db_session: AsyncSession):
        dao = DocumentDAO(db_session)
        docs = await dao.find_by_user(9999)
        assert len(docs) == 0

    async def test_find_by_user_isolation(self, db_session: AsyncSession):
        """Документы разных пользователей не пересекаются."""
        user_dao = UserDAO(db_session)
        u1 = await user_dao.add(UserCreate(name="U1"), flush=True)
        u2 = await user_dao.add(UserCreate(name="U2"), flush=True)

        doc_dao = DocumentDAO(db_session)
        await doc_dao.add(_doc_dict(u1.id, filename="u1_doc.txt"), flush=True)
        await doc_dao.add(_doc_dict(u2.id, filename="u2_doc.txt"), flush=True)

        assert len(await doc_dao.find_by_user(u1.id)) == 1
        assert len(await doc_dao.find_by_user(u2.id)) == 1


class TestDocumentDAOUpdateStatus:
    async def test_update_status(self, db_session: AsyncSession, user):
        dao = DocumentDAO(db_session)
        doc = await dao.add(_doc_dict(user.id, status="pending"), flush=True)

        count = await dao.update_status(doc.id, "ready")
        assert count == 1

    async def test_update_status_nonexistent(self, db_session: AsyncSession):
        dao = DocumentDAO(db_session)
        count = await dao.update_status(9999, "ready")
        assert count == 0

    async def test_update_status_transitions(self, db_session: AsyncSession, user):
        """Проверка всех переходов статусов."""
        dao = DocumentDAO(db_session)
        doc = await dao.add(_doc_dict(user.id, status="pending"), flush=True)

        for status in ("processing", "ready", "error"):
            count = await dao.update_status(doc.id, status)
            assert count == 1


class TestDocumentDAODelete:
    async def test_delete(self, db_session: AsyncSession, user):
        dao = DocumentDAO(db_session)
        doc = await dao.add(_doc_dict(user.id), flush=True)
        doc_id = doc.id

        deleted = await dao.delete(filters={"id": doc_id})
        assert deleted == 1

        db_session.expunge_all()
        found = await dao.find_one_or_none_by_id(doc_id)
        assert found is None

    async def test_delete_nonexistent(self, db_session: AsyncSession):
        dao = DocumentDAO(db_session)
        deleted = await dao.delete(filters={"id": 9999})
        assert deleted == 0

    async def test_find_one_or_none_by_id_nonexistent(self, db_session: AsyncSession):
        dao = DocumentDAO(db_session)
        assert await dao.find_one_or_none_by_id(9999) is None
