import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.chat.dao import ChatDAO
from src.modules.users.dao import UserDAO
from src.modules.users.schemas import UserCreate


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def user(db_session: AsyncSession):
    dao = UserDAO(db_session)
    return await dao.add(UserCreate(name="ChatUser"), flush=True)


class TestChatDAO:
    async def test_create_session(self, db_session: AsyncSession, user):
        dao = ChatDAO(db_session)
        session = await dao.create_session(user_id=user.id)
        assert session.id is not None
        assert session.user_id == user.id

    async def test_add_message(self, db_session: AsyncSession, user):
        dao = ChatDAO(db_session)
        session = await dao.create_session(user_id=user.id)

        message = await dao.add_message(
            session_id=session.id,
            role="user",
            content="Hello",
        )

        assert message.id is not None
        assert message.role == "user"

    async def test_get_sessions_by_user_with_count(self, db_session: AsyncSession, user):
        dao = ChatDAO(db_session)
        session = await dao.create_session(user_id=user.id)
        await dao.add_message(session.id, "user", "Hi")
        await dao.add_message(session.id, "assistant", "Hello")

        rows = await dao.get_sessions_by_user(user.id)
        assert len(rows) == 1
        loaded_session, count = rows[0]
        assert loaded_session.id == session.id
        assert count == 2

    async def test_get_messages_by_session(self, db_session: AsyncSession, user):
        dao = ChatDAO(db_session)
        session = await dao.create_session(user_id=user.id)
        await dao.add_message(session.id, "user", "A")
        await dao.add_message(session.id, "assistant", "B")

        messages = await dao.get_messages_by_session(session.id)
        assert [m.role for m in messages] == ["user", "assistant"]

    async def test_find_session_for_user(self, db_session: AsyncSession, user):
        dao = ChatDAO(db_session)
        session = await dao.create_session(user_id=user.id)

        found = await dao.find_session_for_user(session.id, user.id)
        not_found = await dao.find_session_for_user(session.id, user.id + 999)

        assert found is not None
        assert not_found is None
