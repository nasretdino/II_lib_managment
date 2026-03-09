import pytest
from pydantic import ValidationError

from src.modules.chat.schemas import ChatRequest, MessageRead, SessionRead


class TestChatRequest:
    def test_valid(self):
        payload = ChatRequest(user_id=1, message="Hello")
        assert payload.user_id == 1
        assert payload.session_id is None

    def test_invalid_user_id(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id=0, message="Hello")

    def test_invalid_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id=1, message="")

    def test_invalid_session_id(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id=1, session_id=0, message="Hello")


class TestSessionRead:
    def test_from_dict(self):
        item = SessionRead(
            id=1,
            title="Session",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            message_count=2,
        )
        assert item.id == 1
        assert item.message_count == 2


class TestMessageRead:
    def test_from_dict(self):
        item = MessageRead(
            id=1,
            role="user",
            content="Hello",
            created_at="2026-01-01T00:00:00Z",
        )
        assert item.role == "user"
