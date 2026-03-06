import pytest
from pydantic import ValidationError

from src.modules.users.schemas import UserCreate, UserFilter, UserRead, UserUpdate


class TestUserCreate:
    def test_valid(self):
        u = UserCreate(name="Alice")
        assert u.name == "Alice"

    def test_strips_whitespace(self):
        u = UserCreate(name="  Bob  ")
        assert u.name == "Bob"

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            UserCreate(name="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError):
            UserCreate(name="   ")

    def test_too_long_name_rejected(self):
        with pytest.raises(ValidationError):
            UserCreate(name="A" * 256)

    def test_max_length_name_accepted(self):
        u = UserCreate(name="A" * 255)
        assert len(u.name) == 255

    def test_min_length_name_accepted(self):
        u = UserCreate(name="A")
        assert u.name == "A"

    def test_missing_name_rejected(self):
        with pytest.raises(ValidationError):
            UserCreate()


class TestUserUpdate:
    def test_partial_update(self):
        u = UserUpdate(name="New Name")
        assert u.name == "New Name"

    def test_no_fields(self):
        u = UserUpdate()
        assert u.name is None

    def test_strips_whitespace(self):
        u = UserUpdate(name="  trimmed  ")
        assert u.name == "trimmed"

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            UserUpdate(name="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError):
            UserUpdate(name="   ")

    def test_too_long_name_rejected(self):
        with pytest.raises(ValidationError):
            UserUpdate(name="A" * 256)

    def test_none_is_valid(self):
        """Явный None — допустим (не обновлять поле)."""
        u = UserUpdate(name=None)
        assert u.name is None


class TestUserFilter:
    def test_defaults_none(self):
        f = UserFilter()
        assert f.name is None

    def test_with_name(self):
        f = UserFilter(name="alice")
        assert f.name == "alice"

    def test_empty_string(self):
        f = UserFilter(name="")
        assert f.name == ""


class TestUserRead:
    def test_valid(self):
        u = UserRead(
            id=1,
            name="Alice",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        assert u.id == 1
        assert u.name == "Alice"

    def test_missing_fields_rejected(self):
        with pytest.raises(ValidationError):
            UserRead(id=1, name="Alice")

    def test_from_attributes(self):
        class FakeUser:
            id = 1
            name = "Bob"
            created_at = "2026-01-01T00:00:00Z"
            updated_at = "2026-01-01T00:00:00Z"

        u = UserRead.model_validate(FakeUser())
        assert u.name == "Bob"
