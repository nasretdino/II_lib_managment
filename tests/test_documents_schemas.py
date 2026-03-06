import pytest
from pydantic import ValidationError

from src.modules.documents.schemas import DocumentRead, DocumentUpload


class TestDocumentUpload:
    def test_valid(self):
        d = DocumentUpload(user_id=1)
        assert d.user_id == 1

    def test_missing_user_id(self):
        with pytest.raises(ValidationError):
            DocumentUpload()

    def test_negative_user_id(self):
        """Отрицательный user_id допускается на уровне схемы (валидация в роутере)."""
        d = DocumentUpload(user_id=-1)
        assert d.user_id == -1

    def test_string_user_id_coerced(self):
        """Строковое число приводится к int."""
        d = DocumentUpload(user_id="42")
        assert d.user_id == 42

    def test_non_numeric_user_id_rejected(self):
        with pytest.raises(ValidationError):
            DocumentUpload(user_id="abc")


class TestDocumentRead:
    def test_from_dict(self):
        d = DocumentRead(
            id=1,
            filename="test.pdf",
            content_type="application/pdf",
            status="ready",
            created_at="2026-01-01T00:00:00Z",
        )
        assert d.filename == "test.pdf"
        assert d.status == "ready"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            DocumentRead(id=1, filename="test.pdf")

    def test_all_statuses(self):
        """DocumentRead принимает любые значения status (нет enum-ограничения)."""
        for status in ("pending", "processing", "ready", "error"):
            d = DocumentRead(
                id=1,
                filename="f.txt",
                content_type="text/plain",
                status=status,
                created_at="2026-01-01T00:00:00Z",
            )
            assert d.status == status

    def test_from_attributes(self):
        """from_attributes=True позволяет создать из ORM-объекта."""

        class FakeDoc:
            id = 1
            filename = "f.txt"
            content_type = "text/plain"
            status = "ready"
            created_at = "2026-01-01T00:00:00Z"

        d = DocumentRead.model_validate(FakeDoc())
        assert d.id == 1
