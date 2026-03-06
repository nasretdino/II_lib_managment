import pytest
from pathlib import Path

from src.modules.documents.parser import extract_text
from src.core.exceptions import DocumentParsingError


pytestmark = pytest.mark.asyncio


class TestExtractText:
    async def test_read_txt_file(self, tmp_path: Path):
        file = tmp_path / "test.txt"
        file.write_text("Hello, world!", encoding="utf-8")

        text = await extract_text(str(file), "text/plain")
        assert text == "Hello, world!"

    async def test_read_txt_file_utf8(self, tmp_path: Path):
        file = tmp_path / "test_ru.txt"
        file.write_text("Привет, мир!", encoding="utf-8")

        text = await extract_text(str(file), "text/plain")
        assert text == "Привет, мир!"

    async def test_unsupported_content_type(self, tmp_path: Path):
        file = tmp_path / "test.xyz"
        file.write_bytes(b"data")

        with pytest.raises(DocumentParsingError, match="Unsupported content type"):
            await extract_text(str(file), "application/octet-stream")

    async def test_empty_txt_file(self, tmp_path: Path):
        file = tmp_path / "empty.txt"
        file.write_text("", encoding="utf-8")

        text = await extract_text(str(file), "text/plain")
        assert text == ""

    async def test_multiline_txt_file(self, tmp_path: Path):
        content = "Line 1\nLine 2\nLine 3"
        file = tmp_path / "multi.txt"
        file.write_text(content, encoding="utf-8")

        text = await extract_text(str(file), "text/plain")
        assert text == content

    async def test_unsupported_image_type(self, tmp_path: Path):
        file = tmp_path / "img.png"
        file.write_bytes(b"\x89PNG")

        with pytest.raises(DocumentParsingError):
            await extract_text(str(file), "image/png")

    async def test_unsupported_audio_type(self, tmp_path: Path):
        file = tmp_path / "audio.mp3"
        file.write_bytes(b"\xff\xfb")

        with pytest.raises(DocumentParsingError):
            await extract_text(str(file), "audio/mpeg")

    async def test_docx_content_type_accepted(self, tmp_path: Path):
        """application/vnd.openxmlformats-... вызывает MarkItDown (не Unsupported)."""
        file = tmp_path / "test.docx"
        # Минимальный zip-файл (невалидный docx — MarkItDown упадёт, но тип принят)
        file.write_bytes(b"PK\x03\x04")

        # MarkItDown может упасть на невалидном файле, но это не DocumentParsingError("Unsupported")
        # Мы проверяем, что не получаем "Unsupported content type"
        try:
            await extract_text(
                str(file),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except DocumentParsingError as e:
            assert "Unsupported content type" not in str(e)
        except Exception:
            pass  # MarkItDown ошибка на невалидном файле — ОК
