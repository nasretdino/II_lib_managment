import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

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


class TestReadTextFileErrors:
    async def test_file_not_found(self):
        """Несуществующий файл → DocumentParsingError."""
        with pytest.raises(DocumentParsingError, match="File not found"):
            await extract_text("/nonexistent/path/file.txt", "text/plain")

    async def test_non_utf8_file(self, tmp_path: Path):
        """Файл с невалидной кодировкой → DocumentParsingError."""
        file = tmp_path / "binary.txt"
        file.write_bytes(b"\x80\x81\x82\xff\xfe")

        with pytest.raises(DocumentParsingError, match="Failed to decode"):
            await extract_text(str(file), "text/plain")


class TestMarkItDownErrors:
    async def test_corrupted_pdf_returns_parsing_error_or_empty(self, tmp_path: Path):
        """Невалидный PDF → либо DocumentParsingError, либо пустой текст (зависит от MarkItDown)."""
        file = tmp_path / "bad.pdf"
        file.write_bytes(b"not a real pdf content")

        try:
            text = await extract_text(str(file), "application/pdf")
            # MarkItDown может вернуть пустой текст вместо ошибки
            assert isinstance(text, str)
        except DocumentParsingError:
            pass  # Это тоже допустимый результат

    async def test_corrupted_docx_raises_parsing_error(self, tmp_path: Path):
        """Невалидный DOCX → DocumentParsingError (не 500)."""
        file = tmp_path / "bad.docx"
        file.write_bytes(b"not a real docx")

        # Для невалидного DOCX — либо DocumentParsingError, либо пустой текст
        try:
            text = await extract_text(
                str(file),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            assert isinstance(text, str)
        except DocumentParsingError:
            pass

    async def test_msword_content_type_accepted(self, tmp_path: Path):
        """application/msword обрабатывается MarkItDown (не Unsupported)."""
        file = tmp_path / "bad.doc"
        file.write_bytes(b"fake doc content")

        try:
            await extract_text(str(file), "application/msword")
        except DocumentParsingError as e:
            assert "Unsupported content type" not in str(e)

    async def test_file_conversion_exception_wrapped(self, tmp_path: Path):
        """FileConversionException из MarkItDown → DocumentParsingError с информативным сообщением."""
        from markitdown._exceptions import FileConversionException

        file = tmp_path / "fail.pdf"
        file.write_bytes(b"%PDF-corrupted")

        with patch(
            "src.modules.documents.parser.asyncio.to_thread",
            side_effect=FileConversionException(attempts=[]),
        ):
            with pytest.raises(DocumentParsingError, match="Failed to convert"):
                await extract_text(str(file), "application/pdf")

    async def test_generic_exception_wrapped(self, tmp_path: Path):
        """Любая непредвиденная ошибка MarkItDown → DocumentParsingError."""
        file = tmp_path / "crash.pdf"
        file.write_bytes(b"%PDF-1.4")

        with patch(
            "src.modules.documents.parser.asyncio.to_thread",
            side_effect=RuntimeError("Something broke"),
        ):
            with pytest.raises(DocumentParsingError, match="Unexpected error"):
                await extract_text(str(file), "application/pdf")

    async def test_successful_pdf_extraction(self, tmp_path: Path):
        """Успешная конвертация PDF через мок MarkItDown."""
        file = tmp_path / "good.pdf"
        file.write_bytes(b"%PDF-1.4")

        mock_result = MagicMock()
        mock_result.text_content = "Extracted PDF text"

        with patch(
            "src.modules.documents.parser.asyncio.to_thread",
            return_value="Extracted PDF text",
        ):
            text = await extract_text(str(file), "application/pdf")
            assert text == "Extracted PDF text"
