import asyncio
from pathlib import Path

from loguru import logger

from src.core.exceptions import DocumentParsingError


async def extract_text(file_path: str, content_type: str) -> str:
    """
    Определяет тип файла и извлекает текст:
    - PDF  → MarkItDown / PyMuPDF (в asyncio.to_thread, т.к. sync)
    - TXT  → простое чтение
    - DOCX → MarkItDown (в asyncio.to_thread)
    """
    path = Path(file_path)

    if content_type == "text/plain":
        return await _read_text_file(path)

    if content_type == "application/pdf":
        return await _extract_with_markitdown(path)

    if content_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        return await _extract_with_markitdown(path)

    raise DocumentParsingError(f"Unsupported content type: {content_type}")


async def _read_text_file(path: Path) -> str:
    """Чтение текстового файла."""
    try:
        return await asyncio.to_thread(path.read_text, encoding="utf-8")
    except FileNotFoundError as exc:
        raise DocumentParsingError(f"File not found: {path.name}") from exc
    except UnicodeDecodeError as exc:
        raise DocumentParsingError(
            f"Failed to decode {path.name} as UTF-8: {exc}"
        ) from exc


async def _extract_with_markitdown(path: Path) -> str:
    """Извлечение текста через MarkItDown (sync → asyncio.to_thread)."""
    from markitdown import MarkItDown
    from markitdown._exceptions import FileConversionException

    def _convert() -> str:
        md = MarkItDown()
        result = md.convert(str(path))
        return result.text_content

    logger.info("Extracting text from {} via MarkItDown", path.name)
    try:
        text = await asyncio.to_thread(_convert)
    except FileConversionException as exc:
        raise DocumentParsingError(f"Failed to convert {path.name}: {exc}") from exc
    except Exception as exc:
        raise DocumentParsingError(
            f"Unexpected error parsing {path.name}: {exc}"
        ) from exc

    if not text or not text.strip():
        logger.warning("MarkItDown extracted empty text from {}", path.name)

    return text
