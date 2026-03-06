import pytest

from src.modules.documents.chunking import split_into_chunks


class TestSplitIntoChunks:
    def test_empty_text(self):
        assert split_into_chunks("") == []

    def test_whitespace_only(self):
        assert split_into_chunks("   \n\n  ") == []

    def test_none_like_empty(self):
        """Текст из одних пробелов/табов."""
        assert split_into_chunks("\t\t  \n") == []

    def test_short_text_single_chunk(self):
        text = "Hello, this is a short document."
        chunks = split_into_chunks(text, chunk_size=512)
        assert len(chunks) >= 1
        assert "Hello" in chunks[0]

    def test_markdown_header_splitting(self):
        text = (
            "# Chapter 1\n\n"
            "Content of chapter one.\n\n"
            "# Chapter 2\n\n"
            "Content of chapter two."
        )
        chunks = split_into_chunks(text, chunk_size=512)
        assert len(chunks) >= 2

    def test_large_text_creates_multiple_chunks(self):
        text = "Word " * 500  # ~2500 chars
        chunks = split_into_chunks(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 250  # допуск на overlap

    def test_overlap_between_chunks(self):
        text = "Sentence one. " * 100
        chunks = split_into_chunks(text, chunk_size=100, chunk_overlap=20)
        if len(chunks) > 1:
            full_text = "".join(chunks)
            assert len(full_text) >= len(text.strip())

    def test_nested_headers(self):
        text = (
            "# Main Title\n\n"
            "Intro text.\n\n"
            "## Section A\n\n"
            "Section A content.\n\n"
            "### Subsection A1\n\n"
            "Subsection A1 content.\n\n"
            "## Section B\n\n"
            "Section B content."
        )
        chunks = split_into_chunks(text, chunk_size=512)
        assert len(chunks) >= 1
        joined = " ".join(chunks)
        assert "Main Title" in joined
        assert "Section A" in joined
        assert "Section B" in joined

    def test_custom_chunk_size(self):
        text = "A " * 1000
        small_chunks = split_into_chunks(text, chunk_size=100, chunk_overlap=10)
        large_chunks = split_into_chunks(text, chunk_size=500, chunk_overlap=50)
        assert len(small_chunks) > len(large_chunks)

    def test_no_empty_chunks_in_result(self):
        """Результат не содержит пустых строк."""
        text = "# Title\n\n\n\nContent\n\n\n\n# Title 2\n\nMore content"
        chunks = split_into_chunks(text, chunk_size=512)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_single_word(self):
        chunks = split_into_chunks("hello")
        assert len(chunks) == 1
        assert chunks[0] == "hello"

    def test_plain_text_without_headers(self):
        """Текст без Markdown-заголовков тоже корректно разбивается."""
        text = "This is a plain paragraph. " * 50
        chunks = split_into_chunks(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 1
        joined = " ".join(chunks)
        assert "plain paragraph" in joined
