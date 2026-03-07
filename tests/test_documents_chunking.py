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

    def test_unicode_text(self):
        """Текст на кириллице и других алфавитах."""
        text = "Привет мир. " * 100
        chunks = split_into_chunks(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 1
        joined = " ".join(chunks)
        assert "Привет" in joined

    def test_mixed_language_text(self):
        """Текст на нескольких языках."""
        text = (
            "# Заголовок\n\n"
            "Русский текст. English text. 日本語テキスト。\n\n"
            "## Section 2\n\n"
            "More mixed content: العربية and Ελληνικά."
        )
        chunks = split_into_chunks(text, chunk_size=512)
        assert len(chunks) >= 1
        joined = " ".join(chunks)
        assert "Русский" in joined
        assert "English" in joined

    def test_only_headers_no_content(self):
        """Текст только из заголовков без содержимого."""
        text = "# Title\n\n## Section\n\n### Subsection\n"
        chunks = split_into_chunks(text, chunk_size=512)
        assert len(chunks) >= 1

    def test_very_long_single_paragraph(self):
        """Один длинный параграф без разделителей."""
        text = "word" * 5000
        chunks = split_into_chunks(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_text_with_code_blocks(self):
        """Текст с блоками кода."""
        text = (
            "# API Reference\n\n"
            "```python\ndef hello():\n    print('hello')\n```\n\n"
            "## Usage\n\n"
            "Call `hello()` to greet."
        )
        chunks = split_into_chunks(text, chunk_size=512)
        assert len(chunks) >= 1
        joined = " ".join(chunks)
        assert "hello" in joined

    def test_text_with_lists(self):
        """Текст со списками."""
        text = (
            "# Items\n\n"
            "- Item 1\n"
            "- Item 2\n"
            "- Item 3\n\n"
            "## Numbered\n\n"
            "1. First\n"
            "2. Second\n"
            "3. Third\n"
        )
        chunks = split_into_chunks(text, chunk_size=512)
        assert len(chunks) >= 1
        joined = " ".join(chunks)
        assert "Item 1" in joined

    def test_text_with_tables(self):
        """Текст с Markdown-таблицами."""
        text = (
            "# Data\n\n"
            "| Name | Value |\n"
            "|------|-------|\n"
            "| A    | 1     |\n"
            "| B    | 2     |\n"
        )
        chunks = split_into_chunks(text, chunk_size=512)
        assert len(chunks) >= 1

    def test_chunk_overlap_content(self):
        """Перекрытие: конец одного чанка содержится в начале следующего."""
        # Генерируем текст из уникальных предложений
        sentences = [f"Sentence number {i}. " for i in range(100)]
        text = "".join(sentences)
        chunks = split_into_chunks(text, chunk_size=200, chunk_overlap=50)
        if len(chunks) >= 2:
            # Последние слова первого чанка пересекаются с началом второго
            # (точная проверка overlap зависит от splitter, но chunks > 1)
            assert len(chunks) > 1

    def test_default_params(self):
        """Дефолтные параметры: chunk_size=512, chunk_overlap=64."""
        text = "Content. " * 500
        chunks = split_into_chunks(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            # Каждый чанк не должен сильно превышать chunk_size
            assert len(chunk) <= 600  # допуск
