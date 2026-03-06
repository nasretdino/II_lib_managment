from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def split_into_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """
    Разбиение текста на чанки с учётом структуры документа.

    1. MarkdownHeaderTextSplitter — разбивает по Markdown-заголовкам,
       сохраняя иерархию (H1 → H2 → H3).
    2. RecursiveCharacterTextSplitter — дробит крупные секции
       до нужного размера с перекрытием.
    """
    if not text or not text.strip():
        return []

    # Шаг 1: разбиение по Markdown-заголовкам
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    md_docs = md_splitter.split_text(text)

    # Шаг 2: дробление крупных секций до chunk_size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    final_docs = text_splitter.split_documents(md_docs)

    return [doc.page_content for doc in final_docs if doc.page_content.strip()]
