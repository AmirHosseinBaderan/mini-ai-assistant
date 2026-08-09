from pathlib import Path

import pytest

from application.rag.loader import DocumentLoader


def test_load_text_file(tmp_path: Path):
    file = tmp_path / "test.txt"

    file.write_text(
        "Hello RAG",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    document = loader.load(file)

    assert document.content == "Hello RAG"

    assert document.metadata["filename"] == "test.txt"


def test_load_markdown_file(tmp_path: Path):
    file = tmp_path / "test.md"

    file.write_text(
        "# Hello RAG",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    document = loader.load(file)

    assert document.content == "# Hello RAG"


def test_file_not_found():
    loader = DocumentLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("does-not-exist.txt")


def test_unsupported_file_type(tmp_path: Path):
    file = tmp_path / "test.pdf"

    file.write_bytes(b"fake pdf")

    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load(file)


def test_path_must_be_file(tmp_path: Path):
    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load(tmp_path)