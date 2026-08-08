from pathlib import Path

import pytest

from application.rag.knowledge_base import (
    KnowledgeBase,
)


class FakeLoader:

    def load(self, path: Path):
        return f"document:{path.name}"


class FakeChunker:

    def split(self, document):
        return [
            f"{document}:chunk1",
            f"{document}:chunk2",
        ]


class FakeIndexer:

    def __init__(self):
        self.chunks = []

    def index(self, chunks):
        self.chunks.extend(chunks)


def create_knowledge_base():
    indexer = FakeIndexer()

    knowledge_base = KnowledgeBase(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        indexer=indexer,
    )

    return knowledge_base, indexer


def test_add_file(tmp_path):

    file = tmp_path / "test.txt"
    file.write_text(
        "Hello",
        encoding="utf-8",
    )

    knowledge_base, indexer = (
        create_knowledge_base()
    )

    count = knowledge_base.add_file(file)

    assert count == 2

    assert indexer.chunks == [
        "document:test.txt:chunk1",
        "document:test.txt:chunk2",
    ]


def test_add_directory(tmp_path):

    (tmp_path / "one.txt").write_text(
        "One",
        encoding="utf-8",
    )

    (tmp_path / "two.txt").write_text(
        "Two",
        encoding="utf-8",
    )

    knowledge_base, indexer = (
        create_knowledge_base()
    )

    count = knowledge_base.add_directory(
        tmp_path
    )

    assert count == 4

    assert len(indexer.chunks) == 4


def test_directory_not_found(tmp_path):

    knowledge_base, _ = (
        create_knowledge_base()
    )

    with pytest.raises(ValueError):
        knowledge_base.add_directory(
            tmp_path / "missing"
        )


def test_path_must_be_directory(tmp_path):

    file = tmp_path / "test.txt"
    file.write_text(
        "Hello",
        encoding="utf-8",
    )

    knowledge_base, _ = (
        create_knowledge_base()
    )

    with pytest.raises(ValueError):
        knowledge_base.add_directory(file)