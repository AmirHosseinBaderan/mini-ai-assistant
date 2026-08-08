import pytest

from application.rag.chunker import Chunker
from application.rag.document import Document


def test_split_document():
    document = Document(
        content="ABCDEFGHIJKLMNO",
    )

    chunker = Chunker(
        chunk_size=10,
        chunk_overlap=3,
    )

    chunks = chunker.split(document)

    assert [chunk.content for chunk in chunks] == [
        "ABCDEFGHIJ",
        "HIJKLMNO",
    ]


def test_chunk_metadata():
    document = Document(
        content="ABCDEFGHIJ",
        metadata={
            "source": "test.txt",
        },
    )

    chunker = Chunker(
        chunk_size=5,
        chunk_overlap=1,
    )

    chunks = chunker.split(document)

    assert chunks[0].metadata == {
        "source": "test.txt",
        "chunk_index": 0,
    }

    assert chunks[1].metadata == {
        "source": "test.txt",
        "chunk_index": 1,
    }


def test_empty_document():
    document = Document(
        content="",
    )

    chunker = Chunker()

    assert chunker.split(document) == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunk_size=0)


def test_negative_overlap():
    with pytest.raises(ValueError):
        Chunker(
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Chunker(
            chunk_size=100,
            chunk_overlap=100,
        )
        
def test_chunk_metadata_does_not_modify_document_metadata():
    document = Document(
        content="ABCDEFGHIJ",
        metadata={
            "source": "test.txt",
        },
    )

    chunker = Chunker(
        chunk_size=5,
        chunk_overlap=1,
    )

    chunker.split(document)

    assert document.metadata == {
        "source": "test.txt",
    }
    
def test_document_length_matches_chunk_size():
    document = Document(
        content="ABCDEFGHIJ",
    )

    chunker = Chunker(
        chunk_size=10,
        chunk_overlap=3,
    )

    chunks = chunker.split(document)

    assert [chunk.content for chunk in chunks] == [
        "ABCDEFGHIJ",
    ]
    
def test_document_shorter_than_chunk_size():
    document = Document(
        content="ABCDE",
    )

    chunker = Chunker(
        chunk_size=10,
        chunk_overlap=3,
    )

    chunks = chunker.split(document)

    assert [chunk.content for chunk in chunks] == [
        "ABCDE",
    ]