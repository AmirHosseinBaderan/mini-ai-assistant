from application.rag.document import Document


def test_create_document():
    document = Document(
        content="Hello RAG",
    )

    assert document.content == "Hello RAG"
    assert document.metadata == {}


def test_create_document_with_metadata():
    document = Document(
        content="Hello RAG",
        metadata={
            "source": "test.txt",
            "page": 1,
        },
    )

    assert document.content == "Hello RAG"
    assert document.metadata == {
        "source": "test.txt",
        "page": 1,
    }


def test_documents_have_independent_metadata():
    first = Document(
        content="First",
    )

    second = Document(
        content="Second",
    )

    first.metadata["source"] = "first.txt"

    assert first.metadata == {
        "source": "first.txt",
    }

    assert second.metadata == {}