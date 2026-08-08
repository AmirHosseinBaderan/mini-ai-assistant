from application.rag.chunk import Chunk
from application.rag.context_builder import ContextBuilder


def test_build_context():

    chunks = [
        Chunk(
            content="Python is a programming language.",
            metadata={
                "filename": "python.txt",
            },
        ),
        Chunk(
            content="Python is easy to learn.",
            metadata={
                "filename": "python.txt",
            },
        ),
    ]

    builder = ContextBuilder()

    context = builder.build(chunks)

    assert context == (
        "[Source: python.txt]\n"
        "Python is a programming language.\n\n"
        "[Source: python.txt]\n"
        "Python is easy to learn."
    )


def test_build_empty_context():

    builder = ContextBuilder()

    context = builder.build([])

    assert context == ""


def test_unknown_source():

    chunks = [
        Chunk(
            content="Some information.",
        ),
    ]

    builder = ContextBuilder()

    context = builder.build(chunks)

    assert (
        "[Source: unknown]\n"
        "Some information."
        in context
    )