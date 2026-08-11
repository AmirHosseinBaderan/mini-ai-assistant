from unittest.mock import Mock

import pytest

from application.rag.chunk import Chunk
from application.tools.knowledge_search import KnowledgeSearchTool


def create_tool():

    retriever = Mock()

    return KnowledgeSearchTool(
        retriever=retriever,
    ), retriever


def test_tool_name():

    tool, _ = create_tool()

    assert tool.name == "knowledge_search"


def test_tool_description():

    tool, _ = create_tool()

    assert "knowledge base" in tool.description


def test_tool_parameters():

    tool, _ = create_tool()

    parameters = tool.parameters

    assert parameters["type"] == "object"

    assert "query" in parameters["properties"]

    assert "query" in parameters["required"]


def test_execute_returns_search_results():

    tool, retriever = create_tool()

    chunk = Chunk(
        content="Python is a programming language.",
        metadata={
            "source": "python.txt",
        },
    )

    retriever.retrieve.return_value = [
        (chunk, 0.91),
    ]

    result = tool.execute(
        query="What is Python?",
    )

    assert result.content == [
        {
            "content": "Python is a programming language.",
            "score": 0.91,
            "metadata": {
                "source": "python.txt",
            },
        }
    ]

    retriever.retrieve.assert_called_once_with(
        "What is Python?",
        top_k=3,
    )


def test_execute_supports_custom_top_k():

    tool, retriever = create_tool()

    retriever.retrieve.return_value = []

    tool.execute(
        query="Python",
        top_k=10,
    )

    retriever.retrieve.assert_called_once_with(
        "Python",
        top_k=10,
    )


def test_execute_requires_query():

    tool, _ = create_tool()

    with pytest.raises(ValueError):
        tool.execute()


def test_execute_rejects_empty_query():

    tool, _ = create_tool()

    with pytest.raises(ValueError):
        tool.execute(
            query="   ",
        )


def test_execute_rejects_invalid_query():

    tool, _ = create_tool()

    with pytest.raises(ValueError):
        tool.execute(
            query=123,
        )