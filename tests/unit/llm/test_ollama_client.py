from unittest.mock import MagicMock

from application.llm.message import LLMMessage
from application.llm.ollama_client import OllamaClient
from application.llm.tool import LLMTool


def create_client():
    client = OllamaClient(
        host="http://localhost:11434",
        model="test-model",
        embedding_model="test-embedding",
    )

    client.client = MagicMock()

    return client


def test_chat_without_tool_calls():

    client = create_client()

    client.client.chat.return_value = MagicMock(
        message=MagicMock(
            content="Hello!",
            tool_calls=None,
        )
    )

    messages = [
        LLMMessage(
            role="user",
            content="Hello",
        )
    ]

    response = client.chat(
        messages=messages,
        tools=[],
    )

    assert response.content == "Hello!"
    assert response.has_tool_calls is False

    client.client.chat.assert_called_once()


def test_chat_with_tool_call():

    client = create_client()

    tool_call = MagicMock()

    tool_call.function.name = "knowledge_search"
    tool_call.function.arguments = {
        "query": "What is Python?"
    }

    client.client.chat.return_value = MagicMock(
        message=MagicMock(
            content="",
            tool_calls=[tool_call],
        )
    )

    messages = [
        LLMMessage(
            role="user",
            content="What is Python?",
        )
    ]

    tools = [
        LLMTool(
            name="knowledge_search",
            description="Search the knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    }
                },
                "required": ["query"],
            },
        )
    ]

    response = client.chat(
        messages=messages,
        tools=tools,
    )

    assert response.has_tool_calls is True
    assert len(response.tool_calls) == 1

    assert response.tool_calls[0].name == (
        "knowledge_search"
    )

    assert response.tool_calls[0].arguments == {
        "query": "What is Python?"
    }


def test_chat_converts_tools_to_ollama_format():

    client = create_client()

    client.client.chat.return_value = MagicMock(
        message=MagicMock(
            content="Done",
            tool_calls=None,
        )
    )

    tool = LLMTool(
        name="knowledge_search",
        description="Search the knowledge base.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                }
            },
            "required": ["query"],
        },
    )

    client.chat(
        messages=[
            LLMMessage(
                role="user",
                content="Search Python.",
            )
        ],
        tools=[tool],
    )

    call = client.client.chat.call_args

    assert call.kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "knowledge_search",
                "description": (
                    "Search the knowledge base."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def test_chat_converts_tool_call_message():

    client = create_client()

    client.client.chat.return_value = MagicMock(
        message=MagicMock(
            content="",
            tool_calls=None,
        )
    )

    messages = [
        LLMMessage(
            role="assistant",
            content="",
        )
    ]

    messages[0].tool_calls = []

    converted = client._to_ollama_message(
        messages[0]
    )

    assert converted == {
        "role": "assistant",
        "content": "",
    }


def test_embed():

    client = create_client()

    client.client.embed.return_value = {
        "embeddings": [
            [0.1, 0.2, 0.3]
        ]
    }

    result = client.embed("Python")

    assert result == [
        0.1,
        0.2,
        0.3,
    ]

    client.client.embed.assert_called_once_with(
        model="test-embedding",
        input="Python",
    )