from unittest.mock import Mock

from application.agent.agent import Agent
from application.llm.message import LLMMessage
from application.llm.ollama_client import OllamaClient
from application.rag.chunk import Chunk
from application.rag.retriever import Retriever
from application.tools.knowledge_search import KnowledgeSearchTool
from application.tools.registry import ToolRegistry

from dotenv import find_dotenv, load_dotenv


load_dotenv(
    find_dotenv(),
    verbose=True,
)

def test_real_ollama_tool_calling():

    llm_client = OllamaClient()

    retriever = Mock(
        spec=Retriever
    )

    retriever.retrieve.return_value = [
        (
            Chunk(
                content=(
                    "Python is a high-level "
                    "programming language."
                ),
                metadata={
                    "source": "python.txt",
                },
            ),
            0.95,
        ),
    ]

    knowledge_tool = KnowledgeSearchTool(
        retriever=retriever,
    )

    tool_registry = ToolRegistry()

    tool_registry.register(
        knowledge_tool
    )

    tool_calls = []

    def on_tool_call(tool_name: str):

        tool_calls.append(
            tool_name
        )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
        on_tool_call=on_tool_call,
    )

    result = "".join(
        agent.stream(
            [
                LLMMessage(
                    role="user",
                    content=(
                        "Use the knowledge search tool "
                        "to find information about Python."
                    ),
                )
            ]
        )
    )

    print()
    print("RESULT:")
    print(result)

    print()
    print("TOOL CALLS:")
    print(tool_calls)

    print()
    print("RETRIEVER CALLS:")
    print(
        retriever.retrieve.call_args_list
    )

    assert result.strip()

    if tool_calls:

        assert (
            "knowledge_search"
            in tool_calls
        )

        retriever.retrieve.assert_called()

        call = (
            retriever
            .retrieve
            .call_args
        )

        query = call.args[0]

        assert isinstance(
            query,
            str,
        )

        assert query.strip()