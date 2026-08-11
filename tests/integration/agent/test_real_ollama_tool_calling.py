from unittest.mock import Mock

from application.agent.agent import Agent
from application.llm.ollama_client import OllamaClient
from application.rag.chunk import Chunk
from application.rag.retriever import Retriever
from application.tools.knowledge_search import KnowledgeSearchTool
from application.tools.registry import ToolRegistry

from application.llm.message import LLMMessage
from dotenv import find_dotenv, load_dotenv
import json
from application.llm.response import ToolCall

load_dotenv(
    find_dotenv(),
    verbose=True,
)

def test_real_ollama_tool_calling():

    llm_client = OllamaClient()

    retriever = Mock(spec=Retriever)

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
        knowledge_tool,
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    result = "".join(
        agent.stream(
            "Use the knowledge search tool "
            "to find information about Python."
        )
    )

    assert result.strip()

    retriever.retrieve.assert_called_once()

    call = retriever.retrieve.call_args

    query = call.args[0]

    assert isinstance(query, str)

    assert "python" in query.lower()
