from unittest.mock import Mock

from application.agent.agent import Agent
from application.assistant.engine import AssistantEngine
from application.llm.response import LLMResponse, ToolCall
from application.rag.chunk import Chunk
from application.rag.retriever import Retriever
from application.tools.knowledge_search import KnowledgeSearchTool
from application.tools.registry import ToolRegistry


class FakeLLM:

    def __init__(self):

        self.calls = 0

    def chat(
        self,
        messages,
        tools,
    ):

        self.calls += 1

        if self.calls == 1:

            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="knowledge_search",
                        arguments={
                            "query": "What is Python?"
                        },
                    )
                ]
            )

        return LLMResponse(
            content=(
                "Python is a programming "
                "language."
            )
        )


def test_assistant_executes_knowledge_tool():

    llm = FakeLLM()

    retriever = Mock(spec=Retriever)

    chunk = Chunk(
        content=(
            "Python is a high-level "
            "programming language."
        ),
        metadata={
            "source": "python.txt",
        },
    )

    retriever.retrieve.return_value = [
        (chunk, 0.95),
    ]

    knowledge_tool = KnowledgeSearchTool(
        retriever=retriever,
    )

    registry = ToolRegistry()

    registry.register(
        knowledge_tool,
    )

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    assistant = AssistantEngine(
        agent=agent,
    )

    result = "".join(
        assistant.stream(
            "What is Python?"
        )
    )

    assert result == (
        "Python is a programming "
        "language."
    )

    assert llm.calls == 2

    retriever.retrieve.assert_called_once_with(
        "What is Python?",
        top_k=3,
    )