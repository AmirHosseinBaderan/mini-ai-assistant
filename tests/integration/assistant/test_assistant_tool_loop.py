from unittest.mock import Mock

from application.agent.agent import Agent
from application.assistant.engine import AssistantEngine
from application.chat.history import ConversationHistory
from application.llm.response import LLMResponse, ToolCall
from application.rag.chunk import Chunk
from application.rag.retriever import Retriever
from application.tools.knowledge_search import KnowledgeSearchTool
from application.tools.registry import ToolRegistry


class FakeLLM:

    def __init__(self):

        self.calls = 0
        self._last_response = None

    def chat(
        self,
        messages,
        tools,
    ):

        self.calls += 1

        if self.calls == 1:

            self._last_response = LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="knowledge_search",
                        arguments={
                            "query": "What is Python?"
                        },
                    )
                ]
            )

            return self._last_response

        self._last_response = LLMResponse(
            content=(
                "Python is a programming "
                "language."
            )
        )

        return self._last_response

    def stream(
        self,
        messages,
    ):
        if self._last_response and self._last_response.content:
            yield self._last_response.content


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
        history=ConversationHistory(),
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