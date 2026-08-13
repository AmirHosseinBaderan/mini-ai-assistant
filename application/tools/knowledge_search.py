from typing import Any

from application.rag.retriever import Retriever
from application.tools.base import Tool
from application.tools.result import ToolResult


class KnowledgeSearchTool(Tool):

    def __init__(
        self,
        retriever: Retriever,
    ):
        self.retriever = retriever

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return (
            "Search internal documents, technical specs, and stored knowledge "
            "to answer questions that require specific facts, data, or "
            "documentation you don't already have from the conversation. "
            "ALWAYS call this tool when the user asks about a specific topic, "
            "technical detail, product spec, or documentation that could exist "
            "in the knowledge base — even if you think you might already know "
            "the answer, prefer verifying with this tool over guessing. "
            "Do NOT use for casual conversation, greetings, personal facts "
            "about the user (use load_memory for those instead), or general "
            "reasoning that doesn't need external facts. "
            "If no relevant documents are found, this returns an empty result "
            "rather than an error — in that case, say you don't have that "
            "information rather than making something up."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search internal documents, technical specs, and stored knowledge "
                        "to answer questions that require specific facts, data, or "
                        "documentation you don't already have from the conversation. "
                        "Do NOT use for casual conversation, greetings, or general "
                        "reasoning that doesn't need external facts."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Maximum number of "
                        "results to return."
                    ),
                },
            },
            "required": ["query"],
        }

    def execute(
        self,
        **kwargs: Any,
    ) -> ToolResult:

        query = kwargs.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query is required"
            )

        top_k = kwargs.get(
            "top_k",
            3,
        )

        results = self.retriever.retrieve(
            query,
            top_k=top_k,
        )

        content = [
            {
                "content": chunk.content,
                "score": score,
                "metadata": chunk.metadata,
            }
            for chunk, score in results
        ]

        return ToolResult(
            content=content,
        )