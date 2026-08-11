from typing import Any

from application.rag.retriever import Retriever
from application.tools.base import Tool

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
            "Search the user's knowledge base "
            "for information relevant to the query."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The query to search "
                        "in the knowledge base."
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
    ) -> list[dict[str, Any]]:

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

        return [
            {
                "content": chunk.content,
                "score": score,
                "metadata": chunk.metadata,
            }
            for chunk, score in results
        ]