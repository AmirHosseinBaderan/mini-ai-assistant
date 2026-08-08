from typing import Iterator

from application.llm.client import LLMClient
from application.rag.context_builder import ContextBuilder
from application.rag.retriever import Retriever


class RAGEngine:

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        llm_client: LLMClient,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm_client = llm_client

    def stream(
        self,
        query: str,
        top_k: int = 3,
    ) -> Iterator[str]:

        results = self.retriever.retrieve(
            query,
            top_k=top_k,
        )

        chunks = [
            chunk
            for chunk, _score in results
        ]

        context = self.context_builder.build(
            chunks
        )

        messages = self._build_messages(
            query=query,
            context=context,
        )

        yield from self.llm_client.stream(
            messages
        )

    def _build_messages(
            self,
            query: str,
            context: str,
    ) -> list[dict[str, str]]:

        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer the user's question using "
            "the provided context. "
            "If the answer is not present in "
            "the context, say that you don't "
            "know based on the provided documents."
        )

        if context:
            user_prompt = (
                f"Context:\n"
                f"{context}\n\n"
                f"Question:\n"
                f"{query}"
            )
        else:
            user_prompt = (
                f"Context:\n\n"
                f"Question:\n"
                f"{query}"
            )

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]