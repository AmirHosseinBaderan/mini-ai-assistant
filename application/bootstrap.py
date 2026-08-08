from application.llm.ollama_client import OllamaClient
from application.rag.context_builder import ContextBuilder
from application.rag.engine import RAGEngine
from application.rag.ollama_embedding import (
    OllamaEmbeddingProvider,
)
from application.rag.qdrant_vector_store import (
    QdrantVectorStore,
)
from application.rag.retriever import Retriever


def create_rag_engine() -> RAGEngine:

    llm_client = OllamaClient()

    embedding_provider = (
        OllamaEmbeddingProvider(
            client=llm_client,
        )
    )

    embedding = embedding_provider.embed(
        "initialization"
    )

    vector_store = QdrantVectorStore(
        collection_name="mini_ai_assistant",
        vector_size=len(embedding),
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    context_builder = ContextBuilder()

    return RAGEngine(
        retriever=retriever,
        context_builder=context_builder,
        llm_client=llm_client,
    )