from application.llm.ollama_client import OllamaClient

from application.rag.chunker import Chunker
from application.rag.context_builder import ContextBuilder
from application.rag.engine import RAGEngine
from application.rag.indexer import Indexer
from application.rag.knowledge_base import KnowledgeBase
from application.rag.loader import DocumentLoader
from application.rag.ollama_embedding import (
    OllamaEmbeddingProvider,
)
from application.rag.qdrant_vector_store import (
    QdrantVectorStore,
)
from application.rag.retriever import Retriever


def create_llm_client():
    return OllamaClient()


def create_rag_components():

    llm_client = create_llm_client()

    embedding_provider = (
        OllamaEmbeddingProvider(
            client=llm_client,
        )
    )

    vector_size = len(
        embedding_provider.embed(
            "initialization"
        )
    )

    vector_store = QdrantVectorStore(
        collection_name="mini_ai_assistant",
        vector_size=vector_size,
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    context_builder = ContextBuilder()

    rag_engine = RAGEngine(
        retriever=retriever,
        context_builder=context_builder,
        llm_client=llm_client,
    )

    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    knowledge_base = KnowledgeBase(
        loader=DocumentLoader(),
        chunker=Chunker(
            chunk_size=500,
            chunk_overlap=50,
        ),
        indexer=indexer,
    )

    return {
        "llm_client": llm_client,
        "rag_engine": rag_engine,
        "knowledge_base": knowledge_base,
    }