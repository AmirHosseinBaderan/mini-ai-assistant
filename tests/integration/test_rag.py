import pytest

from application.llm.ollama_client import OllamaClient
from application.rag.chunker import Chunker
from application.rag.context_builder import ContextBuilder
from application.rag.engine import RAGEngine
from application.rag.indexer import Indexer
from application.rag.loader import DocumentLoader
from application.rag.ollama_embedding import (
    OllamaEmbeddingProvider,
)
from application.rag.pipeline import RAGPipeline
from application.rag.qdrant_vector_store import (
    QdrantVectorStore,
)
from application.rag.retriever import Retriever


@pytest.mark.integration
def test_end_to_end_rag(tmp_path):

    # --------------------------------------------------
    # Documents
    # --------------------------------------------------

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    (documents_dir / "python.txt").write_text(
        (
            "Python is a high-level programming "
            "language. It is widely used for "
            "software development, automation, "
            "data analysis and artificial intelligence."
        ),
        encoding="utf-8",
    )

    (documents_dir / "germany.txt").write_text(
        (
            "Germany is a country in Central Europe. "
            "Berlin is the capital of Germany."
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Ollama
    # --------------------------------------------------

    client = OllamaClient()

    embedding_provider = (
        OllamaEmbeddingProvider(
            client=client,
        )
    )

    # --------------------------------------------------
    # Qdrant
    # --------------------------------------------------

    test_embedding = embedding_provider.embed(
        "test"
    )

    vector_store = QdrantVectorStore(
        collection_name="mini_ai_rag_e2e_test",
        vector_size=len(test_embedding),
    )

    # --------------------------------------------------
    # Indexing
    # --------------------------------------------------

    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    pipeline = RAGPipeline(
        loader=DocumentLoader(),
        chunker=Chunker(
            chunk_size=200,
            chunk_overlap=20,
        ),
        indexer=indexer,
    )

    pipeline.index_directory(
        documents_dir
    )

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "What is Python?",
        top_k=2,
    )

    assert results
    assert any(
        "Python" in chunk.content
        for chunk, _score in results
    )

    # --------------------------------------------------
    # RAG Engine
    # --------------------------------------------------

    engine = RAGEngine(
        retriever=retriever,
        context_builder=ContextBuilder(),
        llm_client=client,
    )

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    answer = "".join(
        engine.stream(
            "What is Python?"
        )
    )

    assert answer.strip()