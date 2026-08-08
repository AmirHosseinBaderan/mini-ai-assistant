import pytest

from application.llm.ollama_client import OllamaClient
from application.rag.chunker import Chunker
from application.rag.indexer import Indexer
from application.rag.in_memory_vector_store import (
    InMemoryVectorStore,
)
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
def test_real_rag_pipeline(tmp_path):

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    (documents_dir / "python.txt").write_text(
        "Python is a programming language used "
        "to build software.",
        encoding="utf-8",
    )

    (documents_dir / "germany.txt").write_text(
        "Germany is a country located in Europe.",
        encoding="utf-8",
    )

    client = OllamaClient()

    embedding_provider = OllamaEmbeddingProvider(
        client=client,
    )

    test_embedding = embedding_provider.embed(
        "test"
    )

    vector_store = QdrantVectorStore(
        collection_name="mini_ai_pipeline_test",
        vector_size=len(test_embedding),
    )

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

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "What is Python?",
        top_k=1,
    )

    assert len(results) == 1

    assert (
        "Python"
        in results[0][0].content
    )