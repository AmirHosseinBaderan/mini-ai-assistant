from application.llm.ollama_client import OllamaClient
from application.rag.chunk import Chunk
from application.rag.indexer import Indexer
from application.rag.ollama_embedding import (
    OllamaEmbeddingProvider,
)
from application.rag.qdrant_vector_store import (
    QdrantVectorStore,
)

from dotenv import find_dotenv, load_dotenv


load_dotenv(
    find_dotenv(),
    verbose=True,
)


def test_real_indexer():

    llm_client = OllamaClient()

    embedding_provider = OllamaEmbeddingProvider(
        client=llm_client,
    )

    vector_store = QdrantVectorStore(
        collection_name="mini_chat_collection",
        vector_size=4096,
    )

    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    chunks = [
        Chunk(
            content=(
                "Python is a high-level programming language "
                "used for backend development, automation, "
                "data science, and machine learning."
            ),
            metadata={
                "source": "python.txt",
            },
        ),
        Chunk(
            content=(
                "Python uses indentation to define code blocks "
                "and supports object-oriented programming."
            ),
            metadata={
                "source": "python.txt",
            },
        ),
    ]

    indexer.index(chunks)

    info = vector_store.client.get_collection(
        "mini_chat_collection",
    )

    print("POINTS:", info.points_count)

    assert info.points_count >= 2