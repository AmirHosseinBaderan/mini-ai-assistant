from application.rag.ollama_embedding import (
    OllamaEmbeddingProvider,
)
from application.rag.qdrant_vector_store import (
    QdrantVectorStore,
)
from application.rag.retriever import Retriever
from application.llm.ollama_client import OllamaClient

from dotenv import find_dotenv, load_dotenv


load_dotenv(
    find_dotenv(),
    verbose=True,
)


def test_qdrant_search():

    llm_client = OllamaClient()

    embedding_provider = OllamaEmbeddingProvider(
        client=llm_client,
    )

    vector_store = QdrantVectorStore(
        collection_name="mini_chat_collection",
        vector_size=4096,
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "What is Python?",
        top_k=3,
    )
    print(results)

    assert isinstance(results, list)

    for chunk, score in results:

        assert chunk.content
        assert isinstance(score, float)

def test_qdrant_collection():

    vector_store = QdrantVectorStore(
        collection_name="mini_chat_collection",
        vector_size=4096,
    )

    info = vector_store.client.get_collection(
        "mini_chat_collection",
    )

    print("STATUS:", info.status)
    print("POINTS:", info.points_count)

    assert info.points_count > 0