from application.bootstrap import create_rag_engine
from application.rag.engine import RAGEngine


def test_create_rag_engine():

    engine = create_rag_engine()

    assert isinstance(
        engine,
        RAGEngine,
    )