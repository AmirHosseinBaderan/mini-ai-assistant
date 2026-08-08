from application.rag.document import Document
from application.rag.pipeline import RAGPipeline


class FakeLoader:

    def load_directory(self, directory):

        return [
            Document(
                content="Python is a programming language."
            ),
            Document(
                content="Germany is located in Europe."
            ),
        ]


class FakeChunker:

    def split(self, document):

        return [
            document,
        ]


class FakeIndexer:

    def __init__(self):

        self.chunks = []

    def index(self, chunks):

        self.chunks.extend(chunks)


def test_pipeline_indexes_documents():

    loader = FakeLoader()
    chunker = FakeChunker()
    indexer = FakeIndexer()

    pipeline = RAGPipeline(
        loader=loader,
        chunker=chunker,
        indexer=indexer,
    )

    pipeline.index_directory(
        "data/documents"
    )

    assert len(indexer.chunks) == 2

    assert (
        indexer.chunks[0].content
        == "Python is a programming language."
    )

    assert (
        indexer.chunks[1].content
        == "Germany is located in Europe."
    )