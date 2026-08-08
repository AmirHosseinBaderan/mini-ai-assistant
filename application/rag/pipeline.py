from pathlib import Path

from application.rag.chunker import Chunker
from application.rag.indexer import Indexer
from application.rag.loader import DocumentLoader


class RAGPipeline:

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        indexer: Indexer,
    ):
        self.loader = loader
        self.chunker = chunker
        self.indexer = indexer

    def index_directory(
        self,
        directory: str | Path,
    ) -> None:

        documents = self.loader.load_directory(
            directory
        )

        chunks = []

        for document in documents:
            document_chunks = self.chunker.split(
                document
            )

            chunks.extend(document_chunks)

        self.indexer.index(chunks)