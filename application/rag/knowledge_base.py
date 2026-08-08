from pathlib import Path

from application.rag.chunker import Chunker
from application.rag.indexer import Indexer
from application.rag.loader import DocumentLoader


class KnowledgeBase:

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        indexer: Indexer,
    ):
        self.loader = loader
        self.chunker = chunker
        self.indexer = indexer

    def add_file(self, path: Path) -> int:
        document = self.loader.load(path)

        chunks = self.chunker.split(
            document
        )

        self.indexer.index(chunks)

        return len(chunks)

    def add_directory(self, path: Path) -> int:
        total = 0

        for file in path.iterdir():

            if not file.is_file():
                continue

            try:
                total += self.add_file(file)
            except ValueError:
                continue

        return total