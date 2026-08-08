from application.rag.chunk import Chunk
from application.rag.document import Document


class Chunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        content = document.content

        if not content:
            return []

        chunks = []

        start = 0
        chunk_index = 0

        step = self.chunk_size - self.chunk_overlap

        while start < len(content):
            end = min(
                start + self.chunk_size,
                len(content),
            )

            chunk_content = content[start:end]

            chunks.append(
                Chunk(
                    content=chunk_content,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                    },
                )
            )

            if end >= len(content):
                break

            start += step
            chunk_index += 1

        return chunks