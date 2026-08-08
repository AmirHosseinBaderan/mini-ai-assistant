from application.rag.chunk import Chunk


class ContextBuilder:

    def build(
        self,
        chunks: list[Chunk],
    ) -> str:

        if not chunks:
            return ""

        contexts = []

        for chunk in chunks:
            source = chunk.metadata.get(
                "filename",
                chunk.metadata.get(
                    "source",
                    "unknown",
                ),
            )

            contexts.append(
                f"[Source: {source}]\n"
                f"{chunk.content}"
            )

        return "\n\n".join(contexts)