from pathlib import Path

from application.rag.document import Document


class DocumentLoader:

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
    }

    def load(self, path: str | Path) -> Document:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document type: {path.suffix}"
            )

        content = path.read_text(
            encoding="utf-8"
        )

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "extension": path.suffix.lower(),
            },
        )