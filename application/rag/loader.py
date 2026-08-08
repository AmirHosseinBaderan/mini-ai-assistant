from pathlib import Path

from application.rag.document import Document


class DocumentLoader:

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
    }

    def load(
        self,
        path: str | Path,
    ) -> Document:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path must be a file: {path}"
            )

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}"
            )

        content = path.read_text(
            encoding="utf-8"
        )

        return Document(
            content=content,
            metadata={
                "filename": path.name,
                "source": str(path),
            },
        )

    def load_directory(
        self,
        directory: str | Path,
    ) -> list[Document]:

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Path must be a directory: {directory}"
            )

        documents = []

        for path in sorted(directory.iterdir()):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in self.SUPPORTED_EXTENSIONS
            ):
                continue

            documents.append(
                self.load(path)
            )

        return documents