from application.rag.embedding import EmbeddingProvider
from application.llm.client import LLMClient


class OllamaEmbeddingProvider(EmbeddingProvider):

    def __init__(
            self,
            client: LLMClient
    ):
        self.client = client

    def embed(self, text: str) -> list[float]:
        response = self.client.embed(
            text
        )
        return response
