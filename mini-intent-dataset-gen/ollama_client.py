import requests
from typing import List


class OllamaClient:
    def __init__(self, host: str, gen_model: str, embed_model: str, timeout: int = 60):
        self.host = host
        self.gen_model = gen_model
        self.embed_model = embed_model
        self.timeout = timeout

    def generate(self, prompt: str, temperature: float = 0.8) -> str:
        """Call the generation model, return raw text response."""
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.gen_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def embed(self, text: str) -> List[float]:
        """Call the embedding model. Tries the newer /api/embed endpoint first,
        falls back to the older /api/embeddings endpoint for older Ollama servers."""
        try:
            resp = requests.post(
                f"{self.host}/api/embed",
                json={"model": self.embed_model, "input": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" in data:
                return data["embeddings"][0]
        except (requests.RequestException, KeyError, IndexError):
            pass

        resp = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
