import os
from dataclasses import dataclass


@dataclass
class Config:
    ollama_host: str
    ollama_model: str
    ollama_embedding_model: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str


def load_config() -> Config:
    return Config(
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "gemma3:1b"),
        ollama_embedding_model=os.environ.get("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:latest"),
        qdrant_host=os.environ.get("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
        qdrant_collection=os.environ.get("QDRANT_COLLECTION", "intent_dataset"),
    )
