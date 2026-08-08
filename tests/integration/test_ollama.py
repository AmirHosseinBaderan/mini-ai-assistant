import os

import pytest

from application.llm.ollama_client import OllamaClient


@pytest.mark.integration
def test_ollama_stream():
    model = os.getenv("OLLAMA_MODEL")

    if not model:
        pytest.skip("OLLAMA_MODEL is not configured")

    client = OllamaClient()

    messages = [
        {
            "role": "user",
            "content": "Reply with exactly: hello",
        }
    ]

    chunks = list(client.stream(messages))

    response = "".join(chunks)

    assert response.strip()
    assert isinstance(response, str)