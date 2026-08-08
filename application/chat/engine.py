from collections.abc import Iterator

from application.llm.client import LLMClient


class ChatEngine:

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.messages: list[dict[str, str]] = []

    def stream(self, user_message: str) -> Iterator[str]:
        self.messages.append({
            "role": "user",
            "content": user_message,
        })

        response = []

        for chunk in self.llm.stream(self.messages):
            response.append(chunk)
            yield chunk

        assistant_message = "".join(response)

        self.messages.append({
            "role": "assistant",
            "content": assistant_message,
        })