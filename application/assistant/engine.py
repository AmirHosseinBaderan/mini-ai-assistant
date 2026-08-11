from typing import Iterator

from application.router.router import Router


class AssistantEngine:

    def __init__(
        self,
        chat_engine,
    ):
        self.chat_engine = chat_engine

    def stream(
        self,
        text: str,
    ) -> Iterator[str]:
        yield from self.chat_engine.stream(text)