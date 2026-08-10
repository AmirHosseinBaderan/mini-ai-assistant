from typing import Iterator

from application.router.router import Router


class AssistantEngine:

    def __init__(
        self,
        router: Router,
        chat_engine,
        rag_engine,
    ):
        self.router = router
        self.chat_engine = chat_engine
        self.rag_engine = rag_engine

    def stream(
        self,
        text: str,
    ) -> Iterator[str]:

        route = self.router.route(text)

        if not route.accepted:
            yield from self.chat_engine.stream(text)
            return

        if route.label == "chat":
            yield from self.chat_engine.stream(text)
            return

        if route.label == "rag":
            yield from self.rag_engine.stream(text)
            return

        raise ValueError(
            f"Unknown route label: {route.label}"
        )