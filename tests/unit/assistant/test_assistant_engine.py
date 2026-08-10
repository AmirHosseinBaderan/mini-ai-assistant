from dataclasses import dataclass

from application.assistant.engine import AssistantEngine


@dataclass
class FakeRouteResult:
    label: str
    confidence: float
    accepted: bool


class FakeRouter:

    def __init__(self, result):
        self.result = result

    def route(self, text):
        return self.result


class FakeChatEngine:

    def stream(self, text):
        yield "chat:"
        yield text


class FakeRAGEngine:

    def stream(self, text):
        yield "rag:"
        yield text


def test_chat_route():

    router = FakeRouter(
        FakeRouteResult(
            label="chat",
            confidence=0.95,
            accepted=True,
        )
    )

    engine = AssistantEngine(
        router=router,
        chat_engine=FakeChatEngine(),
        rag_engine=FakeRAGEngine(),
    )

    result = "".join(
        engine.stream("hello")
    )

    assert result == "chat:hello"


def test_rag_route():

    router = FakeRouter(
        FakeRouteResult(
            label="rag",
            confidence=0.94,
            accepted=True,
        )
    )

    engine = AssistantEngine(
        router=router,
        chat_engine=FakeChatEngine(),
        rag_engine=FakeRAGEngine(),
    )

    result = "".join(
        engine.stream("what is python?")
    )

    assert result == "rag:what is python?"


def test_low_confidence_falls_back_to_chat():

    router = FakeRouter(
        FakeRouteResult(
            label="rag",
            confidence=0.51,
            accepted=False,
        )
    )

    engine = AssistantEngine(
        router=router,
        chat_engine=FakeChatEngine(),
        rag_engine=FakeRAGEngine(),
    )

    result = "".join(
        engine.stream("ambiguous question")
    )

    assert result == "chat:ambiguous question"


def test_unknown_route_raises():

    router = FakeRouter(
        FakeRouteResult(
            label="unknown",
            confidence=0.99,
            accepted=True,
        )
    )

    engine = AssistantEngine(
        router=router,
        chat_engine=FakeChatEngine(),
        rag_engine=FakeRAGEngine(),
    )

    try:
        list(engine.stream("test"))
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Unknown route label" in str(exc)