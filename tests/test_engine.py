
from collections.abc import Iterator

from application.chat.engine import ChatEngine
from application.llm.client import LLMClient


class FakeLLM(LLMClient):

    def __init__(self):
        self.received_messages = []

    def stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:

        self.received_messages = messages

        yield "Hello"
        yield " from"
        yield " FakeLLM"


def test_chat_stream():
    llm = FakeLLM()
    chat = ChatEngine(llm)

    chunks = list(
        chat.stream("Hi")
    )

    assert chunks == [
        "Hello",
        " from",
        " FakeLLM",
    ]


def test_assistant_response_is_saved():
    llm = FakeLLM()
    chat = ChatEngine(llm)

    list(chat.stream("Hi"))

    assert chat.history.get_messages() == [
        {
            "role": "user",
            "content": "Hi",
        },
        {
            "role": "assistant",
            "content": "Hello from FakeLLM",
        },
    ]


def test_history_is_sent_to_llm():
    llm = FakeLLM()
    chat = ChatEngine(llm)

    list(chat.stream("Hello"))

    assert llm.received_messages == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


def test_multi_turn_conversation():
    llm = FakeLLM()
    chat = ChatEngine(llm)

    list(chat.stream("Hello"))
    list(chat.stream("How are you?"))

    assert chat.history.get_messages() == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hello from FakeLLM",
        },
        {
            "role": "user",
            "content": "How are you?",
        },
        {
            "role": "assistant",
            "content": "Hello from FakeLLM",
        },
    ]