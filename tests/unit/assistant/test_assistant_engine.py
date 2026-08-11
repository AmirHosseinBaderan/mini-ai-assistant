from unittest.mock import Mock

from application.assistant.engine import AssistantEngine
from application.chat.history import ConversationHistory


def test_assistant_engine_stores_user_and_assistant_messages():

    agent = Mock()

    agent.stream.return_value = iter(
        [
            "Hello",
            " Amir",
        ]
    )

    history = ConversationHistory()

    engine = AssistantEngine(
        agent=agent,
        history=history,
    )

    result = "".join(
        engine.stream("Hi")
    )

    assert result == "Hello Amir"

    messages = history.get_messages()

    assert len(messages) == 2

    assert messages[0].role == "user"
    assert messages[0].content == "Hi"

    assert messages[1].role == "assistant"
    assert messages[1].content == "Hello Amir"


def test_assistant_engine_passes_history_to_agent():

    agent = Mock()

    agent.stream.return_value = iter(
        ["Amir"]
    )

    history = ConversationHistory()

    engine = AssistantEngine(
        agent=agent,
        history=history,
    )

    list(
        engine.stream("My name is Amir")
    )

    agent.stream.assert_called_once()

    messages = agent.stream.call_args.args[0]

    assert len(messages) == 1

    assert messages[0].role == "user"
    assert messages[0].content == "My name is Amir"


def test_assistant_engine_preserves_previous_history():

    agent = Mock()

    agent.stream.side_effect = [
        iter(["Nice to meet you."]),
        iter(["Your name is Amir."]),
    ]

    history = ConversationHistory()

    engine = AssistantEngine(
        agent=agent,
        history=history,
    )

    first_result = "".join(
        engine.stream("My name is Amir")
    )

    second_result = "".join(
        engine.stream("What is my name?")
    )

    assert first_result == "Nice to meet you."

    assert second_result == "Your name is Amir."

    assert agent.stream.call_count == 2

    second_messages = (
        agent.stream.call_args_list[1]
        .args[0]
    )

    assert len(second_messages) == 3

    assert second_messages[0].role == "user"
    assert second_messages[0].content == (
        "My name is Amir"
    )

    assert second_messages[1].role == "assistant"
    assert second_messages[1].content == (
        "Nice to meet you."
    )

    assert second_messages[2].role == "user"
    assert second_messages[2].content == (
        "What is my name?"
    )