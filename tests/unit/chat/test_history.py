from application.chat.history import ConversationHistory


def test_add_user_message():
    history = ConversationHistory()

    history.add_user("Hello")

    assert history.get_messages() == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


def test_add_assistant_message():
    history = ConversationHistory()

    history.add_assistant("Hi!")

    assert history.get_messages() == [
        {
            "role": "assistant",
            "content": "Hi!",
        }
    ]


def test_conversation_order():
    history = ConversationHistory()

    history.add_user("Hello")
    history.add_assistant("Hi!")
    history.add_user("How are you?")

    assert history.get_messages() == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi!",
        },
        {
            "role": "user",
            "content": "How are you?",
        },
    ]


def test_clear():
    history = ConversationHistory()

    history.add_user("Hello")
    history.add_assistant("Hi!")

    history.clear()

    assert history.get_messages() == []


def test_get_messages_returns_copy():
    history = ConversationHistory()

    history.add_user("Hello")

    messages = history.get_messages()
    messages.clear()

    assert history.get_messages() == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]