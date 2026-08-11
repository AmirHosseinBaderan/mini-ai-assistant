from application.llm.message import LLMMessage


class ConversationHistory:

    def __init__(self):
        self._messages: list[LLMMessage] = []

    def add_user(
        self,
        content: str,
    ) -> None:

        self._messages.append(
            LLMMessage(
                role="user",
                content=content,
            )
        )

    def add_assistant(
        self,
        content: str,
    ) -> None:

        self._messages.append(
            LLMMessage(
                role="assistant",
                content=content,
            )
        )

    def get_messages(self) -> list[LLMMessage]:
        return self._messages.copy()

    def clear(self) -> None:
        self._messages.clear()