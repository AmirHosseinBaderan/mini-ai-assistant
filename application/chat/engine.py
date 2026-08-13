from collections.abc import Iterator

from application.chat.history import ConversationHistory
from application.llm.client import LLMClient


class ChatEngine:

    def __init__(
        self,
        llm: LLMClient,
        history: ConversationHistory | None = None,
    ):
        self.llm = llm
        self.history = history or ConversationHistory()

    def stream(self, user_message: str) -> Iterator[str]:
        self.history.add_user(user_message)

        response = []
        try:
            for chunk in self.llm.stream(
                self.history.get_messages()
            ):
                response.append(chunk)
                yield chunk

        except Exception:
            raise

        assistant_message = "".join(response)

        self.history.add_assistant(assistant_message)