from collections.abc import Iterator

from application.agent.agent import Agent
from application.chat.history import ConversationHistory


class AssistantEngine:

    def __init__(
        self,
        agent: Agent,
        history: ConversationHistory | None = None,
    ):
        self.agent = agent
        self.history = history or ConversationHistory()

    def stream(
        self,
        text: str,
    ) -> Iterator[str]:

        self.history.add_user(text)

        response = []

        for chunk in self.agent.stream(
            self.history.get_messages()
        ):
            response.append(chunk)
            yield chunk

        assistant_message = "".join(response)

        self.history.add_assistant(
            assistant_message
        )