from collections.abc import Iterator, AsyncIterator

from application.agent.agent import Agent
from application.chat.history import ConversationHistory
from application.llm.message import LLMMessage


class AssistantEngine:

    def __init__(
        self,
        agent: Agent,
        history: ConversationHistory,
    ):
        self.agent = agent
        self.history = history

    def stream(
        self,
        text: str,
    ) -> Iterator[str]:

        self.history.add_user(text)

        messages = self.history.get_messages()

        response = []

        for chunk in self.agent.stream(messages):
            response.append(chunk)
            yield chunk

        self.history.add_assistant(
            "".join(response)
        )

    async def astream(
            self,
            text: str,
    ) -> AsyncIterator[str]:

        messages = self.history.get_messages()

        messages.append(
            LLMMessage(
                role="user",
                content=text,
            )
        )

        self.history.add_user(
            content=text
        )

        assistant_chunks = []

        async for chunk in self.agent.astream(
                messages
        ):
            assistant_chunks.append(chunk)

            yield chunk

        self.history.add_assistant(
            content="".join(assistant_chunks)
        )