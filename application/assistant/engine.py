from typing import Iterator

from application.agent.agent import Agent


class AssistantEngine:

    def __init__(
        self,
        agent: Agent,
    ):
        self.agent = agent

    def stream(
        self,
        text: str,
    ) -> Iterator[str]:

        yield from self.agent.stream(text)