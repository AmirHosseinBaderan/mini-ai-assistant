from application.assistant.engine import AssistantEngine
from application.bootstrap import create_assistant

from dotenv import find_dotenv, load_dotenv


load_dotenv(
    find_dotenv(),
    verbose=True,
)

def test_create_assistant():

    assistant = create_assistant()

    assert isinstance(
        assistant,
        AssistantEngine,
    )

    assert assistant.agent is not None

    assert assistant.agent.llm_client is not None

    assert assistant.agent.tool_registry is not None