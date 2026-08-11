from application.llm.ollama_client import OllamaClient
from application.llm.message import LLMMessage
from application.llm.stream_event import LLMStreamEvent
from application.llm.tool import LLMTool
from dotenv import find_dotenv, load_dotenv
load_dotenv(
    find_dotenv(),
    verbose=True,
)


def test_ollama_stream_chat():

    client = OllamaClient()

    tool = LLMTool(
        name="knowledge_search",
        description=(
            "Search the knowledge base "
            "for relevant information."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                },
            },
            "required": ["query"],
        },
    )

    events = list(
        client.stream_chat(
            messages=[
                LLMMessage(
                    role="user",
                    content=(
                        "Use the knowledge_search "
                        "tool to find information "
                        "about Python."
                    ),
                )
            ],
            tools=[tool],
        )
    )

    for event in events:
        print(event)

    assert events
    assert events[-1].type == "done"

    tool_events = [
        event
        for event in events
        if event.type == "tool_call"
    ]

    assert tool_events

    assert (
        tool_events[0].tool_name
        == "knowledge_search"
    )

    assert (
        "query"
        in tool_events[0].tool_arguments
    )