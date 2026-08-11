from application.llm.stream_event import LLMStreamEvent


def test_text_stream_event():

    event = LLMStreamEvent(
        type="text",
        content="Hello",
    )

    assert event.type == "text"
    assert event.content == "Hello"
    assert event.tool_name is None
    assert event.tool_arguments is None


def test_tool_call_stream_event():

    event = LLMStreamEvent(
        type="tool_call",
        tool_name="knowledge_search",
        tool_arguments={
            "query": "Python",
        },
    )

    assert event.type == "tool_call"
    assert event.tool_name == "knowledge_search"
    assert event.tool_arguments == {
        "query": "Python",
    }


def test_done_stream_event():

    event = LLMStreamEvent(
        type="done",
    )

    assert event.type == "done"
    assert event.content is None
    assert event.tool_name is None
    assert event.tool_arguments is None