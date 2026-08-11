
import pytest

from application.tools.base import Tool


class FakeTool(Tool):

    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "A fake tool for testing."

    def execute(self, **kwargs):
        return kwargs


def test_tool_metadata():

    tool = FakeTool()

    assert tool.name == "fake_tool"
    assert tool.description == "A fake tool for testing."


def test_tool_execute():

    tool = FakeTool()

    result = tool.execute(
        query="hello",
    )

    assert result == {
        "query": "hello",
    }


def test_tool_requires_implementation():

    class InvalidTool(Tool):
        pass

    with pytest.raises(TypeError):
        InvalidTool()