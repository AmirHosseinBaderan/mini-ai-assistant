import pytest

from application.tools.base import Tool
from application.tools.registry import ToolRegistry

class FakeTool(Tool):

    @property

    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "Fake tool."

    def execute(self, **kwargs):
        return "ok"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
        }

def test_register_tool():
    registry = ToolRegistry()
    tool = FakeTool()
    registry.register(tool)
    assert registry.get(
        "fake_tool"
    ) is tool

def test_list_tools():
    registry = ToolRegistry()
    tool = FakeTool()
    registry.register(tool)
    assert registry.all() == [tool]

def test_duplicate_tool():
    registry = ToolRegistry()
    registry.register(
        FakeTool()
    )
    with pytest.raises(ValueError):
        registry.register(
            FakeTool()
        )

def test_unknown_tool():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.get(
            "unknown_tool"
        )

def test_tool_schema():

    registry = ToolRegistry()

    tool = FakeTool()

    registry.register(tool)

    schemas = registry.schemas()

    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "fake_tool",
                "description": "Fake tool.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }
    ]