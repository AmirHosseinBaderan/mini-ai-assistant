import json
from pathlib import Path

import pytest

from application.tools.save_memory import SaveMemoryTool


def create_tool(path=None):
    return SaveMemoryTool(path=path)


def test_tool_name():
    tool = create_tool()
    assert tool.name == "save_memory"


def test_tool_description():
    tool = create_tool()
    assert "memory" in tool.description.lower()
    assert "save" in tool.description.lower()


def test_tool_parameters():
    tool = create_tool()
    parameters = tool.parameters

    assert parameters["type"] == "object"
    assert "details" in parameters["properties"]
    assert "details" in parameters["required"]


def test_default_path_is_memo_json():
    tool = create_tool()
    assert str(tool.path).endswith("data/memory/memo.json")


def test_custom_path(tmp_path):
    custom_path = tmp_path / "custom_memory.json"
    tool = create_tool(path=str(custom_path))
    assert tool.path == custom_path


def test_execute_saves_memory_to_file(tmp_path):
    memory_file = tmp_path / "memo.json"
    tool = create_tool(path=str(memory_file))

    result = tool.execute(
        details="User likes Python and coffee",
    )

    assert result.success is True
    assert "Successfully saved memory" in result.content
    assert memory_file.exists()

    with open(memory_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data == {"memory": "User likes Python and coffee"}


def test_execute_creates_parent_directories(tmp_path):
    memory_file = tmp_path / "subdir" / "nested" / "memo.json"
    tool = create_tool(path=str(memory_file))

    result = tool.execute(
        details="Test details",
    )

    assert result.success is True
    assert memory_file.exists()


def test_execute_requires_details():
    tool = create_tool()

    with pytest.raises(ValueError):
        tool.execute()


def test_execute_rejects_empty_details():
    tool = create_tool()

    with pytest.raises(ValueError):
        tool.execute(
            details="   ",
        )


def test_execute_rejects_non_string_details():
    tool = create_tool()

    with pytest.raises(ValueError):
        tool.execute(
            details=123,
        )
