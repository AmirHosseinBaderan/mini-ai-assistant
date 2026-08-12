import json
from pathlib import Path

import pytest

from application.tools.load_memory import LoadMemoryTool


def create_tool(path=None):
    return LoadMemoryTool(path=path)


def test_tool_name():
    tool = create_tool()
    assert tool.name == "load_memory"


def test_tool_description():
    tool = create_tool()
    assert "memory" in tool.description.lower()
    assert "load" in tool.description.lower()


def test_tool_parameters():
    tool = create_tool()
    parameters = tool.parameters

    assert parameters["type"] == "object"
    assert "key" in parameters["properties"]
    assert parameters["required"] == []


def test_default_path_is_memo_json():
    tool = create_tool()
    assert str(tool.path).endswith("data/memory/memo.json")


def test_custom_path(tmp_path):
    custom_path = tmp_path / "custom_memory.json"
    tool = create_tool(path=str(custom_path))
    assert tool.path == custom_path


def test_execute_loads_all_memory(tmp_path):
    memory_file = tmp_path / "memo.json"
    memory_data = {"name": "Amir", "language": "Python"}

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f)

    tool = create_tool(path=str(memory_file))
    result = tool.execute()

    assert result.success is True
    assert result.content == memory_data


def test_execute_loads_specific_key(tmp_path):
    memory_file = tmp_path / "memo.json"
    memory_data = {"name": "Amir", "language": "Python"}

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f)

    tool = create_tool(path=str(memory_file))
    result = tool.execute(key="name")

    assert result.success is True
    assert result.content == "Amir"


def test_execute_returns_message_when_key_not_found(tmp_path):
    memory_file = tmp_path / "memo.json"
    memory_data = {"name": "Amir"}

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f)

    tool = create_tool(path=str(memory_file))
    result = tool.execute(key="language")

    assert result.success is True
    assert "not found" in result.content.lower()


def test_execute_returns_message_when_no_memory_file(tmp_path):
    memory_file = tmp_path / "nonexistent.json"
    tool = create_tool(path=str(memory_file))

    result = tool.execute()

    assert result.success is True
    assert "No memory found" in result.content


def test_execute_returns_message_when_memory_empty(tmp_path):
    memory_file = tmp_path / "memo.json"
    memory_data = {}

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f)

    tool = create_tool(path=str(memory_file))
    result = tool.execute()

    assert result.success is True
    assert "empty" in result.content.lower()


def test_execute_handles_invalid_json(tmp_path):
    memory_file = tmp_path / "memo.json"

    with open(memory_file, "w", encoding="utf-8") as f:
        f.write("not valid json")

    tool = create_tool(path=str(memory_file))
    result = tool.execute()

    assert result.success is False
    assert "Error loading memory" in result.content


def test_execute_handles_io_error(tmp_path):
    memory_file = tmp_path / "memo.json"
    tool = create_tool(path=str(memory_file))

    # Make the file unreadable
    memory_file.write_text("test")
    memory_file.chmod(0o000)

    try:
        result = tool.execute()
        assert result.success is False
        assert "Error loading memory" in result.content
    finally:
        # Restore permissions so pytest can clean up
        memory_file.chmod(0o644)
