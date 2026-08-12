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
    assert "key" in parameters["properties"]
    assert "value" in parameters["properties"]
    assert "key" in parameters["required"]
    assert "value" in parameters["required"]


def test_default_path_is_memo_json():
    tool = create_tool()
    assert str(tool.path).endswith("data/memory/memo.json")


def test_custom_path(tmp_path):
    custom_path = tmp_path / "custom_memory.json"
    tool = create_tool(path=str(custom_path))
    assert tool.path == custom_path


def test_execute_saves_key_value_to_file(tmp_path):
    memory_file = tmp_path / "memo.json"
    tool = create_tool(path=str(memory_file))

    result = tool.execute(
        key="name",
        value="Amir",
    )

    assert result.success is True
    assert "Successfully saved memory" in result.content
    assert memory_file.exists()

    with open(memory_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data == {"name": "Amir"}


def test_execute_saves_multiple_key_values(tmp_path):
    memory_file = tmp_path / "memo.json"
    tool = create_tool(path=str(memory_file))

    # Save first key-value
    tool.execute(key="name", value="Amir")
    # Save second key-value
    tool.execute(key="language", value="Python")

    with open(memory_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data == {"name": "Amir", "language": "Python"}


def test_execute_overwrites_existing_key(tmp_path):
    memory_file = tmp_path / "memo.json"
    tool = create_tool(path=str(memory_file))

    # Save initial value
    tool.execute(key="name", value="Amir")
    # Overwrite with new value
    result = tool.execute(key="name", value="Ali")

    assert result.success is True

    with open(memory_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data == {"name": "Ali"}


def test_execute_creates_parent_directories(tmp_path):
    memory_file = tmp_path / "subdir" / "nested" / "memo.json"
    tool = create_tool(path=str(memory_file))

    result = tool.execute(
        key="test",
        value="value",
    )

    assert result.success is True
    assert memory_file.exists()


def test_execute_requires_key():
    tool = create_tool()

    with pytest.raises(ValueError):
        tool.execute()


def test_execute_requires_value():
    tool = create_tool()

    with pytest.raises(ValueError):
        tool.execute(key="name")


def test_execute_rejects_empty_key():
    tool = create_tool()

    with pytest.raises(ValueError):
        tool.execute(
            key="   ",
            value="Amir",
        )


def test_execute_rejects_empty_value():
    tool = create_tool()

    with pytest.raises(ValueError):
        tool.execute(
            key="name",
            value="   ",
        )


def test_execute_rejects_non_string_key():
    tool = create_tool()

    with pytest.raises(ValueError):
        tool.execute(
            key=123,
            value="Amir",
        )


def test_execute_rejects_non_string_value():
    tool = create_tool()

    with pytest.raises(ValueError):
        tool.execute(
            key="name",
            value=123,
        )
