import json
from pathlib import Path

import pytest

from application.tools.update_memory import UpdateMemoryTool

DEFAULT_MEMORY_PATH = (
    Path("./data/memory")
    / "memo.json"
)


def create_tool(path=None):
    return UpdateMemoryTool(path=path)


def test_tool_name():
    tool = create_tool()
    assert tool.name == "update_memory"


def test_tool_description():
    tool = create_tool()
    assert "memory" in tool.description.lower()
    assert "update" in tool.description.lower()


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


def test_execute_updates_existing_key(tmp_path):
    memory_file = tmp_path / "memo.json"
    memory_data = {"name": "Amir", "language": "Python"}

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f)

    tool = create_tool(path=str(memory_file))
    result = tool.execute(key="name", value="Ali")

    assert result.success is True
    assert "updated" in result.content.lower()
    assert "Ali" in result.content

    with open(memory_file, "r", encoding="utf-8") as f:
        updated_data = json.load(f)

    assert updated_data == {"name": "Ali", "language": "Python"}


def test_execute_preserves_other_keys(tmp_path):
    memory_file = tmp_path / "memo.json"
    memory_data = {"name": "Amir", "language": "Python", "city": "Tehran"}

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f)

    tool = create_tool(path=str(memory_file))
    result = tool.execute(key="language", value="JavaScript")

    assert result.success is True

    with open(memory_file, "r", encoding="utf-8") as f:
        updated_data = json.load(f)

    assert updated_data == {
        "name": "Amir",
        "language": "JavaScript",
        "city": "Tehran",
    }


def test_execute_returns_error_when_key_not_found(tmp_path):
    memory_file = tmp_path / "memo.json"
    memory_data = {"name": "Amir"}

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f)

    tool = create_tool(path=str(memory_file))
    result = tool.execute(key="language", value="Python")

    assert result.success is False
    assert "not found" in result.content.lower()


def test_execute_returns_error_when_no_memory_file(tmp_path):
    memory_file = tmp_path / "nonexistent.json"
    tool = create_tool(path=str(memory_file))

    result = tool.execute(key="name", value="Amir")

    assert result.success is False
    assert "No memory found" in result.content


def test_execute_returns_error_when_memory_empty(tmp_path):
    memory_file = tmp_path / "memo.json"
    memory_data = {}

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f)

    tool = create_tool(path=str(memory_file))
    result = tool.execute(key="name", value="Amir")

    assert result.success is False
    assert "not found" in result.content.lower()


def test_execute_handles_invalid_json(tmp_path):
    memory_file = tmp_path / "memo.json"

    with open(memory_file, "w", encoding="utf-8") as f:
        f.write("not valid json")

    tool = create_tool(path=str(memory_file))
    result = tool.execute(key="name", value="Amir")

    assert result.success is False
    assert "Error reading memory" in result.content


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


def test_execute_updates_from_default_path():
    """Test that update_memory works with the actual default path"""
    # Ensure file exists with known content (different from existing file content)
    DEFAULT_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump({"name": "Amir", "language": "Python", "city": "Tehran"}, f)

    tool = create_tool()  # uses default path
    result = tool.execute(key="name", value="Ali")

    assert result.success is True
    assert "Ali" in result.content

    with open(DEFAULT_MEMORY_PATH, "r", encoding="utf-8") as f:
        updated_data = json.load(f)

    assert updated_data == {"name": "Ali", "language": "Python", "city": "Tehran"}
    print(f"\n[update_memory test] file at: {DEFAULT_MEMORY_PATH}")
    print(f"[update_memory test] content: {updated_data}")
