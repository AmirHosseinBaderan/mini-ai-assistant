from unittest.mock import patch, mock_open
import json

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
    assert tool.path.endswith("data/memory/memo.json")


def test_custom_path():
    custom_path = "/tmp/custom_memory.json"
    tool = create_tool(path=custom_path)
    assert tool.path == custom_path


def test_execute_saves_memory_to_file():
    tool = create_tool()
    m = mock_open()

    with patch("application.tools.save_memory.open", m):
        with patch("application.tools.save_memory.os.path.exists", return_value=False):
            with patch("application.tools.save_memory.os.makedirs"):
                result = tool.execute(
                    details="User likes Python and coffee",
                )

    assert result.success is True
    assert "Successfully saved memory" in result.content
    m.assert_called_once_with(
        tool.path,
        "w",
        encoding="utf-8",
    )


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
