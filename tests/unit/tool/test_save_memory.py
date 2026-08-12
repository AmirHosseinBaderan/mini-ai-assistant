from unittest.mock import patch, mock_open
import json

import pytest

from application.tools.save_memory import SaveMemoryTool


def create_tool():
    return SaveMemoryTool()


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
    m.assert_called_once()

    # Verify json.dump was called with correct data
    written_data = None
    for call_args in m.return_value.write.call_args_list:
        pass  # json.dump writes directly to file handle

    # Get the file handle from the mock
    file_handle = m.return_value
    # json.dump is called, so we check the call
    assert m.call_args[0][0].endswith("memory.json")
    assert m.call_args[0][1] == "w"
    assert m.call_args[1]["encoding"] == "utf-8"


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
