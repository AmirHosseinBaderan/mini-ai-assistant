from unittest.mock import patch, mock_open
import json

import pytest

from application.tools.load_memory import LoadMemoryTool


def create_tool():
    return LoadMemoryTool()


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
    assert parameters["required"] == []


def test_execute_loads_memory_from_file():
    tool = create_tool()
    memory_content = "User likes Python and coffee"
    memory_data = {"memory": memory_content}

    m = mock_open(
        read_data=json.dumps(memory_data),
    )

    with patch("application.tools.load_memory.open", m):
        with patch("application.tools.load_memory.os.path.exists", return_value=True):
            result = tool.execute()

    assert result.success is True
    assert result.content == memory_content


def test_execute_returns_message_when_no_memory_file():
    tool = create_tool()

    with patch("application.tools.load_memory.os.path.exists", return_value=False):
        result = tool.execute()

    assert result.success is True
    assert "No memory found" in result.content


def test_execute_returns_message_when_memory_empty():
    tool = create_tool()
    memory_data = {"memory": ""}

    m = mock_open(
        read_data=json.dumps(memory_data),
    )

    with patch("application.tools.load_memory.open", m):
        with patch("application.tools.load_memory.os.path.exists", return_value=True):
            result = tool.execute()

    assert result.success is True
    assert "empty" in result.content.lower()


def test_execute_handles_invalid_json():
    tool = create_tool()

    m = mock_open(
        read_data="not valid json",
    )

    with patch("application.tools.load_memory.open", m):
        with patch("application.tools.load_memory.os.path.exists", return_value=True):
            result = tool.execute()

    assert result.success is False
    assert "Error loading memory" in result.content


def test_execute_handles_io_error():
    tool = create_tool()

    m = mock_open()
    m.side_effect = IOError("Permission denied")

    with patch("application.tools.load_memory.open", m):
        with patch("application.tools.load_memory.os.path.exists", return_value=True):
            result = tool.execute()

    assert result.success is False
    assert "Error loading memory" in result.content
