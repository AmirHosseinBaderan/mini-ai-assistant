import pytest

from application.tools.bootstrap import register_mcp_tools
from application.tools.registry import ToolRegistry


class FakeTool:

    name = "product_search"
    description = "Search products."
    parameters = {}


class FakeClient:
    pass


@pytest.mark.anyio
async def test_register_mcp_tools(
    monkeypatch,
):

    async def fake_discover_tools(
        clients,
    ):

        return [
            FakeTool(),
        ]

    monkeypatch.setattr(
        "application.tools.bootstrap.discover_tools_from_clients",
        fake_discover_tools,
    )

    registry = ToolRegistry()

    await register_mcp_tools(
        registry=registry,
        clients=[
            FakeClient(),
        ],
    )

    assert registry.get(
        "product_search",
    ).name == "product_search"