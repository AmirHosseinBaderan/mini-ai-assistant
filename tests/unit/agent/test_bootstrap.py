import pytest

from application.agent.bootstrap import create_agent


class FakeLLMClient:
    pass


class FakeMCPClient:
    pass


@pytest.mark.anyio
async def test_create_agent(
    monkeypatch,
):

    registered = {}

    async def fake_register_mcp_tools(
        registry,
        clients,
    ):

        registered["registry"] = registry
        registered["clients"] = clients

    monkeypatch.setattr(
        "application.agent.bootstrap.register_mcp_tools",
        fake_register_mcp_tools,
    )

    llm_client = FakeLLMClient()

    mcp_client = FakeMCPClient()

    agent = await create_agent(
        llm_client=llm_client,
        mcp_clients=[mcp_client],
    )

    assert agent.llm_client is llm_client
    assert registered["clients"] == [mcp_client]
    assert agent.tool_registry is registered["registry"]