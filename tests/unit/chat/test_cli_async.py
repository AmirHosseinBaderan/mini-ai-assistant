import pytest

from cli.chat.cli import ChatCLI


class FakeAgent:

    def __init__(self):
        self.on_tool_call = None


class FakeEngine:

    def __init__(self):
        self.agent = FakeAgent()

    async def astream(self, text):
        yield "Hello"
        yield " Amir"


@pytest.mark.anyio
async def test_chat_cli_uses_async_stream(
    monkeypatch,
    capsys,
):
    engine = FakeEngine()

    cli = ChatCLI(
        engine=engine,
    )

    inputs = iter(
        [
            "hello",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    await cli.arun()

    output = capsys.readouterr().out

    assert "Assistant:" in output
    assert "Hello Amir" in output