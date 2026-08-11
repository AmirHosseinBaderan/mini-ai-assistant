from application.tools.base import Tool

class ToolRegistry:

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        tool: Tool,
    ) -> None:

        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool

    def get(
        self,
        name: str,
    ) -> Tool:

        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(
                f"Tool not found: {name}"
            )

    def list(self) -> list[Tool]:
        return list(
            self._tools.values()
        )