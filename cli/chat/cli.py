class ChatCLI:

    def __init__(self, engine):
        self.engine = engine
        self._first_token = True
        self._status_shown = False

    def _on_tool_call(self, tool_name):
        self._status_shown = True
        print(
            f"\r\033[K\033[33mCalling {tool_name}...\033[0m",
            end="",
            flush=True,
        )

    def run(self):
        while True:
            query = input("You: ").strip()

            if query.lower() in {"exit", "quit", "/exit"}:
                break

            if not query:
                continue

            self.engine.agent.on_tool_call = self._on_tool_call
            self._first_token = True
            self._status_shown = False

            print("Assistant: ", end="", flush=True)

            for token in self.engine.stream(query):

                if self._first_token:

                    if self._status_shown:
                        print(
                            f"\r\033[KAssistant: {token}",
                            end="",
                            flush=True,
                        )
                        self._status_shown = False
                    else:
                        print(
                            token,
                            end="",
                            flush=True,
                        )

                    self._first_token = False

                    continue

                print(
                    token,
                    end="",
                    flush=True,
                )

            print()
