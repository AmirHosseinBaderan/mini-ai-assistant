class ChatCLI:

    def __init__(self, engine):
        self.engine = engine

    def run(self):
        while True:
            query = input("You: ").strip()

            if query.lower() in {"exit", "quit", "/exit"}:
                break

            if not query:
                continue

            print("Assistant: ", end="", flush=True)

            for token in self.engine.stream(query):
                print(
                    token,
                    end="",
                    flush=True,
                )

            print()