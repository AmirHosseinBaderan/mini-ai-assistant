from pathlib import Path


class KnowledgeCLI:

    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def run(self):
        while True:
            command = input("Knowledge> ").strip()

            if command in {"/back", "/exit", "exit"}:
                break

            if command.startswith("/add "):
                self._add_file(
                    command[5:].strip()
                )
                continue

            if command.startswith("/add-dir "):
                self._add_directory(
                    command[9:].strip()
                )
                continue

            if command == "/help":
                self._help()
                continue

            print(
                "Unknown command. "
                "Use /help."
            )

    def _add_file(self, value: str):
        path = Path(value).expanduser()

        try:
            count = self.knowledge_base.add_file(
                path
            )

            print(
                f"✓ Indexed {count} chunks"
            )

        except Exception as exc:
            print(
                f"✗ Failed: {exc}"
            )

    def _add_directory(self, value: str):
        path = Path(value).expanduser()

        try:
            count = (
                self.knowledge_base.add_directory(
                    path
                )
            )

            print(
                f"✓ Indexed {count} chunks"
            )

        except Exception as exc:
            print(
                f"✗ Failed: {exc}"
            )

    @staticmethod
    def _help():
        print(
            "\n"
            "/add <file>\n"
            "/add-dir <directory>\n"
            "/help\n"
            "/back\n"
        )