from application.bootstrap import (
    create_rag_components,
)

from cli.chat.cli import ChatCLI
from cli.knowledge.cli import KnowledgeCLI


def main():

    components = create_rag_components()

    chat_cli = ChatCLI(
        engine=components["rag_engine"]
    )

    knowledge_cli = KnowledgeCLI(
        knowledge_base=components[
            "knowledge_base"
        ]
    )

    while True:

        command = input("\n> ").strip()

        if command in {
            "/exit",
            "/quit",
            "exit",
            "quit",
        }:
            break

        if command == "/chat":
            chat_cli.run()
            continue

        if command == "/knowledge":
            knowledge_cli.run()
            continue

        print(
            "Commands:\n"
            "  /chat\n"
            "  /knowledge\n"
            "  /exit"
        )


if __name__ == "__main__":
    main()