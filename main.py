from dotenv import load_dotenv

from application.chat.engine import ChatEngine
from application.llm.ollama_client import OllamaClient


def main():
    load_dotenv()

    llm = OllamaClient()
    chat = ChatEngine(llm)

    print("Mini AI Assistant")
    print("Type 'exit' to quit.")
    print()

    while True:
        user_message = input("You: ")

        if user_message.lower() == "exit":
            break

        print("Assistant: ", end="")

        chat.chat(user_message)


if __name__ == "__main__":
    main()