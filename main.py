from application.assistant.engine import AssistantEngine
from application.chat.engine import ChatEngine
from application.llm.ollama_client import OllamaClient

from cli.chat.cli import ChatCLI

from dotenv import find_dotenv, load_dotenv

load_dotenv(
    find_dotenv(),
    verbose=True,
)

def main():
    llm_client = OllamaClient()


    chat_engine = ChatEngine(
        llm=llm_client,
    )

    assistant_engine = AssistantEngine(
        chat_engine=chat_engine,
    )

    cli = ChatCLI(
        engine=assistant_engine,
    )

    cli.run()

if __name__ == "main":
    main()
