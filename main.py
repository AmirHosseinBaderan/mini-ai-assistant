from cli.chat.cli import ChatCLI

from application.bootstrap import create_assistant

from dotenv import find_dotenv, load_dotenv


load_dotenv(
    find_dotenv(),
    verbose=True,
)


def main():

    assistant = create_assistant()

    cli = ChatCLI(
        engine=assistant,
    )

    cli.run()


if __name__ == "__main__":
    main()