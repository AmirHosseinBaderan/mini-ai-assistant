from dotenv import load_dotenv

from application.llm.ollama_client import OllamaClient


load_dotenv()


def main():
    client = OllamaClient()

    messages = [
        {
            "role": "user",
            "content": "Hello! Introduce yourself in one sentence.",
        }
    ]

    for token in client.stream(messages):
        print(token, end="", flush=True)

    print()


if __name__ == "__main__":
    main()