from application.bootstrap import create_rag_engine


def main():

    rag_engine = create_rag_engine()

    while True:

        query = input("\nYou: ")

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        print("\nAssistant: ", end="")

        for token in rag_engine.stream(
            query
        ):
            print(
                token,
                end="",
                flush=True,
            )

        print()


if __name__ == "__main__":
    main()