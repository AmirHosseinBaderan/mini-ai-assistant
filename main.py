
from application.llm.ollama_client import OllamaClient
from application.rag.chunker import Chunker
from application.rag.indexer import Indexer
from application.rag.knowledge_base import KnowledgeBase
from application.rag.loader import DocumentLoader
from application.rag.ollama_embedding import OllamaEmbeddingProvider
from application.rag.qdrant_vector_store import QdrantVectorStore
from application.bootstrap import create_assistant
import asyncio
from cli.chat.cli import ChatCLI
from cli.knowledge.cli import KnowledgeCLI
import asyncio

from application.bootstrap import create_assistant
from application.llm.ollama_client import OllamaClient
from application.mcp.client.client import MCPClient
from application.mcp.server.bootstrap import create_server

from dotenv import find_dotenv, load_dotenv

load_dotenv(
    find_dotenv(),
    verbose=True,
)

KNOWLEDGE_COLLECTION = "mini_knowledge"
VECTOR_SIZE = 4096


def build_knowledge_base(
        llm_client: OllamaClient,
) -> KnowledgeBase:
    embedding_provider = OllamaEmbeddingProvider(
        client=llm_client,
    )

    vector_store = QdrantVectorStore(
        collection_name=KNOWLEDGE_COLLECTION,
        vector_size=VECTOR_SIZE,
    )

    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    loader = DocumentLoader()
    chunker = Chunker()

    return KnowledgeBase(
        loader=loader,
        chunker=chunker,
        indexer=indexer,
    )


def show_menu() -> None:
    print()
    print("=" * 40)
    print("  Mini AI Assistant")
    print("=" * 40)
    print("  [1] Chat")
    print("  [2] Knowledge")
    print("  [3] Exit")
    print("=" * 40)


async def main():

    llm_client = OllamaClient()

    server = create_server()

    async with MCPClient(server) as mcp_client:

        assistant = await create_assistant(
            llm_client=llm_client,
            mcp_client=mcp_client,
        )

        knowledge_base = build_knowledge_base(
            llm_client=llm_client,
        )

        chat_cli = ChatCLI(
            engine=assistant,
        )

        knowledge_cli = KnowledgeCLI(
            knowledge_base=knowledge_base,
        )

        while True:

            show_menu()

            choice = input(
                "Select> "
            ).strip()

            if choice in {
                "3",
                "exit",
                "quit",
            }:
                print("Goodbye!")
                break

            if choice == "1" or choice.lower() == "chat":

                await chat_cli.arun()

                continue

            if choice == "2" or choice.lower() == "knowledge":

                knowledge_cli.run()

                continue

            print(
                "Unknown option. Use 1, 2, or 3."
            )


if __name__ == "__main__":
    asyncio.run(main())