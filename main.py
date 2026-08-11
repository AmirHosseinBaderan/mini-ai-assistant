from dotenv import find_dotenv, load_dotenv

from application.agent.agent import Agent
from application.assistant.engine import AssistantEngine
from application.chat.history import ConversationHistory
from application.llm.ollama_client import OllamaClient
from application.rag.chunker import Chunker
from application.rag.indexer import Indexer
from application.rag.knowledge_base import KnowledgeBase
from application.rag.loader import DocumentLoader
from application.rag.ollama_embedding import OllamaEmbeddingProvider
from application.rag.qdrant_vector_store import QdrantVectorStore
from application.rag.retriever import Retriever
from application.tools.knowledge_search import KnowledgeSearchTool
from application.tools.registry import ToolRegistry

from cli.chat.cli import ChatCLI
from cli.knowledge.cli import KnowledgeCLI


load_dotenv(
    find_dotenv(),
    verbose=True,
)


def build_assistant() -> AssistantEngine:

    llm_client = OllamaClient()

    embedding_provider = OllamaEmbeddingProvider(
        client=llm_client,
    )

    vector_store = QdrantVectorStore(
        collection_name="mini_assistant",
        vector_size=4096,
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    knowledge_search = KnowledgeSearchTool(
        retriever=retriever,
    )

    tool_registry = ToolRegistry()

    tool_registry.register(
        knowledge_search,
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    history = ConversationHistory()

    return AssistantEngine(
        agent=agent,
        history=history,
    )


def build_knowledge_base() -> KnowledgeBase:

    llm_client = OllamaClient()

    embedding_provider = OllamaEmbeddingProvider(
        client=llm_client,
    )

    vector_store = QdrantVectorStore(
        collection_name="mini_knowledge",
        vector_size=4096,
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


def main() -> None:

    assistant = build_assistant()
    knowledge_base = build_knowledge_base()

    chat_cli = ChatCLI(
        engine=assistant,
    )

    knowledge_cli = KnowledgeCLI(
        knowledge_base=knowledge_base,
    )

    while True:
        show_menu()

        choice = input("Select> ").strip()

        if choice in {"3", "exit", "quit"}:
            print("Goodbye!")
            break

        if choice == "1" or choice.lower() == "chat":
            chat_cli.run()
            continue

        if choice == "2" or choice.lower() == "knowledge":
            knowledge_cli.run()
            continue

        print("Unknown option. Use 1, 2, or 3.")


if __name__ == "__main__":
    main()
