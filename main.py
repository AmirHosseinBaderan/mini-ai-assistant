from application.agent.agent import Agent
from application.assistant.engine import AssistantEngine
from application.chat.history import ConversationHistory
from application.llm.ollama_client import OllamaClient
from application.rag.ollama_embedding import (
    OllamaEmbeddingProvider,
)
from application.rag.qdrant_vector_store import (
    QdrantVectorStore,
)
from application.rag.retriever import Retriever
from application.tools.knowledge_search import (
    KnowledgeSearchTool,
)
from application.tools.registry import ToolRegistry
from cli.chat.cli import ChatCLI
from dotenv import find_dotenv, load_dotenv


load_dotenv(
    find_dotenv(),
    verbose=True,
)


def main():

    llm_client = OllamaClient()

    embedding_provider = OllamaEmbeddingProvider(
        client=llm_client,
    )

    vector_store = QdrantVectorStore(
        collection_name="mini_chat_collection",
        vector_size=4096,
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    knowledge_tool = KnowledgeSearchTool(
        retriever=retriever,
    )

    tool_registry = ToolRegistry()

    tool_registry.register(
        knowledge_tool,
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    assistant_engine = AssistantEngine(
        agent=agent,
        history=ConversationHistory(),
    )

    cli = ChatCLI(
        engine=assistant_engine,
    )

    cli.run()


if __name__ == "__main__":
    main()