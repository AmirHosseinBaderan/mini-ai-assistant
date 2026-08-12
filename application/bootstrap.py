from application.agent.agent import Agent
from application.assistant.engine import AssistantEngine
from application.llm.ollama_client import OllamaClient
from application.rag.ollama_embedding import OllamaEmbeddingProvider
from application.rag.qdrant_vector_store import QdrantVectorStore
from application.rag.retriever import Retriever
from application.tools.knowledge_search import KnowledgeSearchTool
from application.tools.load_memory import LoadMemoryTool
from application.tools.save_memory import SaveMemoryTool
from application.tools.registry import ToolRegistry


def create_assistant() -> AssistantEngine:

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

    knowledge_search_tool = KnowledgeSearchTool(
        retriever=retriever,
    )

    save_memory_tool = SaveMemoryTool()
    load_memory_tool = LoadMemoryTool()

    tool_registry = ToolRegistry()

    tool_registry.register(
        knowledge_search_tool,
    )

    tool_registry.register(
        save_memory_tool,
    )

    tool_registry.register(
        load_memory_tool,
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    return AssistantEngine(
        agent=agent,
    )