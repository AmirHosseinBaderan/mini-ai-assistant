from dotenv import find_dotenv, load_dotenv
from application.agent.agent import Agent
from application.assistant.engine import AssistantEngine
from application.chat.history import ConversationHistory
from application.llm.ollama_client import OllamaClient
from application.rag.ollama_embedding import OllamaEmbeddingProvider
from application.rag.qdrant_vector_store import QdrantVectorStore
from application.rag.retriever import Retriever
from application.tools.knowledge_search import KnowledgeSearchTool
from application.tools.registry import ToolRegistry

load_dotenv(
    find_dotenv(),
    verbose=True,
)

def test_real_assistant_conversation():

    llm_client = OllamaClient()

    tool_registry = ToolRegistry()

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    history = ConversationHistory()

    assistant = AssistantEngine(
        agent=agent,
        history=history,
    )

    first_response = "".join(
        assistant.stream(
            "My name is Amir Hossein."
        )
    )

    assert first_response.strip()

    second_response = "".join(
        assistant.stream(
            "What is my name?"
        )
    )

    assert second_response.strip()

    print(
        "\nFIRST:",
        first_response,
    )

    print(
        "SECOND:",
        second_response,
    )


def test_real_assistant_with_knowledge_search():

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

    assistant = AssistantEngine(
        agent=agent,
        history=ConversationHistory(),
    )

    result = "".join(
        assistant.stream(
            "Use the knowledge search tool "
            "to find information about Python."
        )
    )

    print("\nANSWER:")
    print(result)

    assert result.strip()