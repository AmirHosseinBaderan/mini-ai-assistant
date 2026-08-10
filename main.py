from application.assistant.engine import AssistantEngine
from application.chat.engine import ChatEngine
from application.llm.ollama_client import OllamaClient
from application.rag.engine import RAGEngine
from application.rag.context_builder import ContextBuilder
from application.rag.ollama_embedding import OllamaEmbeddingProvider
from application.rag.qdrant_vector_store import QdrantVectorStore
from application.rag.retriever import Retriever
from application.router.router import Router
from intent_classifier.config import IntentConfig

from cli.chat.cli import ChatCLI

from intent_classifier.predictor import IntentPredictor
from intent_classifier.tokenizer import IntentTokenizer

from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv(),verbose=True)

def main():
    # LLM
    llm_client = OllamaClient()
    embedding_provider = OllamaEmbeddingProvider(
        client=llm_client,
    )

    # Intent
    tokenizer = IntentTokenizer()
    tokenizer.build_vocab_from_file(IntentConfig.TRAIN_PATH)

    config = IntentConfig()
    predictor = IntentPredictor(
        tokenizer=tokenizer,
        config=config,
        checkpoint_dir=IntentConfig.CHECKPOINT_DIR
    )

    router = Router(
        predictor=predictor,
        confidence_threshold=0.80,
    )

    # Chat
    chat_engine = ChatEngine(
        llm=llm_client,
    )

    # RAG
    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=QdrantVectorStore(
            collection_name="mini_chat_collection",
            vector_size=4096
        )
    )

    context_builder = ContextBuilder()

    rag_engine = RAGEngine(
        retriever=retriever,
        context_builder=context_builder,
        llm_client=llm_client,
    )

    # Assistant
    assistant_engine = AssistantEngine(
        router=router,
        chat_engine=chat_engine,
        rag_engine=rag_engine,
    )

    # CLI
    cli = ChatCLI(
        engine=assistant_engine,
    )

    cli.run()


if __name__ == "__main__":
    main()
