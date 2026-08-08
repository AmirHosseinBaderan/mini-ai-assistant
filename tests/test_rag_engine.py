from application.rag.engine import RAGEngine


class FakeRetriever:

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ):
        return [
            (
                FakeChunk(
                    content="Python is a programming language.",
                    metadata={
                        "filename": "python.txt",
                    },
                ),
                0.95,
            )
        ]


class FakeChunk:

    def __init__(
        self,
        content,
        metadata,
    ):
        self.content = content
        self.metadata = metadata


class FakeContextBuilder:

    def build(self, chunks):

        if not chunks:
            return ""

        return (
            "[Source: python.txt]\n"
            "Python is a programming language."
        )


class FakeLLMClient:

    def __init__(self):
        self.messages = None

    def stream(self, messages):

        self.messages = messages

        yield "Python"
        yield " is"
        yield " a"
        yield " programming"
        yield " language."

def test_rag_engine():

    llm = FakeLLMClient()

    engine = RAGEngine(
        retriever=FakeRetriever(),
        context_builder=FakeContextBuilder(),
        llm_client=llm,
    )

    result = "".join(
        engine.stream(
            "What is Python?"
        )
    )

    assert result == (
        "Python is a programming language."
    )

    assert len(llm.messages) == 2

    assert (
        llm.messages[0]["role"]
        == "system"
    )

    assert (
        "Python is a programming language."
        in llm.messages[1]["content"]
    )

class EmptyRetriever:

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ):
        return []

def test_rag_engine_without_results():

    llm = FakeLLMClient()

    engine = RAGEngine(
        retriever=EmptyRetriever(),
        context_builder=FakeContextBuilder(),
        llm_client=llm,
    )

    result = "".join(
        engine.stream(
            "What is quantum computing?"
        )
    )

    assert result == (
        "Python is a programming language."
    )

    assert (
        llm.messages[1]["content"]
        == (
            "Context:\n\n"
            "Question:\n"
            "What is quantum computing?"
        )
    )