from application.rag.chunk import Chunk
from application.tools.knowledge_search import KnowledgeSearchTool

class FakeRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):
        return [
            (
                Chunk(
                    content="Python is a programming language.",
                    metadata={
                        "filename": "python.txt",
                    },
                ),
                0.95,
            )
        ]

def test_knowledge_search_tool():

    tool = KnowledgeSearchTool(
        retriever=FakeRetriever(),
    )

    assert tool.name == "knowledge_search"

    result = tool.execute(
        query="What is Python?",
    )

    assert result == [
        {
            "content": "Python is a programming language.",
            "score": 0.95,
            "metadata": {
                "filename": "python.txt",
            },
        }
    ]

def test_knowledge_search_tool_empty_query():
    tool = KnowledgeSearchTool(
        retriever=FakeRetriever(),
    )

    try:
        tool.execute(query="")
        assert False
    except ValueError as error:
        assert str(error) == "query is required"