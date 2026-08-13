# Mini AI Assistant - Project Architecture Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Philosophy](#architecture-philosophy)
3. [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
   - [1. Presentation Layer (CLI)](#1-presentation-layer-cli)
   - [2. Application Layer](#2-application-layer)
   - [3. Domain Layer](#3-domain-layer)
   - [4. Infrastructure Layer](#4-infrastructure-layer)
   - [5. ML/AI Layer](#5-mlai-layer)
4. [How Layers Work Together](#how-layers-work-together)
5. [Why Layers Exist](#why-layers-exist)
6. [Key Design Patterns](#key-design-patterns)
7. [Data Flow Examples](#data-flow-examples)
8. [Teaching Notes](#teaching-notes)

---

## Project Overview

**Mini AI Assistant** is a modular, production-ready AI assistant framework built in Python. It combines multiple AI capabilities into a unified system:

- **Conversational AI** with streaming responses via Ollama (local LLM)
- **Retrieval-Augmented Generation (RAG)** for knowledge-based Q&A
- **Intent Classification** using a custom transformer model for message routing
- **Tool Calling** with a registry-based tool system
- **MCP (Model Context Protocol)** integration for external tool servers
- **Product Search** across multiple e-commerce sites
- **Memory Management** for persistent conversation context

The project demonstrates clean architecture principles, dependency injection, abstract base classes, and comprehensive testing (unit + integration).

---

## Architecture Philosophy

The project follows a **Layered Architecture** pattern with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│                   (CLI - Chat Interface)                     │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Assistant   │  │    Agent    │  │  Router (Intent)    │  │
│  │   Engine     │  │   Engine    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    Tools    │  │    Chat     │  │  Product Search     │  │
│  │  (Abstract) │  │  History    │  │  (Domain Models)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Ollama     │  │  Qdrant     │  │  HTTP Fetchers      │  │
│  │  Client     │  │  Vector DB  │  │  (httpx)            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      ML/AI Layer                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Intent Classifier (Transformer)                     │    │
│  │  Embeddings (Ollama)                                 │    │
│  │  RAG Pipeline (Chunk → Embed → Store → Retrieve)     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer-by-Layer Breakdown

### 1. Presentation Layer (CLI)

**Location:** `cli/chat/cli.py`

**What it does:**
- Provides the user interface for interacting with the assistant
- Handles user input (text queries) and displays streaming responses
- Manages terminal UI with status indicators (e.g., "Calling tool_name...")
- Supports both synchronous (`run()`) and asynchronous (`arun()`) modes

**How it works:**
```python
class ChatCLI:
    def __init__(self, engine):
        self.engine = engine  # Injected dependency
    
    def run(self):
        while True:
            query = input("You: ").strip()
            # Stream tokens from the engine
            for token in self.engine.stream(query):
                print(token, end="", flush=True)
```

**Why it exists:**
- Separates user interaction from business logic
- Makes the core engine testable without a terminal
- Allows multiple UI implementations (CLI, web, API) to use the same engine

**Key files:**
- [`cli/chat/cli.py`](cli/chat/cli.py) - Main chat CLI
- [`main.py`](main.py) - Entry point with menu system

---

### 2. Application Layer

This is the **orchestration layer** that coordinates domain services and infrastructure.

#### 2.1 Assistant Engine

**Location:** `application/assistant/engine.py`

**What it does:**
- Orchestrates the Agent, Router, and Conversation History
- Provides the main `stream()` and `astream()` methods used by the CLI
- Manages the complete conversation lifecycle

**How it works:**
```python
class AssistantEngine:
    def __init__(self, agent, router, history):
        self.agent = agent
        self.router = router
        self.history = history
    
    def stream(self, query: str):
        # Route the query if needed
        # Delegate to agent for processing
        # Manage conversation history
        pass
```

**Why it exists:**
- Acts as the **composition root** for the application
- Provides a single entry point for all assistant operations
- Encapsulates complex orchestration logic

#### 2.2 Agent Engine

**Location:** `application/agent/agent.py`

**What it does:**
- Implements the **ReAct (Reason + Act) pattern** for tool-calling
- Manages the conversation loop: LLM → Tool Call → Result → LLM → Response
- Supports streaming responses with tool execution
- Handles both sync and async execution paths

**How it works:**
```python
class Agent:
    def __init__(self, llm_client, tool_registry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
    
    def stream(self, query: str):
        # 1. Add user message to history
        # 2. Call LLM with available tools
        # 3. If LLM requests tool call:
        #    a. Execute tool
        #    b. Add result to history
        #    c. Call LLM again
        # 4. Stream final response
        pass
```

**Why it exists:**
- Encapsulates the AI reasoning loop
- Makes tool calling transparent to upper layers
- Provides streaming for better UX

#### 2.3 Router (Intent Classification)

**Location:** `application/router/router.py`

**What it does:**
- Classifies user intent using the ML model
- Routes messages to appropriate handlers based on confidence
- Falls back to general chat if confidence is low

**How it works:**
```python
class Router:
    def __init__(self, predictor, threshold=0.7):
        self.predictor = predictor
        self.threshold = threshold
    
    def route(self, message: str):
        label, confidence = self.predictor.predict(message)
        if confidence >= self.threshold:
            return self.handlers[label]
        return self.handlers["default"]
```

**Why it exists:**
- Enables intent-based routing without hardcoding logic
- Makes the system extensible for new intents
- Provides confidence-based fallback

---

### 3. Domain Layer

Contains business logic and domain models, independent of frameworks.

#### 3.1 Tools System

**Location:** `application/tools/`

**What it does:**
- Defines the `Tool` abstract base class
- Provides a registry for managing available tools
- Implements concrete tools: Knowledge Search, Memory (Save/Load/Update), MCP Tools

**How it works:**
```python
class Tool(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def description(self) -> str: ...
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult: ...

class ToolRegistry:
    def register(self, tool: Tool): ...
    def get(self, name: str) -> Tool: ...
    def all(self) -> list[Tool]: ...
```

**Why it exists:**
- Provides a plugin-like architecture for capabilities
- Makes the agent extensible without modifying core logic
- Enables dynamic tool discovery (especially from MCP servers)

**Key files:**
- [`application/tools/base.py`](application/tools/base.py) - Abstract Tool base
- [`application/tools/registry.py`](application/tools/registry.py) - Tool registry
- [`application/tools/knowledge_search.py`](application/tools/knowledge_search.py) - RAG search tool
- [`application/tools/save_memory.py`](application/tools/save_memory.py) - Memory persistence
- [`application/tools/load_memory.py`](application/tools/load_memory.py) - Memory retrieval
- [`application/tools/update_memory.py`](application/tools/update_memory.py) - Memory updates

#### 3.2 Chat History

**Location:** `application/chat/history.py`

**What it does:**
- Manages conversation message history
- Provides methods to add, retrieve, and clear messages
- Supports both in-memory and persistent storage

**Why it exists:**
- Maintains context for multi-turn conversations
- Enables the LLM to reference previous messages
- Abstracts storage mechanism from the agent

#### 3.3 Product Search Domain

**Location:** `application/product_search/`

**What it does:**
- Defines domain models (`Product`, `SearchResult`)
- Implements search engine for multi-site product search
- Uses parser registry for site-specific HTML parsing

**Key components:**
- [`application/product_search/models.py`](application/product_search/models.py) - Domain models
- [`application/product_search/engine.py`](application/product_search/engine.py) - Search orchestration
- [`application/product_search/service.py`](application/product_search/service.py) - Business logic
- [`application/product_search/fetcher.py`](application/product_search/fetcher.py) - HTTP fetching

---

### 4. Infrastructure Layer

Provides technical capabilities and external integrations.

#### 4.1 LLM Client

**Location:** `application/llm/`

**What it does:**
- Abstracts LLM provider interactions
- Implements Ollama client for local LLM inference
- Supports streaming, tool calling, and chat completion

**How it works:**
```python
class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages, tools) -> LLMResponse: ...
    @abstractmethod
    def stream(self, messages) -> Iterator[str]: ...
    @abstractmethod
    def stream_chat(self, messages, tools) -> Iterator[LLMStreamEvent]: ...

class OllamaClient(LLMClient):
    def stream_chat(self, messages, tools):
        # Stream tokens from Ollama API
        # Yield LLMStreamEvent for text, tool_calls, etc.
        pass
```

**Why it exists:**
- Decouples the application from specific LLM providers
- Enables swapping Ollama for OpenAI, Anthropic, etc.
- Provides consistent interface for streaming and tool calling

**Key files:**
- [`application/llm/client.py`](application/llm/client.py) - Abstract base
- [`application/llm/ollama_client.py`](application/llm/ollama_client.py) - Ollama implementation
- [`application/llm/message.py`](application/llm/message.py) - Message types
- [`application/llm/response.py`](application/llm/response.py) - Response types
- [`application/llm/stream_event.py`](application/llm/stream_event.py) - Streaming events

#### 4.2 RAG (Retrieval-Augmented Generation)

**Location:** `application/rag/`

**What it does:**
- Implements the complete RAG pipeline
- Chunks documents, creates embeddings, stores in vector DB
- Retrieves relevant context for user queries

**Pipeline:**
```
Document → Chunking → Embedding → Vector Store (Qdrant)
                                              ↑
User Query → Embedding → Similarity Search → Context → LLM
```

**Key components:**
- [`application/rag/engine.py`](application/rag/engine.py) - RAG orchestration
- [`application/rag/retriever.py`](application/rag/retriever.py) - Similarity search
- [`application/rag/indexer.py`](application/rag/indexer.py) - Document indexing
- [`application/rag/chunker.py`](application/rag/chunker.py) - Text chunking
- [`application/rag/qdrant_vector_store.py`](application/rag/qdrant_vector_store.py) - Qdrant integration
- [`application/rag/ollama_embedding.py`](application/rag/ollama_embedding.py) - Embedding generation

**Why it exists:**
- Enables knowledge-based Q&A without fine-tuning
- Keeps domain knowledge separate from model weights
- Allows dynamic knowledge base updates

#### 4.3 MCP (Model Context Protocol)

**Location:** `application/mcp/`

**What it does:**
- Implements MCP client/server for external tool integration
- Allows the assistant to use tools from external MCP servers
- Discovers and registers tools dynamically

**Architecture:**
```
┌─────────────────┐         ┌─────────────────┐
│   Mini AI       │  MCP    │   External      │
│   Assistant     │◄───────►│   MCP Server    │
│   (Client)      │ Protocol│   (Tools)       │
└─────────────────┘         └─────────────────┘
```

**Key files:**
- [`application/mcp/client/client.py`](application/mcp/client/client.py) - MCP client wrapper
- [`application/mcp/server/server.py`](application/mcp/server/server.py) - MCP server setup
- [`application/mcp/tools.py`](application/mcp/tools.py) - Tool discovery
- [`application/mcp/tool.py`](application/mcp/tool.py) - MCP tool wrapper
- [`application/tools/mcp_tool.py`](application/tools/mcp_tool.py) - Adapter to internal Tool interface

**Why it exists:**
- Enables extensibility without code changes
- Allows integration with external services
- Follows the Model Context Protocol standard

#### 4.4 Product Search Infrastructure

**Location:** `application/product_search/parsers/`

**What it does:**
- Implements site-specific HTML parsers
- Uses registry pattern for multi-site support
- Fetches and parses product data from e-commerce sites

**Key files:**
- [`application/product_search/parsers/base.py`](application/product_search/parsers/base.py) - Abstract parser
- [`application/product_search/parsers/torob.py`](application/product_search/parsers/torob.py) - Torob.ir parser
- [`application/product_search/parsers/registry.py`](application/product_search/parsers/registry.py) - Parser registry

---

### 5. ML/AI Layer

Contains machine learning models and training infrastructure.

#### 5.1 Intent Classifier

**Location:** `intent_classifier/`

**What it does:**
- Custom transformer-based text classification model
- Classifies user messages into intents (e.g., "search_product", "ask_knowledge", "chat")
- Trained on custom dataset with attention mechanism

**Architecture:**
```
Input Text → Tokenizer → Embedding → Positional Encoding
    → Transformer Encoder (Multi-Head Attention)
    → Mean Pooling → Classifier → Intent Label
```

**Key components:**
- [`intent_classifier/model.py`](intent_classifier/model.py) - Model architecture
- [`intent_classifier/tokenizer.py`](intent_classifier/tokenizer.py) - Text tokenization
- [`intent_classifier/trainer.py`](intent_classifier/trainer.py) - Training loop
- [`intent_classifier/predictor.py`](intent_classifier/predictor.py) - Inference wrapper
- [`intent_classifier/encoder.py`](intent_classifier/encoder.py) - Transformer encoder
- [`intent_classifier/attention.py`](intent_classifier/attention.py) - Multi-head attention

**Why it exists:**
- Enables intelligent routing without hardcoded rules
- Learns from data rather than manual pattern matching
- Provides confidence scores for fallback handling

#### 5.2 Training Pipeline

**Location:** `train.py`, `predict.py`

**What it does:**
- Trains the intent classifier on JSONL dataset
- Saves checkpoints for inference
- Provides prediction interface

**Why it exists:**
- Separates training from inference
- Enables model improvement without code changes
- Provides reproducible training pipeline

---

## How Layers Work Together

### Complete Request Flow

```
User Input (CLI)
    │
    ▼
┌─────────────────────────────────────────┐
│  Presentation Layer (ChatCLI)           │
│  - Reads input                          │
│  - Calls engine.stream(query)           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Application Layer (AssistantEngine)    │
│  - Routes query via Router              │
│  - Manages conversation history         │
│  - Delegates to Agent                   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Application Layer (Agent)              │
│  - Builds message list                  │
│  - Calls LLM with tools                 │
│  - Executes tool calls if needed        │
│  - Streams response                     │
└─────────────────────────────────────────┘
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Domain      │  │ Domain      │  │ Domain      │
│ (Tool       │  │ (Chat       │  │ (Product    │
│  Registry)  │  │  History)   │  │  Search)    │
└─────────────┘  └─────────────┘  └─────────────┘
    │                  │                  │
    ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Infra       │  │ Infra       │  │ Infra       │
│ (LLM        │  │ (In-Memory  │  │ (HTTP       │
│  Client)    │  │  Storage)   │  │  Fetcher)   │
└─────────────┘  └─────────────┘  └─────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ML/AI Layer                            │
│  - Intent Classification (if routing)   │
│  - Embeddings (if RAG)                  │
│  - LLM Inference (Ollama)               │
└─────────────────────────────────────────┘
    │
    ▼
Response Streamed Back to CLI
```

### Tool Call Flow

```
Agent calls LLM
    │
    ▼
LLM requests tool call: search_products(query="laptop")
    │
    ▼
Agent looks up tool in ToolRegistry
    │
    ▼
Tool.execute(query="laptop")
    │
    ├──────────────────┐
    ▼                  ▼
┌─────────────┐  ┌─────────────┐
│ Product     │  │ MCP Client  │
│ Search      │  │ (External)  │
│ Engine      │  │             │
└─────────────┘  └─────────────┘
    │                  │
    ▼                  ▼
┌─────────────┐  ┌─────────────┐
│ HTTP        │  │ MCP Server  │
│ Fetcher     │  │ (External)  │
└─────────────┘  └─────────────┘
    │
    ▼
ToolResult returned to Agent
    │
    ▼
Agent adds result to history
    │
    ▼
Agent calls LLM again with result
    │
    ▼
LLM generates final response
    │
    ▼
Response streamed to user
```

---

## Why Layers Exist

### 1. Separation of Concerns
Each layer has a single responsibility:
- **Presentation**: User interaction
- **Application**: Orchestration and workflow
- **Domain**: Business rules and logic
- **Infrastructure**: Technical capabilities
- **ML/AI**: Intelligence and learning

### 2. Testability
Layers can be tested independently:
```python
# Unit test: Test Agent without CLI
def test_agent_tool_calling():
    mock_llm = MockLLMClient()
    mock_registry = MockToolRegistry()
    agent = Agent(mock_llm, mock_registry)
    # Test agent behavior

# Integration test: Test full flow
def test_assistant_stream():
    engine = create_test_engine()
    cli = ChatCLI(engine)
    # Test complete flow
```

### 3. Maintainability
Changes in one layer don't cascade to others:
- Swap Ollama for OpenAI → Only change `OllamaClient`
- Add new tool → Only add new `Tool` implementation
- Change CLI to web → Only change presentation layer

### 4. Extensibility
New features can be added without modifying existing code:
- New intent → Add to classifier training data
- New tool → Implement `Tool` interface
- New MCP server → Add client configuration
- New parser → Implement `ParserBase`

### 5. Dependency Direction
Dependencies flow inward (outer layers depend on inner):
```
CLI → AssistantEngine → Agent → LLMClient
                    ↘         ↗
                     ToolRegistry
```

Inner layers don't know about outer layers.

---

## Key Design Patterns

### 1. Dependency Injection
All dependencies are injected via constructors:
```python
class Agent:
    def __init__(self, llm_client, tool_registry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
```

**Benefit:** Easy testing, flexible configuration, clear dependencies.

### 2. Abstract Base Classes (ABC)
Multiple ABCs define contracts:
- `LLMClient` - LLM provider interface
- `Tool` - Tool interface
- `ParserBase` - HTML parser interface
- `VectorStore` - Vector database interface

**Benefit:** Multiple implementations, clear contracts, polymorphism.

### 3. Registry Pattern
Used for tools and parsers:
```python
registry = ToolRegistry()
registry.register(KnowledgeSearchTool())
registry.register(SaveMemoryTool())
```

**Benefit:** Dynamic discovery, plugin architecture, decoupled registration.

### 4. Factory/Composition Root
Bootstrap functions create configured instances:
```python
async def create_agent(llm_client, mcp_clients):
    registry = ToolRegistry()
    await register_mcp_tools(registry, mcp_clients)
    return Agent(llm_client, registry)
```

**Benefit:** Centralized configuration, easy testing with mocks.

### 5. Strategy Pattern
Different strategies for different intents:
```python
router = Router(predictor, handlers={
    "search_product": handle_product_search,
    "ask_knowledge": handle_knowledge_search,
    "chat": handle_general_chat,
})
```

**Benefit:** Swappable behaviors, open for extension.

### 6. Adapter Pattern
MCP tools adapted to internal Tool interface:
```python
class MCPTool(Tool):
    def __init__(self, mcp_client, tool_name):
        self.mcp_client = mcp_client
        self.tool_name = tool_name
    
    def execute(self, **kwargs):
        return self.mcp_client.call_tool(self.tool_name, kwargs)
```

**Benefit:** Integration without tight coupling.

---

## Data Flow Examples

### Example 1: Simple Chat

```
User: "Hello, how are you?"
    │
    ▼
ChatCLI.run() → engine.stream("Hello, how are you?")
    │
    ▼
AssistantEngine.stream()
    │
    ▼
Agent.stream()
    │
    ▼
LLMClient.stream_chat(messages, tools=[])
    │
    ▼
Ollama API → "I'm doing well, thank you!"
    │
    ▼
Tokens streamed back to CLI
    │
    ▼
User sees: "I'm doing well, thank you!"
```

### Example 2: Tool Calling (Product Search)

```
User: "Find me a laptop under 20 million"
    │
    ▼
Router.classify("Find me a laptop under 20 million")
    │
    ▼
IntentClassifier → "search_product" (confidence: 0.92)
    │
    ▼
Agent.stream() with routing
    │
    ▼
LLM requests tool call: search_products(query="laptop", max_price=20000000)
    │
    ▼
ToolRegistry.get("search_products")
    │
    ▼
ProductSearchTool.execute(query="laptop", max_price=20000000)
    │
    ▼
ProductSearchEngine.search(sites, query, max_price)
    │
    ├─► HttpxFetcher.fetch("https://torob.com/search?query=laptop")
    │       │
    │       ▼
    │   TorobParser.parse(html) → [Product(...), ...]
    │
    ▼
ToolResult(products=[...])
    │
    ▼
Agent adds result to history, calls LLM again
    │
    ▼
LLM generates: "I found 3 laptops under 20 million..."
    │
    ▼
Response streamed to user
```

### Example 3: RAG (Knowledge Search)

```
User: "What is the return policy?"
    │
    ▼
Router.classify("What is the return policy?")
    │
    ▼
IntentClassifier → "ask_knowledge" (confidence: 0.88)
    │
    ▼
Agent.stream() with routing
    │
    ▼
LLM requests tool call: knowledge_search(query="return policy")
    │
    ▼
KnowledgeSearchTool.execute(query="return policy")
    │
    ▼
RAGEngine.query(query)
    │
    ├─► OllamaEmbedding.embed("return policy")
    │       │
    │       ▼
    │   [0.123, -0.456, ...] (embedding vector)
    │
    ├─► QdrantVectorStore.search(embedding, top_k=3)
    │       │
    │       ▼
    │   [Document("Returns accepted within 30 days"), ...]
    │
    ├─► ContextBuilder.build(documents, query)
    │       │
    │       ▼
    │   "Context: Returns accepted within 30 days...\nQuestion: What is the return policy?"
    │
    ▼
ToolResult(context="Returns accepted within 30 days...")
    │
    ▼
Agent adds result to history, calls LLM again
    │
    ▼
LLM generates: "According to our policy, returns are accepted within 30 days..."
    │
    ▼
Response streamed to user
```

---

## Teaching Notes

### For Students Learning This Architecture

#### 1. Why Layered Architecture?
Think of it like a restaurant:
- **Presentation Layer** = Waiter (takes orders, serves food)
- **Application Layer** = Manager (coordinates kitchen, waiters, customers)
- **Domain Layer** = Recipes and cooking techniques (business logic)
- **Infrastructure Layer** = Kitchen equipment (ovens, stoves)
- **ML/AI Layer** = The chef's expertise (intelligence)

Each layer can change without breaking the others. You can get a new waiter (CLI) without changing the recipes (domain).

#### 2. Why Abstract Base Classes?
ABCs are like contracts. When you define `class Tool(ABC)`, you're saying:
> "Anyone who wants to be a Tool MUST implement these methods"

This ensures consistency and enables polymorphism. You can treat all tools the same way, even though they do different things.

#### 3. Why Dependency Injection?
Imagine building a car. If you hardcode the engine into the chassis, you can't swap it out. But if you use bolts (dependency injection), you can swap engines easily.

In code:
```python
# Bad: Hardcoded dependency
class Agent:
    def __init__(self):
        self.llm = OllamaClient()  # Can't change!

# Good: Injected dependency
class Agent:
    def __init__(self, llm_client):
        self.llm = llm_client  # Can be anything!
```

#### 4. Why the Registry Pattern?
Imagine a library. Instead of memorizing where every book is, you use a catalog (registry). When you want a book, you look it up by name.

```python
registry = ToolRegistry()
registry.register(KnowledgeSearchTool())  # Add to catalog
tool = registry.get("knowledge_search")   # Look up by name
```

This makes the system extensible. You can add new tools without changing the agent code.

#### 5. Why Streaming?
Streaming is like reading a book page by page instead of waiting for the whole book to be printed. The user sees the first word immediately, improving perceived performance.

```python
# Non-streaming: Wait for entire response
response = llm.chat(messages)  # 5 seconds...
print(response)  # Then show

# Streaming: Show as it generates
for token in llm.stream(messages):
    print(token, end="")  # Shows immediately
```

#### 6. Why MCP?
MCP is like a universal power adapter. Your assistant (phone) can work in any country (MCP server) as long as you have the right adapter (MCP client).

This means you can add new capabilities (tools) from external services without modifying your core code.

#### 7. Why RAG?
RAG is like having a reference library. Instead of memorizing everything (fine-tuning), you look up relevant books (documents) and answer based on them.

This is more efficient because:
- You can update the library without retraining
- You can cite sources
- You can handle large knowledge bases

---

## Project Structure Summary

```
mini-ai-assistant/
├── cli/                          # Presentation Layer
│   └── chat/
│       └── cli.py               # Chat interface
├── application/                  # Application + Domain + Infrastructure
│   ├── bootstrap.py             # Composition root
│   ├── agent/                   # Application Layer
│   │   ├── agent.py            # Core agent with tool loop
│   │   └── bootstrap.py        # Agent factory
│   ├── assistant/               # Application Layer
│   │   └── engine.py           # Assistant orchestration
│   ├── router/                  # Application Layer
│   │   └── router.py           # Intent-based routing
│   ├── llm/                     # Infrastructure Layer
│   │   ├── client.py           # Abstract LLM interface
│   │   ├── ollama_client.py    # Ollama implementation
│   │   ├── message.py          # Message types
│   │   ├── response.py         # Response types
│   │   └── stream_event.py     # Streaming events
│   ├── rag/                     # Infrastructure + ML Layer
│   │   ├── engine.py           # RAG orchestration
│   │   ├── retriever.py        # Similarity search
│   │   ├── indexer.py          # Document indexing
│   │   ├── chunker.py          # Text chunking
│   │   ├── qdrant_vector_store.py  # Vector DB
│   │   └── ollama_embedding.py # Embeddings
│   ├── tools/                   # Domain Layer
│   │   ├── base.py             # Abstract Tool
│   │   ├── registry.py         # Tool registry
│   │   ├── knowledge_search.py # RAG search tool
│   │   ├── save_memory.py      # Memory tool
│   │   ├── load_memory.py      # Memory tool
│   │   └── update_memory.py    # Memory tool
│   ├── mcp/                     # Infrastructure Layer
│   │   ├── client/             # MCP client
│   │   │   └── client.py
│   │   ├── server/             # MCP server
│   │   │   ├── server.py
│   │   │   └── tools/          # MCP tools
│   │   │       ├── product_search.py
│   │   │       └── math_tools.py
│   │   ├── tool.py             # MCP tool wrapper
│   │   └── tools.py            # Tool discovery
│   ├── product_search/          # Domain + Infrastructure
│   │   ├── engine.py           # Search orchestration
│   │   ├── service.py          # Business logic
│   │   ├── fetcher.py          # HTTP fetching
│   │   ├── models.py           # Domain models
│   │   └── parsers/            # Site-specific parsers
│   │       ├── base.py
│   │       ├── torob.py
│   │       └── registry.py
│   ├── chat/                    # Domain Layer
│   │   ├── engine.py           # Chat orchestration
│   │   └── history.py          # Message history
│   └── utils/                   # Utilities
│       └── logger.py
├── intent_classifier/           # ML/AI Layer
│   ├── model.py                # Transformer model
│   ├── trainer.py              # Training loop
│   ├── predictor.py            # Inference wrapper
│   ├── tokenizer.py            # Text tokenization
│   ├── encoder.py              # Transformer encoder
│   ├── attention.py            # Multi-head attention
│   ├── config.py               # Model config
│   └── labels.py               # Intent labels
├── data/                        # Data files
│   ├── intent/                 # Training data
│   ├── documents/              # Knowledge base
│   ├── memory/                 # Persistent memory
│   └── vector_db/              # Vector storage
├── main.py                      # Entry point
├── train.py                     # Training script
├── predict.py                   # Prediction script
└── requirements.txt             # Dependencies
```

---

## Conclusion

The Mini AI Assistant demonstrates a well-architected, production-ready AI system with:

- **Clean separation of concerns** across 5 layers
- **Extensibility** through ABCs, registries, and dependency injection
- **Testability** with comprehensive unit and integration tests
- **Modern AI patterns** including RAG, tool calling, and intent classification
- **Real-world integrations** with Ollama, Qdrant, and MCP

This architecture serves as an excellent teaching example for:
- Layered architecture design
- Dependency injection and inversion of control
- Abstract base classes and polymorphism
- Registry and factory patterns
- Streaming and async programming
- RAG and tool-calling patterns in AI applications
