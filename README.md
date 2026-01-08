# AI Agent: Multi-Tool Intelligent Chat Interface

> A production-ready agentic pipeline that routes queries to specialized tools: real-time weather API and PDF document RAG. Built with **LangGraph**, **LangChain**, **Qdrant**, and **LangSmith**, powered by **Streamlit**.

## Overview

This project implements an intelligent agent that can:

- **🌤️ Fetch Real-Time Weather Data** – Query the OpenWeatherMap API for current conditions, temperature, and forecasts
- **📄 Answer PDF Questions via RAG** – Retrieve relevant content from PDF documents using semantic search with Qdrant vector embeddings
- **🧠 Intelligent Routing** – Use an LLM to classify queries as weather or document-related, with keyword fallback
- **⚡ LangGraph Orchestration** – Define complex multi-step workflows with type-safe state management
- **🔍 Full LangSmith Integration** – Log all interactions for evaluation, debugging, and continuous improvement
- **💬 Interactive Chat UI** – Beautiful Streamlit interface with real-time streaming responses and debug visibility

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (for LLM and embeddings)
- OpenWeatherMap API key (optional, for weather features)
- A PDF file for RAG (required for document queries)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or on macOS/Linux:
   # source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Create a `.env` file** in the project root:
   ```bash
   cp .env.example .env
   ```

2. **Required environment variables:**
   ```env
   # LLM Configuration
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL_NAME=gpt-4o-mini  # or your preferred OpenAI model
   
   # Weather API
   OPENWEATHER_API_KEY=your-api-key
   OPENWEATHER_DEFAULT_CITY=London
   OPENWEATHER_UNITS=metric
   
   # Document Storage
   PDF_PATH=/path/to/your/document.pdf
   PDF_COLLECTION_NAME=pdf_documents
   QDRANT_PATH=qdrant_local  # Local persistence directory
   ```

3. **Optional: LangSmith Configuration** (for evaluation and tracing):
   ```env
   LANGSMITH_API_KEY=ls-...
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=ai-agent
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   ```

### Running the Application

Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

The interface will open at `http://localhost:8501`. Ask questions about:

- **Weather:** "What's the temperature in Paris?" or "Will it rain in New York tomorrow?"
- **Documents:** "Summarize the main concepts from the PDF" or "What does the document say about X?"

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Chat UI                         │
└────────────────┬────────────────────────────────────────────┘
                 │ User Question
                 ▼
┌─────────────────────────────────────────────────────────────┐
│               LangGraph Agent Workflow                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐   ┌─────────────┐  │
│  │   Router     │─────►│   Weather    │   │   Qdrant    │  │
│  │  (LLM-Based) │      │   Node       │   │   Vector DB │  │
│  └──────────────┘      └──────────────┘   └─────────────┘  │
│         │                      │                   │         │
│         └──────────────────────┼───────────────────┘         │
│                                │                             │
│                         ┌──────▼────────┐                   │
│                         │   RAG Node     │                   │
│                         │  (LangChain)   │                   │
│                         └────────────────┘                   │
│                                                               │
└──────────────────────────────────┬──────────────────────────┘
                                   │ Answer
                         ┌─────────▼──────────┐
                         │  LangSmith Logger  │
                         └────────────────────┘
```

### Module Structure

```
ai_agent/
├── __init__.py              # Package exports
├── config.py                # Configuration management (env-based settings)
├── llm.py                   # LLM and embedding model initialization (cached)
├── weather.py               # OpenWeatherMap API client & prompt generation
├── rag.py                   # PDF loading, Qdrant vector store, RAG chains
├── graph.py                 # LangGraph agent definition (routing & orchestration)
└── evaluation.py            # LangSmith logging for evaluation

tests/
├── test_graph.py            # Router logic tests
├── test_weather.py          # Weather extraction and API tests
└── test_rag.py              # RAG chain and retrieval tests

app.py                        # Streamlit application entry point
requirements.txt              # Python dependencies
```

## Key Features

### 🔀 Intelligent Query Routing

The router uses a two-tier approach:

1. **LLM-Based Routing** (Primary): Uses GPT to classify queries as "weather" or "rag"
2. **Keyword Fallback** (Fallback): Detects weather keywords like "temperature", "rain", "forecast"

```python
# Example: automatic routing
"What's the weather in London?" → Weather node
"Summarize the document" → RAG node
```

### 🌐 Weather Integration

- Real-time weather data from OpenWeatherMap API
- Automatic city extraction from natural language queries
- LLM-powered summarization for human-friendly responses
- Error handling with graceful fallbacks

### 📚 PDF RAG Pipeline

- **Loading**: PyPDF loader for robust PDF parsing
- **Embedding**: OpenAI text-embedding-3-small for semantic search
- **Storage**: Qdrant vector database (local persistence)
- **Retrieval**: Cosine similarity search with LangChain retrievers
- **Generation**: LLM-powered RAG chain with prompt templates

### 📊 Observability & Evaluation

- **LangSmith Integration**: Log all queries, routes, and answers for offline evaluation
- **Debug Mode**: Inspect retrieved documents and raw API responses in the UI
- **Response Tracing**: Full execution traces for debugging agent behavior

## Environment Variables Reference

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `OPENAI_API_KEY` | String | ✓ | OpenAI API key for LLM and embeddings |
| `OPENAI_MODEL_NAME` | String | | Model to use (default: `gpt-4o-mini`) |
| `OPENWEATHER_API_KEY` | String | | OpenWeatherMap API key (required for weather) |
| `OPENWEATHER_DEFAULT_CITY` | String | | Default city if none extracted (default: `London`) |
| `OPENWEATHER_UNITS` | String | | Unit system: `metric` or `imperial` (default: `metric`) |
| `PDF_PATH` | String | ✓ | Path to PDF file for RAG |
| `PDF_COLLECTION_NAME` | String | | Qdrant collection name (default: `pdf_documents`) |
| `QDRANT_PATH` | String | | Local Qdrant persistence path (default: `qdrant_local`) |
| `LANGSMITH_API_KEY` | String | | LangSmith API key for evaluation (optional) |
| `LANGCHAIN_TRACING_V2` | Boolean | | Enable LangChain tracing (optional) |
| `LANGCHAIN_PROJECT` | String | | LangSmith project name (optional) |

## Testing

Run the test suite to validate core functionality:

```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest tests/test_graph.py -v   # Run specific test file
```

**Test Coverage:**
- Router keyword detection and LLM routing
- Weather city extraction and API integration
- RAG document retrieval and chain execution

## Development

### Project Dependencies

- **LangChain**: Core LLM framework and utilities
- **LangGraph**: Agentic workflow orchestration
- **Qdrant**: Vector database for embeddings
- **Streamlit**: Interactive web UI
- **OpenAI**: LLM and embedding models
- **PyPDF**: PDF document loading
- **LangSmith**: Observability and evaluation

### Extending the Agent

Add new tools/nodes by:

1. Create a new module in `ai_agent/` with your tool logic
2. Add a new node function in `graph.py`
3. Update the router to recognize queries for your tool
4. Add test cases in `tests/`

Example:
```python
def custom_node(state: AgentState) -> AgentState:
    result = my_tool(state["question"])
    return {"answer": result, "route": "custom", "context_docs": []}

workflow.add_node("custom", custom_node)
```

## Troubleshooting

### "PDF_PATH is not set" Error
- Ensure `.env` file exists with valid `PDF_PATH`
- Use absolute path or relative path from project root

### Weather API Returns 401
- Verify `OPENWEATHER_API_KEY` is correct and active
- Check API key has not expired

### Qdrant Collection Already Exists
- The agent automatically handles existing collections
- To reset: delete the `qdrant_local/` directory

### LangSmith Logging Failures
- Won't break the application (graceful failure)
- Configure `LANGSMITH_API_KEY` and `LANGCHAIN_TRACING_V2=true` to enable
- Check logs in LangSmith dashboard

## Performance Notes

- **LLM Caching**: Chat models and embeddings are cached to reduce redundant calls
- **Vector Store**: Qdrant uses local persistence, suitable for small-to-medium documents
- **Streaming**: Streamlit UI supports real-time response streaming
- **Concurrency**: State-based design supports horizontal scaling

## License

[MIT Licensed]

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Last Updated:** January 2026  
**Python Version:** 3.9+  
**Status:** Production-Ready

Tests cover:

- **Weather API handling** (mocked HTTP responses).
- **RAG retrieval and LLM processing logic** (with fake embeddings and fake LLM).
- **LangGraph routing logic** (ensuring weather vs RAG paths work as expected).

## 5. LangSmith evaluation

This project is wired to work with **LangSmith**:

- If `LANGSMITH_API_KEY` is set, runs will be logged to your LangSmith workspace.
- The `evaluation.log_response_for_evaluation` helper creates examples for each answer so you can:
  - Inspect runs and traces.
  - Attach examples to datasets.
  - Run **LLM-as-judge evaluations** directly in the LangSmith UI.

Include screenshots of your LangSmith project showing example runs and any evaluations you perform.

## 6. Loom demo

For the Loom video, you can:

- Walk through the **code structure** (config, LLM, weather, RAG, graph, UI).
- Show the **Streamlit UI** answering both weather and PDF questions.
- Show **LangSmith** capturing runs and any evaluations.

This repository is intentionally kept small and focused so you can quickly understand and extend it.
