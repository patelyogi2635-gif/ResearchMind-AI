<div align="center">

<img src="https://img.shields.io/badge/ResearchMind-AI-E8823A?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTEyIDh2NGwyIDIiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9zdmc+" alt="ResearchMind AI"/>

# ResearchMind AI

### Autonomous AI Research Agent powered by LangGraph + MCP

[![Live Demo](https://img.shields.io/badge/Live_Demo-researchmind--ai.netlify.app-1B2B4B?style=flat-square&logo=netlify)](https://researchmind-ai.netlify.app)
[![Backend](https://img.shields.io/badge/Backend-Railway-7BA694?style=flat-square&logo=railway)](https://researchmind-ai-production.up.railway.app/api/health)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-E8823A?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

**ResearchMind AI** is a production-grade autonomous research agent that breaks your question into sub-queries, searches the live web and GitHub via MCP servers, scores and deduplicates sources, then synthesizes a cited markdown report — all in ~35 seconds.

<br/>

[🚀 Live Demo](https://researchmind-ai.netlify.app) · [📖 API Docs](https://researchmind-ai-production.up.railway.app/docs) · [🐛 Report Bug](https://github.com/patelyogi2635-gif/ResearchMind-AI/issues) · [💡 Request Feature](https://github.com/patelyogi2635-gif/ResearchMind-AI/issues)

</div>

---

## 📸 Demo

> Ask any research question → watch 3 LangGraph graphs fire in sequence → get a cited report with real sources

**Example query:** *"How does LangGraph compare to CrewAI for building production AI agents?"*

```
🧠 Planner      → Generated 4 sub-questions                    (1.4s)
🔍 Researcher   → Found 8 sources via Tavily + GitHub MCP      (34s)
📝 Writer       → Synthesized 2100 char cited report            (1.8s)
                                                          Total: ~37s
```

---

## ✨ Features

- **🧠 Query Decomposition** — Planner graph uses Groq LLM to break your query into 4 focused sub-questions covering definition, comparison, use cases, and limitations
- **🔍 Dual MCP Search** — Tavily MCP searches the live web; GitHub MCP searches repositories. Both run for every sub-question
- **📊 Intelligent Scoring** — Raw results are deduplicated by URL, scored by relevance, and ranked. Only top sources reach the Writer
- **📝 Cited Report Synthesis** — Writer graph produces a structured markdown report with inline citations, key findings, and a sources list
- **⚡ Groq Inference** — All LLM nodes run on `llama-3.3-70b-versatile` at ~800 tokens/sec — fast and free
- **🔗 MCP Protocol** — Tools connect via the open Model Context Protocol standard. Swap any MCP server without touching graph logic
- **🌊 Streaming Ready** — Pipeline service has `run_pipeline_streaming()` for SSE streaming (Phase 4)
- **🏗️ Modular Architecture** — Each LangGraph graph is independently testable and replaceable

---

## 🏛️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              FastAPI Backend                │
│         POST /api/research                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│          LangGraph Pipeline                 │
│                                             │
│  ┌─────────────┐                            │
│  │   PLANNER   │  decompose_query node      │
│  │   GRAPH     │  → sub_questions: [...]    │
│  └──────┬──────┘                            │
│         │                                   │
│  ┌──────▼──────┐                            │
│  │  RESEARCH   │  search_web node           │
│  │   GRAPH     │  search_github node        │
│  │             │  score_sources node        │
│  └──────┬──────┘  → sources: [...]         │
│         │                                   │
│  ┌──────▼──────┐                            │
│  │   WRITER    │  select_sources node       │
│  │   GRAPH     │  synthesize_report node    │
│  └──────┬──────┘  → report: "..."          │
└─────────┼───────────────────────────────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
Tavily MCP   GitHub MCP
(web search) (repo search)
```

### State flow

```python
ResearchState = {
    "query":          str,                    # input
    "sub_questions":  list[str],              # after Planner
    "search_results": Annotated[list, add],   # after Research (accumulates)
    "sources":        list[SearchResult],     # after scoring
    "report":         str,                    # after Writer
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Orchestration** | LangGraph 0.2 — multi-graph state machine |
| **Tool Integration** | langchain-mcp-adapters — MCP protocol client |
| **Web Search** | Tavily MCP Server — live web search |
| **Code Search** | GitHub MCP Server — repository search |
| **LLM** | Groq llama-3.3-70b-versatile — ~800 tok/s |
| **Backend** | FastAPI + uvicorn — async REST API |
| **Frontend** | Vanilla HTML/CSS/JS — no build step |
| **Deployment** | Railway (backend) + Netlify (frontend) |

---

## 📁 Project Structure

```
ResearchMind-AI/
├── backend/
│   ├── app/
│   │   ├── graphs/
│   │   │   ├── planner.py       # Graph 1: query decomposition
│   │   │   ├── research.py      # Graph 2: MCP search + scoring
│   │   │   ├── writer.py        # Graph 3: report synthesis
│   │   │   └── prompts.py       # all LLM prompt templates
│   │   ├── routers/
│   │   │   ├── health.py        # GET /api/health, /api/health/mcp
│   │   │   └── research.py      # POST /api/research
│   │   ├── services/
│   │   │   └── pipeline.py      # orchestrates all 3 graphs
│   │   ├── config.py            # pydantic settings (env vars)
│   │   ├── llm.py               # Groq LLM factory
│   │   ├── mcp_client.py        # MultiServerMCPClient setup
│   │   ├── main.py              # FastAPI app + lifespan
│   │   └── state.py             # ResearchState TypedDict
│   ├── tests/
│   │   ├── test_phase1_mcp.py   # MCP connection smoke tests
│   │   └── test_phase2_graphs.py # graph integration tests
│   ├── railway.toml             # Railway deployment config
│   ├── requirements.txt
│   └── run.py                   # Windows-safe uvicorn launcher
└── frontend/
    └── index.html               # full landing page + live demo UI
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for MCP server subprocesses)
- API keys: [Groq](https://console.groq.com) · [Tavily](https://app.tavily.com) · [GitHub Token](https://github.com/settings/tokens)

### 1. Clone the repo

```bash
git clone https://github.com/patelyogi2635-gif/ResearchMind-AI.git
cd ResearchMind-AI/backend
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_...          # console.groq.com — free
TAVILY_API_KEY=tvly-...       # app.tavily.com — free (1000/month)
GITHUB_TOKEN=ghp_...          # github.com/settings/tokens — free
APP_ENV=development
LOG_LEVEL=INFO
```

### 5. Run the backend

```bash
# Windows
python run.py

# Mac/Linux
uvicorn app.main:app --reload --port 8000
```

### 6. Open the frontend

Just open `frontend/index.html` in your browser. No build step needed.

Or visit the live demo: **[researchmind-ai.netlify.app](https://researchmind-ai.netlify.app)**

---

## 🧪 Running Tests

```bash
cd backend/

# Test MCP server connections (requires API keys)
pytest tests/test_phase1_mcp.py -v -s

# Test all three LangGraph graphs end to end
pytest tests/test_phase2_graphs.py -v -s
```

---

## 📡 API Reference

### `POST /api/research`

Run the full research pipeline.

**Request:**
```json
{
  "query": "How does LangGraph compare to CrewAI for production AI agents?"
}
```

**Response:**
```json
{
  "query": "How does LangGraph...",
  "sub_questions": ["What is LangGraph?", "What is CrewAI?", "..."],
  "sources": [
    {
      "title": "LangGraph Tutorial - DataCamp",
      "url": "https://datacamp.com/tutorial/langgraph-tutorial",
      "source": "tavily",
      "relevance_score": 0.7
    }
  ],
  "report": "## Executive Summary\n...",
  "source_count": 3
}
```

### `GET /api/health`
```json
{ "status": "ok", "service": "research-agent" }
```

### `GET /api/health/mcp`
```json
{
  "tavily": { "status": "connected", "tools": ["tavily_search", "..."] },
  "github": { "status": "connected", "tools": ["search_repositories", "..."] },
  "total_connected": 2
}
```

---

## 🌐 Deployment

### Backend → Railway

1. Push to GitHub
2. New project → Deploy from GitHub repo
3. Set **Root Directory** to `backend`
4. Add environment variables (GROQ_API_KEY, TAVILY_API_KEY, GITHUB_TOKEN)
5. Railway reads `railway.toml` and auto-deploys

```toml
# railway.toml
[build]
builder = "nixpacks"

[build.nixpacksPlan.phases.setup]
nixPkgs = ["python311", "nodejs_20"]

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/api/health"
```

### Frontend → Netlify

1. Update `const API` in `index.html` to your Railway URL
2. Drag `frontend/` folder to [netlify.com/drop](https://netlify.com/drop)
3. Done — live in 10 seconds

---

## 🔑 Getting API Keys (All Free)

| Service | URL | Free Tier |
|---------|-----|-----------|
| **Groq** | [console.groq.com](https://console.groq.com) | Unlimited requests, rate limited |
| **Tavily** | [app.tavily.com](https://app.tavily.com) | 1,000 searches/month |
| **GitHub Token** | [github.com/settings/tokens](https://github.com/settings/tokens) | Free forever — check `public_repo` + `read:user` |

---

## 🧠 Key Concepts

### Why LangGraph over a single LLM call?

A single LLM call can't browse the internet, search GitHub, or synthesize from multiple real sources. LangGraph orchestrates multiple nodes in a directed graph — each node does one thing and passes state forward. This makes the pipeline modular, debuggable, and extensible.

### Why MCP (Model Context Protocol)?

MCP is an open standard by Anthropic for connecting AI agents to external tools. Instead of writing custom API wrappers for Tavily and GitHub, `langchain-mcp-adapters` connects to their MCP servers and converts tool definitions to LangChain `BaseTool` objects automatically. Swap Tavily for any MCP-compatible search server without touching graph logic.

### Why Groq over OpenAI?

Groq runs `llama-3.3-70b-versatile` at ~800 tokens/sec — 10x faster than OpenAI GPT-4o and completely free on the developer tier. For a research pipeline that makes 3+ LLM calls per query, speed and cost matter.

### State accumulation with `operator.add`

```python
# In state.py
search_results: Annotated[list[SearchResult], operator.add]
```

This tells LangGraph to **append** when nodes return new results, not overwrite. So after `search_web` and `search_github` both write to `search_results`, the state contains results from both — not just the last one.

---

## 🗺️ Roadmap

- [x] Phase 1 — MCP server connections (Tavily + GitHub)
- [x] Phase 2 — Three LangGraph graphs (Planner → Research → Writer)
- [x] Phase 3 — FastAPI backend + frontend UI
- [x] Phase 4 — Deployed on Railway + Netlify
- [ ] Phase 5 — SSE streaming (real-time progress to frontend)
- [ ] Phase 6 — Session history + report export (PDF/Markdown)
- [ ] Phase 7 — Add ArXiv MCP for academic paper search
- [ ] Phase 8 — Async parallel research with LangGraph `Send()` API

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
# Fork the repo, then:
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with LangGraph · MCP · FastAPI · Groq · Python 3.11**

[![GitHub](https://img.shields.io/badge/GitHub-patelyogi2635--gif-181717?style=flat-square&logo=github)](https://github.com/patelyogi2635-gif)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Patel_Yogi-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/patel-yogi-0a2526346)

⭐ Star this repo if you found it useful

</div>
