# Health Agent

An agentic healthcare intake and triage system using LangGraph for multi-step workflow orchestration with deterministic branching, safety gating, and MCP-based tool execution. Built this out of curiousity to play around with book Langgraph and MCP tools. Working on integrating real appointment scheduling and other MCP tools next!

## Features

- **LangGraph Workflow Orchestration**: Multi-step intake process with deterministic branching and conditional routing
- **Safety Gating**: Red flag detection at multiple stages (keyword-based + rule-based) to identify urgent cases
- **RAG-Powered Responses**: LlamaIndex ingestion pipeline grounds triage decisions in uploaded medical documents
- **MCP Tool Execution**: Post-triage actions (logging, alerts, scheduling) only trigger after approved decision state
- **Dual Interface**: REST API for integration + CLI for testing

## Architecture

```
User Query → Init Intake → Red Flag Check → Intake Questions (loop)
                                ↓
                         Normalize → Post-Intake Red Flags
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
              [If Urgent]            [If Non-Urgent]
            Urgent Response          RAG Retrieve → Generate Response
                    ↓                       ↓
                    └───────────┬───────────┘
                                ↓
                        MCP Tool Execution
                    (log, alert, or schedule)
                                ↓
                               END
```

## Project Structure

```
health-agent/
├── backend/
│   ├── langgraph_agent.py   # Core agent with state machine
│   ├── api.py               # FastAPI REST endpoint
│   ├── cli.py               # Command-line interface
│   ├── mcp_server.py        # MCP server with healthcare tools
│   ├── mcp_client.py        # MCP client helper
│   ├── requirements.txt     # Python dependencies
│   └── knowledge/
│       └── headache_basics.md  # RAG knowledge base
├── frontend/
│   └── frontend.html        # Web UI (single-file, no build)
├── .env.example             # Environment variables template
├── .gitignore
└── README.md
```

## Setup

### 1. Clone and navigate
```bash
git clone https://github.com/YOUR_USERNAME/health-agent.git
cd health-agent
```

### 2. Create virtual environment
```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
# Create .env in the backend folder with your OpenAI API key
cp ../.env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Option 1: CLI (for testing)
```bash
cd backend
python cli.py
```

### Option 2: API + Web UI
```bash
# Terminal 1: Start the API
cd backend
python api.py

# Terminal 2: Open the frontend
# Just open frontend/frontend.html in your browser
```

The API runs at `http://localhost:8000`

## MCP Tools

The system executes these tools **after** triage is complete:

| Tool | Trigger | Action |
|------|---------|--------|
| `log_triage_result` | Always | Saves triage outcome to `triage_logs.json` |
| `send_urgent_alert` | If urgent | Logs alert to `urgent_alerts.json` |
| `schedule_followup` | If non-urgent | Creates appointment in `appointments.json` |

## Intake Questions

The agent collects:
1. **Onset time** - When symptoms started
2. **Sudden vs gradual** - How symptoms appeared
3. **Severity** - Mild, moderate, or severe
4. **Fever** - Yes or no
5. **Neurological symptoms** - Weakness, numbness, confusion, vision changes
6. **Head injury** - Recent trauma

## Red Flag Detection

**Immediate (keyword-based):**
- "worst headache", "confusion", "weakness", "faint", "seizure", "stiff neck"

**Post-intake (rule-based):**
- Neurological symptoms → Urgent
- Sudden onset + severe → Urgent
- Head injury + severe → Urgent

## Tech Stack

- **LangGraph** - State machine orchestration
- **LangChain + OpenAI** - LLM integration (GPT-4o-mini)
- **LlamaIndex** - RAG pipeline with OpenAI embeddings
- **MCP (Model Context Protocol)** - Tool execution framework
- **FastAPI** - REST API
- **Tailwind CSS** - Frontend styling

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/chat` | Process message through agent |
| POST | `/reset` | Reset session (placeholder) |

## License

MIT
