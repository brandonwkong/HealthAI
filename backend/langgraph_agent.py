from typing import TypedDict, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# MCP client for post-triage tool execution
from mcp_client import call_tool_sync

load_dotenv()

# Settings
Settings.llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

DOCS_DIR = "knowledge"
INTAKE_QUESTIONS = [
    ("onset_time", "When did it start? (e.g., today, yesterday, 3 days ago)"),
    ("sudden", "Did it come on suddenly (seconds/minutes) or gradually? (sudden/gradual)"),
    ("severity", "How bad is it? (mild/moderate/severe)"),
    ("fever", "Do you have a fever? (yes/no)"),
    ("neuro", "Any new weakness/numbness, confusion, fainting, or vision changes? (yes/no)"),
    ("injury", "Any recent head injury? (yes/no)"),
]

# Initialize RAG components
documents = SimpleDirectoryReader(DOCS_DIR).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=4)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# State definition
class State(TypedDict, total=False):
    user_message: str
    intake: Dict[str, Any]
    follow_up_question: Optional[str]
    follow_up_answer: Optional[str]
    urgent: bool
    final: str
    context: str
    mcp_results: Dict[str, Any]  # Results from MCP tool executions


# --- Node Functions ---
def init_intake(state: State) -> State:
    state.setdefault("intake", {})
    return state


def record_answer(state: State) -> State:
    if state.get("follow_up_question") and state.get("follow_up_answer"):
        intake = state.setdefault("intake", {})
        for field, q in INTAKE_QUESTIONS:
            if q == state["follow_up_question"]:
                intake[field] = state["follow_up_answer"].strip()
                break
        print("\n[DEBUG] Updated intake:")
        for k, v in intake.items():
            print(f"  - {k}: {v}")
        state["follow_up_question"] = None
        state["follow_up_answer"] = None
    return state


def red_flag_check(state: State) -> State:
    text = state["user_message"].lower()
    urgent_terms = ["worst headache", "confusion", "weakness", "faint", "seizure", "stiff neck"]
    state["urgent"] = any(t in text for t in urgent_terms)
    return state


def urgent_response(state: State) -> State:
    state["final"] = (
        "This could be urgent based on what you said. "
        "Seek urgent medical care now or call local emergency services. "
        "If you can, have someone stay with you."
    )
    return state


def ask_next_question(state: State) -> State:
    intake = state.setdefault("intake", {})
    for field, q in INTAKE_QUESTIONS:
        if field not in intake:
            state["follow_up_question"] = q
            return state
    return state


def normalize_intake(state: State) -> State:
    """Convert intake strings into normalized values (booleans/enums)."""
    intake = state.setdefault("intake", {})
    norm = intake.setdefault("_norm", {})

    def yn(v: Any) -> Optional[bool]:
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"yes", "y", "true", "1"}: return True
        if s in {"no", "n", "false", "0"}: return False
        return None

    sudden = intake.get("sudden")
    if sudden is not None:
        s = str(sudden).strip().lower()
        if s in {"sudden", "s"}: norm["sudden"] = "sudden"
        elif s in {"gradual", "g"}: norm["sudden"] = "gradual"

    sev = intake.get("severity")
    if sev is not None:
        s = str(sev).strip().lower()
        if s in {"mild", "moderate", "severe"}:
            norm["severity"] = s

    norm["fever"] = yn(intake.get("fever"))
    norm["neuro"] = yn(intake.get("neuro"))
    norm["injury"] = yn(intake.get("injury"))

    return state


def post_intake_red_flags(state: State) -> State:
    """Deterministic escalation using normalized intake."""
    state["urgent"] = False

    intake = state.get("intake", {})
    norm = intake.get("_norm", {})

    sudden = norm.get("sudden")
    severity = norm.get("severity")
    neuro = norm.get("neuro")
    injury = norm.get("injury")

    if neuro is True:
        state["urgent"] = True
    elif sudden == "sudden" and severity == "severe":
        state["urgent"] = True
    elif injury is True and severity == "severe":
        state["urgent"] = True

    return state


def generate_response(state: State) -> State:
    intake = state.get("intake", {})
    context = state.get("context", "")

    prompt = f"""
You are a health information assistant. Do not diagnose. Provide general info and safe next steps.
Use the provided Context if relevant; if not, ignore it.

Context:
{context}

User: {state["user_message"]}
Intake: {intake}

Return:
- 3 brief possible common explanations (general, not diagnostic)
- 3 self-care steps
- 3 reasons to seek urgent care
- 1 short clinician-summary sentence
"""
    state["final"] = llm.invoke(prompt).content
    return state


def retrieve_context(state: State) -> State:
    intake = state.get("intake", {})
    q = f"User message: {state.get('user_message','')}\nIntake: {intake}\n"

    resp = query_engine.query(q)

    print("\n[DEBUG] LlamaIndex retrieval query:")
    print(q)

    snippets = []
    for i, sn in enumerate(getattr(resp, "source_nodes", [])[:4]):
        txt = sn.node.get_content(metadata_mode="none").strip()
        print(f"\n[DEBUG] Retrieved chunk {i+1}:")
        print(txt[:500])
        snippets.append(txt)

    if not snippets:
        print("\n[DEBUG] No chunks retrieved.")

    state["context"] = "\n\n---\n\n".join(snippets)
    return state


# =============================================================================
# MCP Tool Execution Node
# =============================================================================
# This node runs AFTER triage is complete. It calls MCP tools based on the
# triage decision. This is the "approved decision state" - we only trigger
# external actions after the agent has made its decision.

def execute_post_triage_actions(state: State) -> State:
    """
    Execute MCP tools after triage decision is finalized.

    This is the key integration point - MCP tools only fire AFTER:
    1. All intake questions are answered
    2. Red flag checks are complete
    3. The triage decision (urgent/non-urgent) is made
    4. The response has been generated

    This ensures we never trigger actions prematurely.
    """
    mcp_results = {}
    intake = state.get("intake", {})
    is_urgent = state.get("urgent", False)

    print("\n[MCP] Executing post-triage actions...")

    # TOOL 1: Always log the triage result
    try:
        log_result = call_tool_sync("log_triage_result", {
            "patient_message": state.get("user_message", ""),
            "intake_data": intake,
            "triage_decision": "urgent" if is_urgent else "non-urgent",
            "is_urgent": is_urgent,
            "ai_response": state.get("final", "")
        })
        mcp_results["log"] = log_result
        print(f"[MCP] Triage logged: {log_result.get('log_id', 'unknown')}")
    except Exception as e:
        print(f"[MCP] Error logging triage: {e}")
        mcp_results["log"] = {"success": False, "error": str(e)}

    # TOOL 2: If urgent, send an alert
    if is_urgent:
        try:
            # Determine why it was flagged urgent
            norm = intake.get("_norm", {})
            reasons = []
            if norm.get("neuro") is True:
                reasons.append("neurological symptoms reported")
            if norm.get("sudden") == "sudden" and norm.get("severity") == "severe":
                reasons.append("sudden onset with severe pain")
            if norm.get("injury") is True and norm.get("severity") == "severe":
                reasons.append("recent head injury with severe symptoms")

            urgency_reason = "; ".join(reasons) if reasons else "red flag keywords detected"

            alert_result = call_tool_sync("send_urgent_alert", {
                "patient_message": state.get("user_message", ""),
                "intake_summary": str(intake),
                "urgency_reason": urgency_reason
            })
            mcp_results["alert"] = alert_result
            print(f"[MCP] Urgent alert sent: {alert_result.get('alert_id', 'unknown')}")
        except Exception as e:
            print(f"[MCP] Error sending alert: {e}")
            mcp_results["alert"] = {"success": False, "error": str(e)}

    # TOOL 3: If non-urgent, schedule follow-up
    else:
        try:
            # Determine recommended timeframe based on severity
            severity = intake.get("_norm", {}).get("severity", "mild")
            if severity == "severe":
                timeframe = "within 24-48 hours"
            elif severity == "moderate":
                timeframe = "within 1 week"
            else:
                timeframe = "within 2 weeks if symptoms persist"

            followup_result = call_tool_sync("schedule_followup", {
                "patient_message": state.get("user_message", ""),
                "recommended_timeframe": timeframe,
                "visit_type": "primary_care"
            })
            mcp_results["followup"] = followup_result
            print(f"[MCP] Follow-up scheduled: {followup_result.get('appointment_id', 'unknown')}")
        except Exception as e:
            print(f"[MCP] Error scheduling follow-up: {e}")
            mcp_results["followup"] = {"success": False, "error": str(e)}

    state["mcp_results"] = mcp_results
    print("[MCP] Post-triage actions complete.\n")

    return state


# --- Route Functions ---
def route_after_red_flags(state: State) -> str:
    return "urgent" if state.get("urgent") else "continue"


def route_after_post_red_flags(state: State) -> str:
    return "urgent" if state.get("urgent") else "retrieve"


def route_after_ask_next(state: State) -> str:
    if state.get("follow_up_question"):
        return "ask"
    return "normalize"


# --- Graph Construction ---
def create_graph():
    """Create and compile the health agent graph."""
    g = StateGraph(State)

    # Add nodes
    g.add_node("init_intake", init_intake)
    g.add_node("record_answer", record_answer)
    g.add_node("red_flags", red_flag_check)
    g.add_node("urgent_response", urgent_response)
    g.add_node("ask_next", ask_next_question)
    g.add_node("respond", generate_response)
    g.add_node("normalize_intake", normalize_intake)
    g.add_node("post_red_flags", post_intake_red_flags)
    g.add_node("retrieve", retrieve_context)
    g.add_node("mcp_actions", execute_post_triage_actions)  # MCP tool execution

    # Add edges
    g.add_edge("init_intake", "record_answer")
    g.add_edge("record_answer", "red_flags")
    g.add_edge("normalize_intake", "post_red_flags")
    # NOTE: post_red_flags uses conditional edges (not a regular edge)
    # to route to either urgent_response OR retrieve
    g.add_edge("retrieve", "respond")
    # After responses, execute MCP tools, then end
    g.add_edge("urgent_response", "mcp_actions")
    g.add_edge("respond", "mcp_actions")
    g.add_edge("mcp_actions", END)

    # Set entry point
    g.set_entry_point("init_intake")

    # Add conditional edges
    g.add_conditional_edges(
        "red_flags",
        route_after_red_flags,
        {"urgent": "urgent_response", "continue": "ask_next"},
    )

    g.add_conditional_edges(
        "post_red_flags",
        route_after_post_red_flags,
        {"urgent": "urgent_response", "retrieve": "retrieve"},
    )

    g.add_conditional_edges(
        "ask_next",
        route_after_ask_next,
        {"ask": END, "normalize": "normalize_intake"},
    )

    return g.compile()


# Create the compiled app (singleton)
app = create_graph()
