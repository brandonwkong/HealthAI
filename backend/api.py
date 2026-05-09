from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from langgraph_agent import app, State

# Initialize FastAPI app
api = FastAPI(title="Health Agent API", version="1.0.0")

# Add CORS middleware so frontend can call the API
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class MessageRequest(BaseModel):
    user_message: str
    intake: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    follow_up_question: Optional[str] = None
    follow_up_answer: Optional[str] = None


class MessageResponse(BaseModel):
    final: Optional[str] = None
    follow_up_question: Optional[str] = None
    intake: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    urgent: Optional[bool] = None
    mcp_results: Optional[Dict[str, Any]] = None  # Results from MCP tool executions


@api.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Health Agent API"}


@api.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """
    Process a user message through the health agent.

    The agent may return:
    - A final response (final field populated)
    - A follow-up question (follow_up_question field populated)
    - Urgent flag if immediate care is needed
    """
    try:
        # Prepare state from request
        state: State = {
            "user_message": request.user_message,
            "intake": request.intake or {},
            "context": request.context or "",
        }

        # Add follow-up Q&A if provided
        if request.follow_up_question:
            state["follow_up_question"] = request.follow_up_question
        if request.follow_up_answer:
            state["follow_up_answer"] = request.follow_up_answer

        # Run the agent
        result = app.invoke(state)

        # Return response
        return MessageResponse(
            final=result.get("final"),
            follow_up_question=result.get("follow_up_question"),
            intake=result.get("intake"),
            context=result.get("context"),
            urgent=result.get("urgent"),
            mcp_results=result.get("mcp_results"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@api.post("/reset")
async def reset_session():
    """Reset the session state (for future session management)."""
    return {"status": "ok", "message": "Session reset (not implemented yet)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
