from fastapi import APIRouter

from app.schemas.resume_agent import (
    ResumeAgentDecisionCreate,
    ResumeAgentMessageCreate,
    ResumeAgentSession,
    ResumeAgentSessionCreate,
)
from app.services.resume_agent.orchestrator import get_resume_agent_orchestrator

router = APIRouter(prefix="/resume-agent", tags=["resume-agent"])


@router.post("/sessions", response_model=ResumeAgentSession)
async def create_resume_agent_session(payload: ResumeAgentSessionCreate):
    orchestrator = get_resume_agent_orchestrator()
    return await orchestrator.create_session(payload)


@router.get("/sessions/{session_id}", response_model=ResumeAgentSession)
async def get_resume_agent_session(session_id: str):
    orchestrator = get_resume_agent_orchestrator()
    return orchestrator.get_session(session_id)


@router.post("/sessions/{session_id}/messages", response_model=ResumeAgentSession)
async def post_resume_agent_message(session_id: str, payload: ResumeAgentMessageCreate):
    orchestrator = get_resume_agent_orchestrator()
    return await orchestrator.handle_message(session_id, payload)


@router.post("/sessions/{session_id}/decisions", response_model=ResumeAgentSession)
async def apply_resume_agent_decision(session_id: str, payload: ResumeAgentDecisionCreate):
    orchestrator = get_resume_agent_orchestrator()
    return await orchestrator.apply_decision(session_id, payload)
