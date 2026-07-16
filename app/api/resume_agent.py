from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.resume_agent import (
    ResumeAgentDecisionCreate,
    ResumeAgentMessageCreate,
    ResumeAgentSession,
    ResumeAgentSessionCreate,
)
from app.services.document_parser import parse_upload
from app.services.resume_agent.orchestrator import get_resume_agent_orchestrator

router = APIRouter(prefix="/resume-agent", tags=["resume-agent"])


@router.post("/sessions", response_model=ResumeAgentSession)
async def create_resume_agent_session(payload: ResumeAgentSessionCreate):
    orchestrator = get_resume_agent_orchestrator()
    return await orchestrator.create_session(payload)


@router.post("/sessions/from-document", response_model=ResumeAgentSession)
async def create_resume_agent_session_from_document(
    resume: UploadFile | None = File(None),
    resume_text: str | None = Form(None),
    jd_text: str = Form(..., min_length=20),
):
    """Create an Agent session from pasted text or an uploaded resume."""
    resolved_resume_text = (resume_text or "").strip()

    if not resolved_resume_text and resume is not None:
        try:
            resolved_resume_text = (await parse_upload(resume)).strip()
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if len(resolved_resume_text) < 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a resume file or paste resume text with at least 30 characters.",
        )

    orchestrator = get_resume_agent_orchestrator()
    return await orchestrator.create_session(
        ResumeAgentSessionCreate(resume_text=resolved_resume_text, jd_text=jd_text)
    )


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
