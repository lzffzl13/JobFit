from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse

from app.schemas.interview import InterviewAnswer, InterviewControl, InterviewStart
from app.schemas.resume_agent import (
    ResumeAgentDecisionCreate,
    ResumeAgentMessageCreate,
    ResumeAgentSession,
    ResumeAgentSessionCreate,
)
from app.schemas.resume_document import DocumentApply, DocumentEdit, DocumentRestore
from app.services.document_parser import parse_upload
from app.services.interview import InterviewService
from app.services.resume_agent.document_service import ResumeDocumentService
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


def document_service():
    return ResumeDocumentService(get_resume_agent_orchestrator())


@router.post("/sessions/{session_id}/document/preview", response_model=ResumeAgentSession)
def preview_resume_document(session_id: str):
    return document_service().preview(session_id)


@router.post("/sessions/{session_id}/document/apply", response_model=ResumeAgentSession)
def apply_resume_document(session_id: str, payload: DocumentApply):
    return document_service().apply(session_id, payload)


@router.put("/sessions/{session_id}/document", response_model=ResumeAgentSession)
def edit_resume_document(session_id: str, payload: DocumentEdit):
    return document_service().edit(session_id, payload)


@router.post("/sessions/{session_id}/document/restore", response_model=ResumeAgentSession)
def restore_resume_document(session_id: str, payload: DocumentRestore):
    return document_service().restore(session_id, payload)


@router.get("/sessions/{session_id}/document/export", response_class=PlainTextResponse)
def export_resume_document(session_id: str):
    return PlainTextResponse(document_service().export(session_id), headers={
        "Content-Disposition": 'attachment; filename="jobfit-resume.txt"',
    })


@router.post("/sessions/{session_id}/interview", response_model=ResumeAgentSession)
def start_interview(session_id: str, payload: InterviewStart):
    return InterviewService(get_resume_agent_orchestrator()).start(session_id, payload)


@router.post("/sessions/{session_id}/interview/answers", response_model=ResumeAgentSession)
async def answer_interview(session_id: str, payload: InterviewAnswer):
    return await InterviewService(get_resume_agent_orchestrator()).answer(session_id, payload)


@router.post("/sessions/{session_id}/interview/control", response_model=ResumeAgentSession)
def control_interview(session_id: str, payload: InterviewControl):
    return InterviewService(get_resume_agent_orchestrator()).control(session_id, payload)
