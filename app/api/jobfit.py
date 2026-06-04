import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.jobfit import JobFitAnalysis
from app.services.document_parser import parse_upload
from app.services.jobfit import analyze_job_fit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobfit", tags=["jobfit"])


@router.post("/analyze", response_model=JobFitAnalysis)
async def analyze_resume_and_jd(
    resume: UploadFile | None = File(None),
    resume_text: str | None = Form(None),
    jd_text: str = Form(..., min_length=20),
):
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

    try:
        return await analyze_job_fit(resume_text=resolved_resume_text, jd_text=jd_text)
    except Exception as exc:
        logger.error("Analysis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI 分析服务暂时不可用，请稍后重试。",
        ) from exc
