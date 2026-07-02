"""JobFit orchestrator — three-step pipeline.

Flow:
  1. LLM extracts structured data from resume and JD
  2. Program calculates deterministic match score
  3. LLM generates human-readable suggestions
"""

import logging

from app.core.config import settings
from app.schemas.jobfit import (
    GapItem,
    InterviewQuestion,
    JobFitAnalysis,
    MatchItem,
    ResumeRewrite,
    ScoreBreakdown,
)
from app.services.llm import extract_resume, extract_jd, generate_suggestions
from app.services.matcher import calculate_match
from app.services.llm_clients.factory import get_llm_client

logger = logging.getLogger(__name__)


async def analyze_job_fit(resume_text: str, jd_text: str) -> JobFitAnalysis:
    """Main entry: extract → match → suggest → build response."""

    client = get_llm_client()

    # Step 1: LLM extraction
    logger.info("Step 1: Extracting resume profile...")
    resume_profile = await extract_resume(resume_text, client)
    logger.info(
        "Resume extracted | skills=%d, projects=%d, exp=%s",
        len(resume_profile.skills.hard),
        len(resume_profile.projects),
        resume_profile.experience_years.get("total", 0),
    )

    logger.info("Step 1: Extracting JD profile...")
    jd_profile = await extract_jd(jd_text, client)
    logger.info("JD extracted | requirements=%d", len(jd_profile.requirements))

    # Step 2: Program matching (deterministic)
    logger.info("Step 2: Calculating match...")
    match_result = calculate_match(resume_profile, jd_profile)
    logger.info(
        "Match calculated | total=%d, matched=%d, gaps=%d",
        match_result.total_score,
        len(match_result.matched),
        len(match_result.gaps),
    )

    # Step 3: LLM suggestions
    logger.info("Step 3: Generating suggestions...")
    suggestions = await generate_suggestions(match_result, resume_profile, jd_profile, client)

    # Build final response
    return JobFitAnalysis(
        match_score=match_result.total_score,
        summary=suggestions.get("summary", ""),
        matched_strengths=[
            MatchItem(
                requirement=d.requirement,
                resume_evidence=d.evidence,
                score=int(round(d.match_score * 100)),
            )
            for d in match_result.matched
        ][:12],
        gaps=[
            GapItem(
                requirement=g.requirement,
                suggestion=g.suggestion,
            )
            for g in match_result.gaps
        ],
        resume_rewrites=[
            ResumeRewrite(
                before=r.get("before", ""),
                after=r.get("after", ""),
                reason=r.get("reason", ""),
            )
            for r in suggestions.get("resume_rewrites", [])
        ],
        interview_questions=[
            InterviewQuestion(
                question=q.get("question", ""),
                focus=q.get("focus", ""),
                difficulty=q.get("difficulty", "medium"),
            )
            for q in suggestions.get("interview_questions", [])
        ],
        evidence=[],  # No RAG evidence anymore
        score_breakdown=match_result.score_breakdown,
        requirement_analysis=match_result.requirement_analyses,
        analysis_overview=match_result.analysis_overview,
        core_requirements=match_result.core_requirements,
        bonus_requirements=match_result.bonus_requirements,
        risk_items=match_result.risk_items,
        risk_details=match_result.risk_details,
        model_used=settings.deepseek_model,
        fallback_used=False,
    )
