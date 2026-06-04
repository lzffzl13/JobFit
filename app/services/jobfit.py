"""JobFit orchestrator — main analysis pipeline.

Flow: resume + JD → chunk → vector retrieve → LLM analysis → validate → output
On LLM failure → raise error (no fallback)
"""

import logging

from app.core.config import settings
from app.schemas.jobfit import (
    Evidence,
    GapItem,
    JobFitAnalysis,
    MatchItem,
    Requirement,
)
from app.services.llm import (
    SYSTEM_PROMPT,
    build_user_prompt,
    normalize_llm_payload,
)
from app.services.llm_clients.factory import get_llm_client
from app.services.retriever import chunk_text, retrieve_evidence
from app.services.validator import validate_and_score

logger = logging.getLogger(__name__)


async def analyze_job_fit(resume_text: str, jd_text: str) -> JobFitAnalysis:
    """Main entry: chunk → retrieve → LLM → validate → build result."""
    # 1. Chunk and retrieve
    resume_chunks = chunk_text(resume_text, source="resume")
    jd_chunks = chunk_text(jd_text, source="jd")

    resume_evidence = retrieve_evidence(jd_text[:4000], resume_chunks, top_k=8)
    jd_evidence = retrieve_evidence(resume_text[:4000], jd_chunks, top_k=5) or jd_chunks[:5]

    resume_context = _format_context(resume_evidence)
    jd_context = _format_context(jd_evidence)

    # 2. LLM analysis
    client = get_llm_client()
    user_prompt = build_user_prompt(
        resume_context[: settings.max_context_chars],
        jd_context[: settings.max_context_chars],
    )
    raw = await client.analyze(SYSTEM_PROMPT, user_prompt)
    normalized = normalize_llm_payload(raw)

    # 3. Validate (clamp + confidence check)
    validated = validate_and_score(
        llm_requirements=normalized.get("requirements", []),
        resume_text=resume_text,
        llm_match_score=normalized.get("match_score", 0),
        llm_bonus_score=normalized.get("bonus_score", 0),
        llm_extra_score=normalized.get("extra_score", 0),
    )

    # 4. Build response
    analysis = JobFitAnalysis(
        match_score=validated["score"],
        summary=normalized.get("summary", ""),
        jd_requirements=[
            _to_requirement(r) for r in normalized.get("requirements", [])
        ],
        matched_strengths=[
            MatchItem(
                requirement=r["name"],
                resume_evidence=r.get("evidence_quote") or r.get("reasoning", ""),
                score=int(round(r["evidence_ratio"] * 100)),
            )
            for r in validated["validated_requirements"]
            if r["evidence_ratio"] > 0
        ][:12],
        gaps=[
            GapItem(
                requirement=g.get("requirement", ""),
                suggestion=g.get("suggestion", ""),
            )
            for g in normalized.get("gaps", [])
        ],
        resume_rewrites=[
            _dict_to_rewrite(r) for r in normalized.get("resume_rewrites", [])
        ],
        interview_questions=[
            _dict_to_question(q) for q in normalized.get("interview_questions", [])
        ],
        evidence=[*resume_evidence, *jd_evidence],
        score_breakdown=validated["score_breakdown"],
        core_requirements=validated["core_requirements"],
        bonus_requirements=validated["bonus_requirements"],
        risk_items=validated["risk_items"],
        model_used=settings.deepseek_model,
        fallback_used=False,
    )

    if validated["warnings"]:
        logger.info("Validator warnings: %s", "; ".join(validated["warnings"]))

    logger.info(
        "JobFit analysis completed | model=%s | score=%d | core=%d/%d",
        analysis.model_used,
        analysis.match_score,
        analysis.score_breakdown.core_matched,
        analysis.score_breakdown.core_total,
    )
    return analysis


def _format_context(evidence: list[Evidence]) -> str:
    lines = []
    for item in evidence:
        lines.append(f"[{item.source}#{item.chunk_id} score={item.score}]\n{item.text}")
    return "\n\n".join(lines)


def _to_requirement(req: dict) -> Requirement:
    return Requirement(
        name=req.get("name", "unknown"),
        priority=req.get("priority", "core"),
    )


def _dict_to_rewrite(d: dict) -> "ResumeRewrite":
    from app.schemas.jobfit import ResumeRewrite

    return ResumeRewrite(
        before=d.get("before", ""),
        after=d.get("after", ""),
        reason=d.get("reason", ""),
    )


def _dict_to_question(d: dict) -> "InterviewQuestion":
    from app.schemas.jobfit import InterviewQuestion

    return InterviewQuestion(
        question=d.get("question", ""),
        focus=d.get("focus", "interview"),
        difficulty=d.get("difficulty", "medium"),
    )
