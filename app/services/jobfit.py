import logging

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.jobfit import (
    Evidence,
    GapItem,
    JobFitAnalysis,
    MatchItem,
    Requirement,
    ScoreBreakdown,
)
from app.services.llm import (
    DeepSeekClient,
    evaluate_resume_against_requirements,
    extract_skill_requirements,
    local_fallback_analysis,
    normalize_llm_payload,
    summarize_jd_semantics,
)
from app.services.retriever import chunk_text, retrieve_evidence

logger = logging.getLogger(__name__)


async def analyze_job_fit(resume_text: str, jd_text: str) -> JobFitAnalysis:
    resume_chunks = chunk_text(resume_text, source="resume")
    jd_chunks = chunk_text(jd_text, source="jd")

    jd_query = jd_text[:4000]
    resume_evidence = retrieve_evidence(jd_query, resume_chunks, top_k=8)
    jd_evidence = retrieve_evidence(resume_text[:4000], jd_chunks, top_k=5) or jd_chunks[:5]

    resume_context = _format_context(resume_evidence)
    jd_context = _format_context(jd_evidence)
    jd_semantics = summarize_jd_semantics(jd_text)
    requirement_details = extract_skill_requirements(jd_text)

    try:
        raw = await DeepSeekClient().analyze(
            resume_context=resume_context[: settings.max_context_chars],
            jd_context=jd_context[: settings.max_context_chars],
            jd_semantics=jd_semantics,
        )
        normalized = normalize_llm_payload(raw)
        analysis = JobFitAnalysis.model_validate(
            {
                **normalized,
                "evidence": [*resume_evidence, *jd_evidence],
                "model_used": settings.deepseek_model,
                "fallback_used": False,
            }
        )
        logger.info(
            "JobFit analysis completed | model=%s | fallback=false | score=%d",
            analysis.model_used,
            analysis.match_score,
        )
    except (RuntimeError, ValidationError, KeyError, ValueError, httpx.HTTPError) as exc:
        logger.warning(
            "DeepSeek analysis failed, using local fallback | error_type=%s | error=%s",
            type(exc).__name__,
            exc,
        )
        analysis = local_fallback_analysis(resume_text, jd_text)
        analysis.evidence = [*resume_evidence, *jd_evidence]
        logger.info(
            "JobFit analysis completed | model=%s | fallback=true | score=%d",
            analysis.model_used,
            analysis.match_score,
        )

    _enrich_analysis(analysis, resume_text, requirement_details)
    logger.info(
        "JobFit analysis enriched | model=%s | fallback=%s | score=%d | core=%d/%d",
        analysis.model_used,
        str(analysis.fallback_used).lower(),
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


def _enrich_analysis(analysis: JobFitAnalysis, resume_text: str, requirement_details: list) -> None:
    if not requirement_details:
        analysis.score_breakdown = ScoreBreakdown()
        analysis.core_requirements = []
        analysis.bonus_requirements = []
        analysis.risk_items = []
        return

    evaluation = evaluate_resume_against_requirements(requirement_details, resume_text)
    analysis.match_score = evaluation.score
    analysis.score_breakdown = evaluation.score_breakdown
    analysis.jd_requirements = [_requirement_to_schema(requirement) for requirement in requirement_details]
    analysis.core_requirements = evaluation.core_requirements
    analysis.bonus_requirements = evaluation.bonus_requirements
    analysis.extra_strengths = evaluation.extra_strengths
    analysis.risk_items = evaluation.risk_items
    analysis.summary = _deterministic_summary(analysis, evaluation)
    analysis.matched_strengths = _merge_matched_strengths(
        analysis.matched_strengths,
        evaluation,
        resume_text,
    )
    analysis.gaps = _normalize_gaps(analysis.gaps, evaluation)


def _requirement_to_schema(requirement) -> Requirement:
    return Requirement(
        name=requirement.label,
        category=requirement.category,
        evidence=requirement.clause[:180],
        type=requirement.requirement_type,
        options=list(requirement.options),
        required_count=requirement.required_count,
        priority=requirement.priority,
        weight=requirement.weight,
    )


def _deterministic_summary(analysis: JobFitAnalysis, evaluation) -> str:
    prefix = (
        f"总分 {evaluation.score}/100。岗位核心匹配 "
        f"{evaluation.score_breakdown.core_points}/{evaluation.score_breakdown.core_points_total}，"
        f"JD 加分项 {evaluation.score_breakdown.bonus_points}/"
        f"{evaluation.score_breakdown.bonus_points_total}，"
        f"简历额外竞争力 {evaluation.score_breakdown.extra_points}/"
        f"{evaluation.score_breakdown.extra_points_total}。"
    )
    if analysis.fallback_used or not analysis.summary:
        return prefix
    return f"{prefix} {analysis.summary}"


def _merge_matched_strengths(
    existing_strengths: list[MatchItem],
    evaluation,
    resume_text: str,
) -> list[MatchItem]:
    merged = list(existing_strengths)
    existing_names = {item.requirement.lower() for item in merged}

    for requirement in evaluation.matched:
        if requirement.label.lower() in existing_names:
            continue
        merged.append(
            MatchItem(
                requirement=requirement.label,
                resume_evidence=_find_resume_evidence(resume_text, requirement),
                score=90 if not requirement.is_bonus else 84,
            )
        )
        existing_names.add(requirement.label.lower())

    for extra in evaluation.extra_strengths:
        if extra.lower() in existing_names:
            continue
        merged.append(
            MatchItem(
                requirement="简历额外竞争力",
                resume_evidence=extra,
                score=88,
            )
        )
        existing_names.add(extra.lower())

    return merged[:12]


def _find_resume_evidence(resume_text: str, requirement) -> str:
    terms = [requirement.label, *requirement.options]
    if requirement.dimension == "api_development":
        terms.extend(["接口", "API", "RESTful", "CRUD", "用户", "文章", "评论"])
    if requirement.dimension == "sql_database":
        terms.extend(["MySQL", "SQL", "SQLAlchemy", "数据库", "CRUD"])
    if requirement.dimension == "deployment":
        terms.extend(["Docker", "部署"])

    for line in resume_text.splitlines():
        lower_line = line.lower()
        if any(term and term.lower() in lower_line for term in terms):
            return line.strip()[:240]
    return f"简历中出现了与 {requirement.label} 相关的项目或技能描述。"


def _normalize_gaps(existing_gaps: list, evaluation) -> list:
    normalized: list[GapItem] = []
    matched_group_options = [
        {option.lower() for option in requirement.options}
        for requirement in evaluation.matched
        if len(requirement.options) > 1
    ]
    safe_gap_names = {requirement.label.lower() for requirement in evaluation.missing}

    for gap in existing_gaps:
        gap_name = gap.requirement.lower()
        if any(gap_name in options for options in matched_group_options):
            continue
        if gap_name not in safe_gap_names:
            continue
        normalized.append(gap)

    existing_names = {gap.requirement.lower() for gap in normalized}
    for requirement in evaluation.missing:
        if requirement.label.lower() not in existing_names:
            normalized.append(
                GapItem(
                    requirement=requirement.label,
                    suggestion=(
                        f"补充与 {requirement.label} 相关的项目细节或实践经历。"
                        if not requirement.is_bonus
                        else f"如要提高竞争力，可补充 {requirement.label} 相关实践。"
                    ),
                )
            )
    return normalized[:8]
