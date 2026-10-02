"""Rebuild the full analysis consistently after evidence changes."""

from app.schemas.jobfit import Evidence, GapItem, JobFitAnalysis, MatchDetail, MatchItem
from app.schemas.resume_agent import EvidenceAssessment
from app.services.matcher import summarize_matches
from app.services.resume_agent.identity import requirement_key


def refresh_analysis(
    original: JobFitAnalysis, assessments: dict[str, EvidenceAssessment], *, updated: bool,
) -> JobFitAnalysis:
    details = []
    for item in original.requirement_analysis:
        assessment = assessments.get(requirement_key(item), EvidenceAssessment())
        score = {"full": 1.0, "partial": 0.5}.get(assessment.coverage, 0.0)
        if assessment.outcome != "supported":
            score = 0.0
        details.append(MatchDetail(
            requirement=item.requirement, requirement_id=item.requirement_id,
            category=item.category, level=item.level,
            matched=score >= 0.85, match_score=score,
            evidence=assessment.evidence if score else "",
            method="confirmed_evidence" if score else "unconfirmed",
        ))
    result = summarize_matches(details)
    analysis = original.model_copy(deep=True)
    analysis.match_score = result.total_score
    analysis.score_breakdown = result.score_breakdown
    analysis.requirement_analysis = result.requirement_analyses
    analysis.analysis_overview = result.analysis_overview
    analysis.core_requirements = result.core_requirements
    analysis.bonus_requirements = result.bonus_requirements
    analysis.risk_items = result.risk_items
    analysis.risk_details = result.risk_details
    analysis.matched_strengths = [MatchItem(
        requirement=item.requirement, resume_evidence=item.evidence, score=round(item.match_score * 100)
    ) for item in result.matched]
    analysis.gaps = [GapItem(requirement=item.requirement, suggestion=item.suggestion) for item in result.gaps]
    analysis.evidence = [Evidence(source=item.requirement, text=item.evidence, score=item.match_score)
                         for item in details if item.evidence]
    # Legacy one-shot suggestions must not bypass the clarification/confirmation workflow.
    analysis.resume_rewrites = []
    analysis.interview_questions = []
    stage = "补充后" if updated else "初步"
    analysis.summary = f"{stage}分析：根据目前可核对的事实，匹配度 {result.total_score}/100；未确认项保留为缺口。"
    return analysis
