from dataclasses import dataclass

from app.schemas.jobfit import (
    GapItem,
    InterviewQuestion,
    JobFitAnalysis,
    MatchItem,
    Requirement,
    ResumeRewrite,
    ScoreBreakdown,
)
from app.services.jd_parser import (
    BONUS_POINTS_TOTAL,
    CORE_POINTS_TOTAL,
    EXTRA_POINTS_TOTAL,
    ParsedJD,
    SkillRequirement,
    parse_jd,
)
from app.services.resume_evidence import SkillEvidence, evaluate_requirement
from app.services.scoring_policy import ScoringOutcome, score_resume_against_jd


@dataclass(frozen=True)
class EvaluationResult:
    score: int
    score_breakdown: ScoreBreakdown
    matched: list[SkillRequirement]
    missing: list[SkillRequirement]
    core_requirements: list[str]
    bonus_requirements: list[str]
    extra_strengths: list[str]
    risk_items: list[str]
    outcome: ScoringOutcome | None = None


def summarize_jd_semantics(jd_text: str) -> str:
    parsed = parse_jd(jd_text)
    if not parsed.requirements:
        return "No stable skill requirements were extracted."

    lines = [f"Role type: {parsed.role_type}. Explicit bonus: {parsed.has_explicit_bonus}."]
    for requirement in parsed.requirements[:14]:
        options = ", ".join(requirement.options) or requirement.name
        lines.append(
            "- "
            f"{requirement.name}: type={requirement.requirement_type}, "
            f"priority={requirement.priority}, required_count={requirement.required_count}, "
            f"options=[{options}], source={requirement.clause[:120]}"
        )
    return "\n".join(lines)


def extract_skill_requirements(jd_text: str) -> list[SkillRequirement]:
    return parse_jd(jd_text).requirements


def local_fallback_analysis(resume_text: str, jd_text: str) -> JobFitAnalysis:
    parsed = parse_jd(jd_text)
    evaluation = evaluate_resume_against_requirements(parsed.requirements, resume_text, parsed)
    outcome = evaluation.outcome

    strengths = []
    if outcome:
        strengths.extend(_strengths_from_core_evidence(outcome.core_evidence))
    for extra in evaluation.extra_strengths[:6]:
        strengths.append(MatchItem(requirement="简历额外竞争力", resume_evidence=extra, score=88))

    return JobFitAnalysis(
        match_score=evaluation.score,
        summary=_summary(evaluation),
        jd_requirements=[_to_schema_requirement(req) for req in parsed.requirements],
        matched_strengths=strengths[:12],
        gaps=[
            GapItem(requirement=requirement.label, suggestion=_gap_suggestion(requirement))
            for requirement in evaluation.missing[:8]
        ],
        resume_rewrites=_fallback_rewrites(evaluation),
        interview_questions=_fallback_questions(evaluation),
        evidence=[],
        score_breakdown=evaluation.score_breakdown,
        core_requirements=evaluation.core_requirements,
        bonus_requirements=evaluation.bonus_requirements,
        extra_strengths=evaluation.extra_strengths,
        risk_items=evaluation.risk_items,
        model_used="local-fallback",
        fallback_used=True,
    )


def evaluate_resume_against_requirements(
    requirements: list[SkillRequirement],
    resume_text: str,
    parsed_jd: ParsedJD | None = None,
) -> EvaluationResult:
    parsed = parsed_jd or _parsed_from_requirements(requirements)
    outcome = score_resume_against_jd(parsed, resume_text)
    score_breakdown = _score_breakdown(parsed, outcome)
    return EvaluationResult(
        score=outcome.score,
        score_breakdown=score_breakdown,
        matched=outcome.matched_requirements,
        missing=outcome.missing_requirements,
        core_requirements=[req.label for req in parsed.core_requirements],
        bonus_requirements=[req.label for req in parsed.bonus_requirements],
        extra_strengths=outcome.extra_strengths,
        risk_items=outcome.risk_items,
        outcome=outcome,
    )


def evidence_for_requirement(
    requirement: SkillRequirement,
    resume_text: str,
) -> SkillEvidence:
    from app.services.resume_evidence import analyze_resume

    return evaluate_requirement(requirement, analyze_resume(resume_text))


def _parsed_from_requirements(requirements: list[SkillRequirement]) -> ParsedJD:
    core = [requirement for requirement in requirements if not requirement.is_bonus]
    bonus = [requirement for requirement in requirements if requirement.is_bonus]
    return ParsedJD(
        role_type="backend" if any(req.dimension in {"web_framework", "api_development"} for req in core) else "general",
        core_requirements=core,
        bonus_requirements=bonus,
        has_explicit_bonus=bool(bonus),
        clauses=[req.clause for req in requirements],
    )


def _score_breakdown(parsed: ParsedJD, outcome: ScoringOutcome) -> ScoreBreakdown:
    core_matched = sum(1 for evidence in outcome.core_evidence if evidence.ratio > 0)
    core_total = len(outcome.core_evidence)
    core_detail = {
        evidence.requirement.label: _core_detail(evidence)
        for evidence in outcome.core_evidence
    }
    return ScoreBreakdown(
        core_matched=core_matched,
        core_total=core_total,
        bonus_matched=outcome.bonus_matched,
        bonus_total=outcome.bonus_total,
        core_ratio=round(core_matched / core_total, 2) if core_total else 0,
        bonus_ratio=round(outcome.bonus_matched / outcome.bonus_total, 2) if outcome.bonus_total else 0,
        core_score=outcome.core_score,
        bonus_jd=outcome.bonus_jd,
        bonus_extra=outcome.bonus_extra,
        raw_score=outcome.raw_score,
        final_cap=outcome.final_cap,
        cap_reasons=outcome.cap_reasons,
        core_detail=core_detail,
        evidence_notes=outcome.evidence_notes,
        core_points=int(round(outcome.core_score)),
        core_points_total=CORE_POINTS_TOTAL,
        bonus_points=int(round(outcome.bonus_jd)),
        bonus_points_total=BONUS_POINTS_TOTAL if parsed.has_explicit_bonus else 10,
        extra_points=int(round(outcome.bonus_extra)),
        extra_points_total=EXTRA_POINTS_TOTAL,
        extra_matched=len(outcome.extra_strengths),
        extra_total=5,
    )


def _core_detail(evidence: SkillEvidence) -> str:
    if evidence.evidence:
        return f"{evidence.level}: {evidence.evidence}"
    return evidence.level


def _strengths_from_core_evidence(core_evidence: list[SkillEvidence]) -> list[MatchItem]:
    strengths = []
    for evidence in core_evidence:
        if evidence.ratio <= 0:
            continue
        strengths.append(
            MatchItem(
                requirement=evidence.requirement.label,
                resume_evidence=evidence.evidence or evidence.level,
                score=int(round(evidence.ratio * 100)),
            )
        )
    return strengths


def _summary(evaluation: EvaluationResult) -> str:
    breakdown = evaluation.score_breakdown
    cap_text = f" 最终上限 {breakdown.final_cap}。" if breakdown.final_cap < 100 else ""
    return (
        f"总分 {evaluation.score}/100。岗位核心匹配 {breakdown.core_score:.2f}/{CORE_POINTS_TOTAL}，"
        f"JD 加分项 {breakdown.bonus_jd:.2f}/{breakdown.bonus_points_total}，"
        f"简历额外竞争力 {breakdown.bonus_extra:.2f}/{EXTRA_POINTS_TOTAL}。"
        f"原始分 {breakdown.raw_score:.2f}。{cap_text}"
    )


def _to_schema_requirement(requirement: SkillRequirement) -> Requirement:
    return Requirement(
        name=requirement.name,
        category=requirement.category,
        evidence=requirement.clause[:180],
        type=requirement.requirement_type,
        options=list(requirement.options),
        required_count=requirement.required_count,
        priority=requirement.priority,
        weight=requirement.weight,
    )


def _gap_suggestion(requirement: SkillRequirement) -> str:
    if requirement.is_bonus:
        return f"这是加分项，可补充 {requirement.label} 的实际使用场景。"
    if requirement.requirement_type == "alternative":
        return f"补充 {requirement.label} 中任一方向的项目证据即可，最好写清具体实现。"
    return f"补充能证明 {requirement.label} 的项目场景、代码实现或学习实践。"


def _fallback_rewrites(evaluation: EvaluationResult) -> list[ResumeRewrite]:
    rewrites = [
        ResumeRewrite(
            before="项目描述只罗列技能，缺少可验证的使用场景。",
            after="按 JD 核心项重写项目：说明语言、框架、接口、数据库/SQL，以及具体实现细节、部署或测试。",
            reason="新评分会区分技能栏提到和项目中真实使用，高分必须有项目证据。",
        )
    ]
    if evaluation.score_breakdown.final_cap < 100:
        rewrites.append(
            ResumeRewrite(
                before="简历方向或核心证据不足，触发岗位方向上限。",
                after="补充与岗位直接相关的后端/Web/API/数据库项目，避免用不相关运营、市场或泛化经历支撑技术岗位。",
                reason="硬性上限用于防止不相关简历被泛化亮点抬高。",
            )
        )
    return rewrites


def _fallback_questions(evaluation: EvaluationResult) -> list[InterviewQuestion]:
    questions = [
        InterviewQuestion(
            question="请结合一个项目说明你如何设计后端接口、数据库表和错误处理。",
            focus="核心项目证据",
            difficulty="medium",
        ),
        InterviewQuestion(
            question="你简历中最高证据等级的技能是哪一项？请说明具体实现细节。",
            focus="证据强度",
            difficulty="medium",
        ),
    ]
    if "MySQL/SQL" in evaluation.core_requirements:
        questions.append(
            InterviewQuestion(
                question="请说明一次你用 SQL 或 ORM 完成查询、插入、更新、删除的实现细节。",
                focus="MySQL/SQL",
                difficulty="medium",
            )
        )
    return questions
