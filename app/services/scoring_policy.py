from dataclasses import dataclass

from app.services.jd_parser import ParsedJD, SkillRequirement
from app.services.resume_evidence import (
    SkillEvidence,
    analyze_resume,
    evaluate_requirement,
    score_explicit_bonus,
    score_extension_bonus,
    score_extra_competitiveness,
)


@dataclass(frozen=True)
class ScoringOutcome:
    score: int
    raw_score: float
    core_score: float
    bonus_jd: float
    bonus_extra: float
    final_cap: int
    cap_reasons: list[str]
    core_evidence: list[SkillEvidence]
    bonus_matched: int
    bonus_total: int
    extra_strengths: list[str]
    evidence_notes: list[str]
    matched_requirements: list[SkillRequirement]
    missing_requirements: list[SkillRequirement]
    risk_items: list[str]


def score_resume_against_jd(parsed_jd: ParsedJD, resume_text: str) -> ScoringOutcome:
    profile = analyze_resume(resume_text)
    core_evidence = [evaluate_requirement(requirement, profile) for requirement in parsed_jd.core_requirements]
    core_score = _score_core(core_evidence)

    if parsed_jd.has_explicit_bonus:
        bonus_jd, bonus_matched, bonus_notes = score_explicit_bonus(parsed_jd.bonus_requirements, profile)
        bonus_total = len(parsed_jd.bonus_requirements)
    else:
        bonus_jd, bonus_matched, extension_notes = score_extension_bonus(profile, parsed_jd.core_requirements)
        bonus_total = 5
        bonus_notes = [f"无明确 JD 加分项，按技术扩展广度给分：{', '.join(extension_notes) or '暂无'}"]

    bonus_extra, extra_items = score_extra_competitiveness(profile)
    extra_strengths = [f"{item.name}: {item.evidence}" for item in extra_items]

    raw_score = round(core_score + bonus_jd + bonus_extra, 2)
    final_cap, cap_reasons = _direction_cap(parsed_jd, profile, core_evidence)
    final_score = min(raw_score, final_cap)

    matched_requirements = [
        evidence.requirement for evidence in core_evidence if evidence.ratio > 0
    ]
    missing_requirements = [
        evidence.requirement for evidence in core_evidence if evidence.ratio == 0
    ]
    for requirement in parsed_jd.bonus_requirements:
        bonus_evidence = evaluate_requirement(requirement, profile)
        if bonus_evidence.ratio > 0:
            matched_requirements.append(requirement)
        else:
            missing_requirements.append(requirement)

    evidence_notes = [
        *bonus_notes,
        *[
            f"{evidence.requirement.label}: {evidence.level}"
            for evidence in core_evidence
            if evidence.ratio < 0.85
        ],
        *cap_reasons,
    ]
    risk_items = [
        f"{evidence.requirement.label}: {evidence.level}"
        for evidence in core_evidence
        if evidence.ratio < 0.85
    ][:6]

    return ScoringOutcome(
        score=max(0, min(100, int(round(final_score)))),
        raw_score=raw_score,
        core_score=round(core_score, 2),
        bonus_jd=round(bonus_jd, 2),
        bonus_extra=round(bonus_extra, 2),
        final_cap=final_cap,
        cap_reasons=cap_reasons,
        core_evidence=core_evidence,
        bonus_matched=bonus_matched,
        bonus_total=bonus_total,
        extra_strengths=extra_strengths,
        evidence_notes=evidence_notes,
        matched_requirements=matched_requirements,
        missing_requirements=missing_requirements,
        risk_items=risk_items,
    )


def _score_core(core_evidence: list[SkillEvidence]) -> float:
    if not core_evidence:
        return 0.0
    ratio_sum = sum(evidence.ratio for evidence in core_evidence)
    return ratio_sum / len(core_evidence) * 70


def _direction_cap(
    parsed_jd: ParsedJD,
    profile,
    core_evidence: list[SkillEvidence],
) -> tuple[int, list[str]]:
    caps: list[tuple[int, str]] = []

    if not profile.has_language_evidence:
        caps.append((45, "无编程语言证据 → 45"))

    if parsed_jd.role_type in {"backend", "java_backend"} and not profile.has_backend_evidence:
        caps.append((55, "无后端/Web/接口开发证据 → 55"))

    has_database_requirement = any(
        requirement.dimension == "database_sql" for requirement in parsed_jd.core_requirements
    )
    if has_database_requirement and not profile.has_database_evidence:
        caps.append((70, "无数据库/SQL使用证据 → 70"))

    if core_evidence:
        zero_count = sum(1 for evidence in core_evidence if evidence.ratio == 0)
        if zero_count / len(core_evidence) >= 0.5:
            caps.append((75, "核心技能项中 >=50% 为 0% → 75"))

    if profile.is_non_technical and not profile.has_technical_project:
        caps.append((55, "非技术简历且无技术项目 → 55"))

    if profile.all_work_non_dev_without_backend_project:
        caps.append((65, "工作经历全部为非开发岗位且无后端项目 → 65"))

    if _looks_like_keyword_stuffing(profile, core_evidence):
        caps.append((70, "核心技能主要停留在技能栏，缺少项目证据 → 70"))

    if not caps:
        return 100, []
    final_cap = min(cap for cap, _reason in caps)
    return final_cap, [reason for cap, reason in caps if cap == final_cap or cap <= 70]


def _looks_like_keyword_stuffing(profile, core_evidence: list[SkillEvidence]) -> bool:
    if profile.has_technical_project:
        return False
    if not core_evidence:
        return False
    weak_hits = sum(1 for evidence in core_evidence if 0 < evidence.ratio <= 0.30)
    return weak_hits >= max(2, len(core_evidence) // 2)
