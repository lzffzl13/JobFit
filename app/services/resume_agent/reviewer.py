"""Rule-based reviewer for Resume Agent V1."""

from app.schemas.jobfit import RequirementAnalysis
from app.schemas.resume_agent import (
    ClarifyingQuestion,
    EvidenceSource,
    ReviewDisposition,
    ReviewItem,
    RiskLevel,
    UserFact,
    WritePolicy,
)


def review_requirements(
    requirement_analysis: list[RequirementAnalysis], facts: list[UserFact] | None = None
) -> list[ReviewItem]:
    """Review requirement analysis and decide whether to optimize, clarify, or skip."""
    facts = facts or []
    fact_map = _group_facts(facts)
    review_items: list[ReviewItem] = []

    for item in requirement_analysis:
        user_facts = fact_map.get(item.requirement, [])
        has_resume_evidence = bool(item.evidence.strip())
        has_user_fact = bool(user_facts)
        combined_evidence = _combine_evidence(item.evidence, user_facts)

        if item.status == "strong_match":
            review_items.append(
                ReviewItem(
                    requirement=item.requirement,
                    disposition=ReviewDisposition.DIRECT_OPTIMIZE,
                    write_policy=WritePolicy.SAFE_REWRITE,
                    reason="已有明确证据，可直接优化表达。",
                    evidence=combined_evidence,
                    evidence_source=_evidence_source(has_resume_evidence, has_user_fact),
                    confidence=0.9 if has_resume_evidence else 0.8,
                    missing_info=_missing_info(item, has_resume_evidence=has_resume_evidence),
                    risk_level=RiskLevel.LOW,
                    risk_reason="证据较明确，适合做表达强化，但仍需用户确认最终措辞。",
                    recommended_section=_infer_section(item),
                    suggested_angle=_suggested_angle(item),
                )
            )
            continue

        if has_user_fact:
            review_items.append(
                ReviewItem(
                    requirement=item.requirement,
                    disposition=ReviewDisposition.DIRECT_OPTIMIZE,
                    write_policy=WritePolicy.SAFE_REWRITE,
                    reason="用户补充了可用事实，可以生成候选改写。",
                    evidence=combined_evidence,
                    evidence_source=_evidence_source(has_resume_evidence, has_user_fact),
                    confidence=0.78 if has_resume_evidence else 0.68,
                    missing_info=[] if _has_enough_fact_detail(user_facts) else _missing_info(item, has_resume_evidence=True),
                    risk_level=RiskLevel.MEDIUM,
                    risk_reason="用户补充事实可用于候选表达，但需要避免写成未确认的主导、精通或量化结果。",
                    recommended_section=_infer_section(item),
                    suggested_angle=_suggested_angle(item),
                )
            )
            continue

        if item.level in ("required", "preferred"):
            missing_info = _missing_info(item, has_resume_evidence=has_resume_evidence)
            risk_level = RiskLevel.HIGH if item.level == "required" and item.score < 40 else RiskLevel.MEDIUM
            review_items.append(
                ReviewItem(
                    requirement=item.requirement,
                    disposition=ReviewDisposition.CLARIFY,
                    write_policy=WritePolicy.ASK_FOR_FACTS,
                    reason=item.explanation or "当前证据不足，需要向用户补充确认。",
                    evidence=item.evidence,
                    evidence_source=_evidence_source(has_resume_evidence, has_user_fact),
                    confidence=0.35 if has_resume_evidence else 0.15,
                    missing_info=missing_info,
                    risk_level=risk_level,
                    risk_reason=_risk_reason(item, missing_info),
                    recommended_section=_infer_section(item),
                    suggested_angle=_suggested_angle(item),
                    question=ClarifyingQuestion(
                        requirement=item.requirement,
                        question=_build_question(item, missing_info),
                        rationale=item.explanation or item.suggestion,
                        expected_evidence=missing_info,
                    ),
                )
            )
            continue

        review_items.append(
            ReviewItem(
                requirement=item.requirement,
                disposition=ReviewDisposition.DO_NOT_WRITE,
                write_policy=WritePolicy.DO_NOT_CLAIM,
                reason="当前不是高优先级要求，且缺少足够证据。",
                evidence=item.evidence,
                evidence_source=_evidence_source(has_resume_evidence, has_user_fact),
                confidence=0.1,
                missing_info=_missing_info(item, has_resume_evidence=has_resume_evidence),
                risk_level=RiskLevel.LOW,
                risk_reason="低优先级且证据不足，暂不建议主动写入简历。",
                recommended_section=_infer_section(item),
                suggested_angle=_suggested_angle(item),
            )
        )

    return review_items


def _build_question(item: RequirementAnalysis, missing_info: list[str]) -> str:
    detail_prompt = "、".join(missing_info[:3]) if missing_info else "真实经历、职责或结果"
    if item.category == "experience":
        return f"关于“{item.requirement}”，你实际做过多久、负责哪些模块？最好补充{detail_prompt}。"
    if item.category == "project":
        return f"关于“{item.requirement}”，有没有对应项目？请补充你负责的模块、使用方式和结果，尤其是{detail_prompt}。"
    if item.category == "education":
        return f"关于“{item.requirement}”，请确认学历层次、专业方向或课程/证书里是否有可写依据。"
    if item.category == "soft":
        return f"关于“{item.requirement}”，有没有真实协作、沟通或推进问题的例子？请补充场景、你的动作和结果。"
    return f"关于“{item.requirement}”，你是否真实使用或实践过？请补充{detail_prompt}，没有的话我们就把它保留为缺口。"


def _missing_info(item: RequirementAnalysis, has_resume_evidence: bool) -> list[str]:
    missing: list[str] = []
    if not has_resume_evidence:
        missing.append("是否真实做过")

    if item.category == "skill":
        missing.extend(["使用场景", "项目名称", "你的具体动作"])
    elif item.category == "experience":
        missing.extend(["具体年限", "职责范围", "业务场景"])
    elif item.category == "project":
        missing.extend(["项目名称", "负责模块", "技术使用方式", "结果"])
    elif item.category == "education":
        missing.extend(["学历层次", "专业相关性"])
    elif item.category == "soft":
        missing.extend(["具体场景", "你的行动", "结果"])
    else:
        missing.extend(["真实依据", "具体场景"])

    if item.level == "required" and item.score < 40:
        missing.append("能否支撑写入核心经历")

    return list(dict.fromkeys(missing))


def _risk_reason(item: RequirementAnalysis, missing_info: list[str]) -> str:
    if item.level == "required" and item.score < 40:
        return f"这是 JD 核心要求，但当前缺少直接证据；需要确认 {', '.join(missing_info[:3])}。"
    if item.status == "partial_match":
        return "已有部分关联，但证据强度不足，适合先追问再生成保守表达。"
    return "当前缺少可写依据，直接生成强表述会有编造风险。"


def _evidence_source(has_resume_evidence: bool, has_user_fact: bool) -> EvidenceSource:
    if has_resume_evidence and has_user_fact:
        return EvidenceSource.RESUME_AND_USER
    if has_resume_evidence:
        return EvidenceSource.RESUME
    if has_user_fact:
        return EvidenceSource.USER
    return EvidenceSource.NONE


def _infer_section(item: RequirementAnalysis) -> str:
    if item.category == "education":
        return "education"
    if item.category == "project":
        return "projects"
    if item.category == "skill":
        return "skills_or_projects"
    return "experience"


def _suggested_angle(item: RequirementAnalysis) -> str:
    if item.category == "skill":
        return f"突出 {item.requirement} 的实际使用场景和项目贡献"
    if item.category == "experience":
        return f"突出 {item.requirement} 对应的年限、职责和业务范围"
    if item.category == "project":
        return f"把 {item.requirement} 放到具体项目职责和结果里表达"
    if item.category == "education":
        return f"强调与 {item.requirement} 相关的学历、专业或课程背景"
    if item.category == "soft":
        return f"用具体协作场景体现 {item.requirement}"
    return f"围绕 {item.requirement} 补充真实依据"


def _has_enough_fact_detail(facts: list[str]) -> bool:
    combined = " ".join(facts)
    return len(combined) >= 20


def _group_facts(facts: list[UserFact]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for fact in facts:
        grouped.setdefault(fact.requirement, []).append(fact.content)
    return grouped


def _combine_evidence(evidence: str, extra_facts: list[str]) -> str:
    parts = [part for part in [evidence, *extra_facts] if part]
    return "；".join(parts)
