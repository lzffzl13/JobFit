"""Translate verified evidence into workflow decisions."""

from app.schemas.jobfit import RequirementAnalysis
from app.schemas.resume_agent import (
    ClarifyingQuestion,
    EvidenceAssessment,
    EvidenceSource,
    ReviewDisposition,
    ReviewItem,
    RiskLevel,
    UserFact,
    WritePolicy,
)
from app.services.resume_agent.identity import requirement_key


def review_requirements(
    requirement_analysis: list[RequirementAnalysis], facts: list[UserFact] | None = None,
    assessments: dict[str, EvidenceAssessment] | None = None,
) -> list[ReviewItem]:
    assessments = assessments or {}
    fact_requirements = {requirement_key(fact) for fact in facts or []}
    reviews = []
    for item in requirement_analysis:
        assessment = assessments.get(requirement_key(item), EvidenceAssessment())
        supported = assessment.outcome == "supported"
        resolved_without_claim = assessment.outcome in {"no_experience", "skipped"}
        should_ask = not supported and not resolved_without_claim and item.level in {"required", "preferred"}
        missing = ["具体使用场景", "你实际完成的动作"] if should_ask else []
        question = ClarifyingQuestion(
            requirement=item.requirement,
            requirement_id=item.requirement_id, category=item.category,
            question=f"关于“{item.requirement}”：{assessment.followup}",
            rationale="确认真实经历后才能生成改写；明确没做过也算本项已确认。",
            expected_evidence=missing,
        ) if should_ask else None
        if supported:
            reason = "已有可核对的具体事实，仅根据这些事实整理表达。"
        elif assessment.outcome == "no_experience":
            reason = "已确认没有相关经历，保留缺口，不写入简历。"
        elif assessment.outcome == "skipped":
            reason = "本项暂不回答，保留缺口，不生成经历。"
        else:
            reason = "当前事实不足以支持具体经历，需要继续确认。"
        reviews.append(ReviewItem(
            requirement=item.requirement,
            requirement_id=item.requirement_id, category=item.category,
            disposition=(ReviewDisposition.DIRECT_OPTIMIZE if supported else
                         ReviewDisposition.CLARIFY if should_ask else ReviewDisposition.DO_NOT_WRITE),
            write_policy=(WritePolicy.SAFE_REWRITE if supported else
                          WritePolicy.ASK_FOR_FACTS if should_ask else WritePolicy.DO_NOT_CLAIM),
            reason=reason, evidence=assessment.evidence if supported else "",
            evidence_source=(EvidenceSource.USER if requirement_key(item) in fact_requirements else
                             EvidenceSource.RESUME) if supported else EvidenceSource.NONE,
            confidence=0.9 if supported and assessment.coverage == "full" else 0.65 if supported else 0.1,
            missing_info=missing,
            risk_level=RiskLevel.LOW if supported and assessment.coverage == "full" else RiskLevel.MEDIUM,
            risk_reason="仅保留已确认范围，不扩展为主导、精通或未提供的量化结果。",
            recommended_section={"skill": "skills_or_projects", "project": "projects",
                                 "education": "education"}.get(item.category, "experience"),
            suggested_angle="根据确认的事实整理简历表达。", question=question,
        ))
    return reviews
