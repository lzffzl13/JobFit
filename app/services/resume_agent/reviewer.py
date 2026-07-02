"""Rule-based reviewer for Resume Agent V1."""

from app.schemas.jobfit import RequirementAnalysis
from app.schemas.resume_agent import ClarifyingQuestion, ReviewDisposition, ReviewItem, UserFact


def review_requirements(
    requirement_analysis: list[RequirementAnalysis], facts: list[UserFact] | None = None
) -> list[ReviewItem]:
    """Review requirement analysis and decide whether to optimize, clarify, or skip."""
    facts = facts or []
    fact_map = _group_facts(facts)
    review_items: list[ReviewItem] = []

    for item in requirement_analysis:
        has_user_fact = item.requirement in fact_map

        if item.status == "strong_match":
            review_items.append(
                ReviewItem(
                    requirement=item.requirement,
                    disposition=ReviewDisposition.DIRECT_OPTIMIZE,
                    reason="已有明确证据，可直接优化表达。",
                    evidence=_combine_evidence(item.evidence, fact_map.get(item.requirement, [])),
                )
            )
            continue

        if has_user_fact:
            review_items.append(
                ReviewItem(
                    requirement=item.requirement,
                    disposition=ReviewDisposition.DIRECT_OPTIMIZE,
                    reason="用户补充了可用事实，可以生成候选改写。",
                    evidence=_combine_evidence(item.evidence, fact_map[item.requirement]),
                )
            )
            continue

        if item.level in ("required", "preferred"):
            review_items.append(
                ReviewItem(
                    requirement=item.requirement,
                    disposition=ReviewDisposition.CLARIFY,
                    reason=item.explanation or "当前证据不足，需要向用户补充确认。",
                    evidence=item.evidence,
                    question=ClarifyingQuestion(
                        requirement=item.requirement,
                        question=_build_question(item),
                        rationale=item.explanation or item.suggestion,
                    ),
                )
            )
            continue

        review_items.append(
            ReviewItem(
                requirement=item.requirement,
                disposition=ReviewDisposition.DO_NOT_WRITE,
                reason="当前不是高优先级要求，且缺少足够证据。",
                evidence=item.evidence,
            )
        )

    return review_items


def _build_question(item: RequirementAnalysis) -> str:
    if item.category == "experience":
        return f"关于“{item.requirement}”，你能补充具体年限、职责范围或业务场景吗？"
    if item.category == "project":
        return f"关于“{item.requirement}”，你能补充相关项目里你负责的模块、技术和结果吗？"
    if item.category == "education":
        return f"关于“{item.requirement}”，你能确认学历层次、专业或学校信息里还有什么可以补充吗？"
    return f"关于“{item.requirement}”，你是否有真实经历、项目或成果可以补充到简历里？"


def _group_facts(facts: list[UserFact]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for fact in facts:
        grouped.setdefault(fact.requirement, []).append(fact.content)
    return grouped


def _combine_evidence(evidence: str, extra_facts: list[str]) -> str:
    parts = [part for part in [evidence, *extra_facts] if part]
    return "；".join(parts)

