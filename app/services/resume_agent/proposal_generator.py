"""Proposal generation for Resume Agent V1."""

from app.schemas.resume_agent import ReviewDisposition, ReviewItem, RewriteProposal


def build_proposals(review_items: list[ReviewItem]) -> list[RewriteProposal]:
    """Build user-reviewable rewrite proposals from review items."""
    proposals: list[RewriteProposal] = []

    for item in review_items:
        if item.disposition != ReviewDisposition.DIRECT_OPTIMIZE:
            continue

        before = item.evidence or "当前简历没有足够具体地体现该要求。"
        after = _build_after_text(item.requirement, item.evidence)
        proposals.append(
            RewriteProposal(
                requirement=item.requirement,
                source_section=_infer_section(item.requirement),
                before=before,
                after=after,
                reason=item.reason,
                evidence_basis=item.evidence or "基于用户已确认的真实经历。",
                needs_user_confirmation=True,
            )
        )

    return proposals


def _build_after_text(requirement: str, evidence: str) -> str:
    if evidence:
        return f"建议把“{evidence}”进一步改写为突出 {requirement} 场景、职责和结果的表达。"
    return f"建议围绕 {requirement} 补充更具体的职责、项目场景和结果表达。"


def _infer_section(requirement: str) -> str:
    if "学历" in requirement or "专业" in requirement:
        return "education"
    if "经验" in requirement:
        return "experience"
    if "项目" in requirement:
        return "projects"
    return "experience"

