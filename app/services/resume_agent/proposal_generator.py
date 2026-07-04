"""Proposal generation for Resume Agent V1."""

from app.schemas.resume_agent import (
    EvidenceSource,
    ProposalTone,
    ReviewDisposition,
    ReviewItem,
    RewriteProposal,
    RiskLevel,
)


def build_proposals(review_items: list[ReviewItem]) -> list[RewriteProposal]:
    """Build user-reviewable rewrite proposals from review items."""
    proposals: list[RewriteProposal] = []

    for item in review_items:
        if item.disposition != ReviewDisposition.DIRECT_OPTIMIZE:
            continue

        before = item.evidence or "当前简历没有足够具体地体现该要求。"
        after = _build_after_text(item)
        proposals.append(
            RewriteProposal(
                requirement=item.requirement,
                source_section=item.recommended_section,
                before=before,
                after=after,
                reason=item.reason,
                evidence_basis=item.evidence or "基于用户已确认的真实经历。",
                confidence=item.confidence,
                tone=_proposal_tone(item),
                safety_notes=_safety_notes(item),
                needs_user_confirmation=True,
            )
        )

    return proposals


def _build_after_text(item: ReviewItem) -> str:
    evidence = _normalize_evidence(item.evidence)
    requirement = item.requirement

    if item.recommended_section == "skills_or_projects":
        return _skill_bullet(requirement, evidence, item.evidence_source)
    if item.recommended_section == "projects":
        return _project_bullet(requirement, evidence)
    if item.recommended_section == "education":
        return _education_bullet(requirement, evidence)
    return _experience_bullet(requirement, evidence)


def _skill_bullet(requirement: str, evidence: str, evidence_source: EvidenceSource) -> str:
    if evidence_source == EvidenceSource.USER:
        return f"补充候选：在项目中实际使用 {requirement}，结合用户补充事实说明使用场景、负责动作和交付结果。"
    if evidence:
        return f"基于 {evidence}，可改写为：在后端项目中使用 {requirement} 支撑核心功能开发，并结合具体模块说明技术落地场景。"
    return f"补充候选：围绕 {requirement} 写清楚真实使用场景、负责内容和项目结果。"


def _experience_bullet(requirement: str, evidence: str) -> str:
    if evidence:
        return f"基于 {evidence}，可改写为：围绕 {requirement} 承担后端开发相关工作，负责具体模块实现、问题定位和交付支持。"
    return f"补充候选：围绕 {requirement} 写清楚真实年限、职责范围、业务场景和交付内容。"


def _project_bullet(requirement: str, evidence: str) -> str:
    if evidence:
        return f"基于 {evidence}，可改写为：在相关项目中负责 {requirement} 相关模块，说明技术方案、个人贡献和最终效果。"
    return f"补充候选：选择一个真实项目，写清楚 {requirement} 对应模块、使用技术、你的职责和结果。"


def _education_bullet(requirement: str, evidence: str) -> str:
    if evidence:
        return f"基于 {evidence}，可在教育背景中突出与 {requirement} 相关的专业、课程或证书信息。"
    return f"补充候选：如有真实依据，可在教育背景中补充 {requirement} 相关专业、课程或证书。"


def _proposal_tone(item: ReviewItem) -> ProposalTone:
    if item.confidence >= 0.85 and item.risk_level == RiskLevel.LOW:
        return ProposalTone.STRONG
    if item.confidence < 0.7 or item.risk_level == RiskLevel.MEDIUM:
        return ProposalTone.CONSERVATIVE
    return ProposalTone.BALANCED


def _safety_notes(item: ReviewItem) -> list[str]:
    notes = ["最终采用前需要用户确认事实准确。"]
    if item.risk_level != RiskLevel.LOW:
        notes.append(item.risk_reason)
    if item.evidence_source == EvidenceSource.USER:
        notes.append("该建议依赖用户补充信息，不能自动视为原简历已包含内容。")
    if item.missing_info:
        notes.append(f"仍需留意：{'、'.join(item.missing_info[:3])}。")
    return list(dict.fromkeys(note for note in notes if note))


def _normalize_evidence(evidence: str) -> str:
    return "；".join(part.strip() for part in evidence.split("；") if part.strip())
