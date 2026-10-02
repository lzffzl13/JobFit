"""Build conservative rewrite candidates from verified source passages."""

from app.schemas.resume_agent import (
    EvidenceSource,
    ProposalTone,
    ReviewDisposition,
    ReviewItem,
    RewriteProposal,
)
from app.services.source_rewrite import rewrite_source_passage


def build_proposals(review_items: list[ReviewItem]) -> list[RewriteProposal]:
    # This guard also protects callers outside the session orchestrator.
    if any(item.disposition == ReviewDisposition.CLARIFY for item in review_items):
        return []
    proposals = []
    for item in review_items:
        if item.disposition != ReviewDisposition.DIRECT_OPTIMIZE or not item.evidence.strip():
            continue
        # Extractive rewriting keeps every action and qualifier traceable to the source.
        after = rewrite_source_passage(item.evidence)
        proposals.append(RewriteProposal(
            requirement=item.requirement, source_section=item.recommended_section,
            requirement_id=item.requirement_id, category=item.category,
            before=item.evidence, after=after, reason=item.reason, evidence_basis=item.evidence,
            confidence=item.confidence,
            tone=ProposalTone.CONSERVATIVE if item.evidence_source == EvidenceSource.USER
            or item.confidence < 0.85 else ProposalTone.BALANCED,
            safety_notes=["仅整理已确认事实，未补充新的职责或结果。"],
        ))
    return proposals
