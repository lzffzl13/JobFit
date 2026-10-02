"""Stable requirement identity and conservative migration of text-keyed sessions."""

from hashlib import sha256

from app.schemas.jobfit import JobFitAnalysis
from app.schemas.resume_agent import ResumeAgentSession


def requirement_key(item) -> str:
    return item.requirement_id or item.requirement


def assign_requirement_ids(analysis: JobFitAnalysis) -> bool:
    changed = False
    for index, item in enumerate(analysis.requirement_analysis):
        if not item.requirement_id:
            digest = sha256(f"{index}:{item.category}:{item.requirement}".encode()).hexdigest()[:16]
            item.requirement_id = f"req_{digest}"
            changed = True
    return changed


def migrate_requirement_ids(session: ResumeAgentSession) -> bool:
    original = session.initial_analysis or session.analysis
    if not assign_requirement_ids(original):
        return False
    grouped = {}
    for item in original.requirement_analysis:
        grouped.setdefault(item.requirement, []).append(item)
    valid_ids = {item.requirement_id for item in original.requirement_analysis}
    assessments = {}
    for key, value in session.assessments.items():
        if key in valid_ids:
            assessments[key] = value
        elif len(grouped.get(key, [])) == 1:
            assessments[grouped[key][0].requirement_id] = value
    session.assessments = assessments
    for fact in session.facts:
        matches = grouped.get(fact.requirement, [])
        if not fact.requirement_id and len(matches) == 1:
            fact.requirement_id = matches[0].requirement_id
        elif not fact.requirement_id:
            # A fact from a collided legacy key cannot be assigned to either requirement.
            fact.confirmed = False
    for attribute in ("proposals", "proposal_history", "pending_questions", "review_items"):
        migrated = []
        for record in getattr(session, attribute):
            matches = grouped.get(record.requirement, [])
            if not record.requirement_id and len(matches) == 1:
                record.requirement_id = matches[0].requirement_id
                record.category = matches[0].category
            if record.requirement_id in valid_ids:
                migrated.append(record)
        setattr(session, attribute, migrated)
    return True
