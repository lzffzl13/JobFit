"""Session orchestration for a fact-first resume workflow."""

from datetime import UTC, datetime
from functools import lru_cache
from uuid import uuid4

from fastapi import HTTPException

from app.core.config import settings
from app.schemas.resume_agent import (
    AgentMessage,
    MessageRole,
    ProposalStatus,
    ResumeAgentDecisionCreate,
    ResumeAgentMessageCreate,
    ResumeAgentSession,
    ResumeAgentSessionCreate,
    ResumeAgentState,
    UserFact,
)
from app.services.jobfit import analyze_job_fit
from app.services.resume_agent.analysis import refresh_analysis
from app.services.resume_agent.document import initialize_document, refresh_document_validity
from app.services.resume_agent.evidence import assess_requirements
from app.services.resume_agent.identity import (
    assign_requirement_ids,
    migrate_requirement_ids,
    requirement_key,
)
from app.services.resume_agent.proposal_generator import build_proposals
from app.services.resume_agent.repository import ResumeAgentRepository
from app.services.resume_agent.reviewer import review_requirements


class ResumeAgentOrchestrator:
    """Confirm facts before exposing rewrite proposals or accepting decisions."""

    def __init__(self, repository: ResumeAgentRepository):
        self.repository = repository

    async def create_session(self, payload: ResumeAgentSessionCreate) -> ResumeAgentSession:
        analysis = await analyze_job_fit(
            payload.resume_text, payload.jd_text, include_suggestions=False
        )
        assign_requirement_ids(analysis)
        assessments = await assess_requirements(
            analysis.requirement_analysis, payload.resume_text, payload.jd_text, []
        )
        analysis = refresh_analysis(analysis, assessments, updated=False)
        session = ResumeAgentSession(
            id=f"ras_{uuid4().hex[:12]}", state=ResumeAgentState.REVIEWING,
            summary="", resume_text=payload.resume_text, jd_text=payload.jd_text,
            analysis=analysis, initial_analysis=analysis.model_copy(deep=True),
            assessments=assessments,
        )
        self._rebuild(session)
        initialize_document(session)
        return self._record_summary(session)

    def get_session(self, session_id: str) -> ResumeAgentSession:
        session = self.repository.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Resume agent session not found.")
        session.messages = self.repository.list_messages(session_id)
        migrated = migrate_requirement_ids(session)
        if session.initial_analysis is None:
            # Old sessions have no verified evidence. Do not expose their pre-confirmation claims.
            session.initial_analysis = session.analysis.model_copy(deep=True)
            session.proposals = []
            session.proposal_history = []
            session.assessments = {}
            self._rebuild(session)
        elif migrated:
            self._rebuild(session)
        initialized = initialize_document(session)
        if migrated or initialized:
            self.repository.save_session(session)
        return session

    async def handle_message(self, session_id: str, payload: ResumeAgentMessageCreate) -> ResumeAgentSession:
        session = self.get_session(session_id)
        requirements = (session.initial_analysis or session.analysis).requirement_analysis
        known = {item.requirement_id: item for item in requirements}
        changed = set()
        for answer in payload.answers:
            question = next((q for q in session.pending_questions if q.id == answer.question_id), None)
            if answer.question_id and question is None:
                raise HTTPException(status_code=409, detail="该问题已更新，请刷新后回答。")
            requirement_id = question.requirement_id if question else answer.requirement_id
            if not requirement_id:
                matches = [item for item in requirements if item.requirement == answer.requirement]
                if len(matches) != 1:
                    raise HTTPException(status_code=400, detail="同名要求需要明确选择具体项目。")
                requirement_id = matches[0].requirement_id
            if requirement_id not in known:
                raise HTTPException(status_code=400, detail="补充内容必须对应当前岗位要求。")
            requirement = known[requirement_id].requirement
            if (answer.requirement and requirement != answer.requirement) or (
                question and answer.requirement_id and question.requirement_id != answer.requirement_id
            ):
                raise HTTPException(status_code=400, detail="回答与待确认问题不一致。")
            if answer.answer_type == "details" and not answer.answer.strip():
                raise HTTPException(status_code=400, detail="请填写实际情况，或选择没有做过、暂不回答。")
            session.facts.append(UserFact(
                requirement=requirement, requirement_id=requirement_id,
                content=answer.answer.strip(), answer_type=answer.answer_type,
                confirmed=False,
            ))
            changed.add(requirement_id)

        # Assess only revised facts and previously unverified legacy requirements.
        to_assess = [item for item in requirements
                     if item.requirement_id in changed or item.requirement_id not in session.assessments]
        if to_assess:
            session.assessments.update(await assess_requirements(
                to_assess, session.resume_text, session.jd_text, session.facts
            ))
        latest_facts = {requirement_key(fact): fact for fact in session.facts}
        for requirement in changed:
            latest_facts[requirement].confirmed = session.assessments[requirement].outcome in {
                "supported", "no_experience"
            }
        session.updated_at = datetime.now(UTC)
        self._rebuild(session)
        if payload.content.strip():
            session.messages.append(AgentMessage(
                role=MessageRole.USER, content=payload.content.strip(), created_at=session.updated_at
            ))
        return self._record_summary(session)

    async def apply_decision(self, session_id: str, payload: ResumeAgentDecisionCreate) -> ResumeAgentSession:
        session = self.get_session(session_id)
        if session.pending_questions or session.state == ResumeAgentState.NEEDS_CLARIFICATION:
            raise HTTPException(status_code=409, detail="请先确认待补充事实，再选择候选改写。")
        target = next((p for p in session.proposals if p.id == payload.proposal_id), None)
        if target is None:
            raise HTTPException(status_code=409, detail="该建议已更新，请刷新后确认当前版本。")
        if payload.version is not None and payload.version != target.version:
            raise HTTPException(status_code=409, detail="建议版本已变化，请重新阅读后确认。")
        if payload.decision not in {ProposalStatus.ACCEPTED, ProposalStatus.REJECTED}:
            raise HTTPException(status_code=422, detail="请选择采纳或暂不采纳。")
        target.status = payload.decision
        session.updated_at = datetime.now(UTC)
        self._rebuild(session)
        if payload.note.strip():
            session.messages.append(AgentMessage(
                role=MessageRole.USER, content=payload.note.strip(), created_at=session.updated_at
            ))
        return self._record_summary(session)

    def _rebuild(self, session: ResumeAgentSession) -> None:
        session.preview = None
        original = session.initial_analysis or session.analysis
        session.analysis = refresh_analysis(original, session.assessments, updated=bool(session.facts))
        session.analysis_overview = session.analysis.analysis_overview
        reviews = review_requirements(
            original.requirement_analysis, facts=session.facts, assessments=session.assessments
        )
        previous_questions = {requirement_key(q): q for q in session.pending_questions}
        for review in reviews:
            if review.question and requirement_key(review) in previous_questions:
                review.question.id = previous_questions[requirement_key(review)].id
        session.review_items = reviews
        session.pending_questions = [item.question for item in reviews if item.question is not None]
        previous = {requirement_key(p): p for p in session.proposal_history}
        previous.update({requirement_key(p): p for p in session.proposals})
        for review in reviews:
            if review.disposition != "direct_optimize" and requirement_key(review) in previous:
                # Revoked or insufficient evidence invalidates earlier consent even if restored later.
                previous[requirement_key(review)].evidence_basis = ""
                previous[requirement_key(review)].status = ProposalStatus.PROPOSED
        generated = build_proposals(reviews)
        session.proposals = self._preserve_proposal_decisions(list(previous.values()), generated)
        previous.update({requirement_key(p): p for p in session.proposals})
        session.proposal_history = list(previous.values())
        session.state = self._resolve_state(session.pending_questions, session.proposals)
        refresh_document_validity(session)
        if session.pending_questions:
            session.summary = (
                f"还有 {len(session.pending_questions)} 项关键事实待确认。"
                "请补充实际情况；没有做过或暂不回答也可以明确选择。确认后再生成候选改写。"
            )
        elif session.state == ResumeAgentState.COMPLETED:
            accepted = sum(p.status == ProposalStatus.ACCEPTED for p in session.proposals)
            session.summary = (
                f"本轮已完成，已采纳 {accepted} 条改写。"
                "请预览字段修改并确认应用，才能生成新的简历版本；缺口不生成经历。"
            )
        else:
            session.summary = f"关键事实已确认，已整理 {len(session.proposals)} 条候选改写，请逐条选择。"

    def _resolve_state(self, pending_questions, proposals) -> ResumeAgentState:
        if pending_questions:
            return ResumeAgentState.NEEDS_CLARIFICATION
        if any(p.status not in {ProposalStatus.ACCEPTED, ProposalStatus.REJECTED} for p in proposals):
            return ResumeAgentState.AWAITING_USER_CHOICE
        return ResumeAgentState.COMPLETED

    def _preserve_proposal_decisions(self, existing, generated):
        existing_by_requirement = {requirement_key(proposal): proposal for proposal in existing}
        for proposal in generated:
            previous = existing_by_requirement.get(requirement_key(proposal))
            if previous is None:
                continue
            same_content = all(getattr(previous, key) == getattr(proposal, key)
                               for key in ("after", "before", "evidence_basis", "source_section"))
            if same_content:
                proposal.id = previous.id
                proposal.status = previous.status
                proposal.version = previous.version
            else:
                # A new id also rejects decisions from old clients that do not send a version.
                proposal.version = previous.version + 1
        return generated

    def _record_summary(self, session: ResumeAgentSession) -> ResumeAgentSession:
        session.messages.append(AgentMessage(
            role=MessageRole.AGENT, content=session.summary, created_at=session.updated_at
        ))
        self.repository.save_session(session)
        return session


@lru_cache(maxsize=1)
def get_resume_agent_orchestrator() -> ResumeAgentOrchestrator:
    return ResumeAgentOrchestrator(ResumeAgentRepository(settings.resume_agent_db_path))
