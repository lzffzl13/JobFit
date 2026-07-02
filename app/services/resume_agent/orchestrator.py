"""Session orchestration for Resume Agent V1."""

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
    ReviewItem,
    UserFact,
)
from app.services.jobfit import analyze_job_fit
from app.services.resume_agent.proposal_generator import build_proposals
from app.services.resume_agent.repository import ResumeAgentRepository
from app.services.resume_agent.reviewer import review_requirements


class ResumeAgentOrchestrator:
    """Coordinate review, clarification, and proposal steps for resume sessions."""

    def __init__(self, repository: ResumeAgentRepository):
        self.repository = repository

    async def create_session(self, payload: ResumeAgentSessionCreate) -> ResumeAgentSession:
        analysis = await analyze_job_fit(payload.resume_text, payload.jd_text)
        review_items = review_requirements(analysis.requirement_analysis)
        pending_questions = [item.question for item in review_items if item.question is not None][:5]
        proposals = build_proposals(review_items)
        state = self._resolve_state(pending_questions, proposals)
        summary = self._build_summary(review_items, pending_questions, proposals)

        session = ResumeAgentSession(
            id=f"ras_{uuid4().hex[:12]}",
            state=state,
            summary=summary,
            resume_text=payload.resume_text,
            jd_text=payload.jd_text,
            analysis=analysis,
            analysis_overview=analysis.analysis_overview,
            review_items=review_items,
            pending_questions=pending_questions,
            proposals=proposals,
        )

        self.repository.save_session(session)
        self.repository.add_message(
            session.id,
            AgentMessage(role=MessageRole.AGENT, content=summary),
        )
        session.messages = self.repository.list_messages(session.id)
        self.repository.save_session(session)
        return session

    def get_session(self, session_id: str) -> ResumeAgentSession:
        session = self.repository.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Resume agent session not found.")
        session.messages = self.repository.list_messages(session_id)
        return session

    async def handle_message(self, session_id: str, payload: ResumeAgentMessageCreate) -> ResumeAgentSession:
        session = self.get_session(session_id)
        now = datetime.now(UTC)

        if payload.content.strip():
            self.repository.add_message(
                session_id,
                AgentMessage(role=MessageRole.USER, content=payload.content.strip(), created_at=now),
            )

        updated_questions = list(session.pending_questions)
        facts = list(session.facts)
        new_fact_requirements: list[str] = []

        for answer in payload.answers:
            question = next((q for q in updated_questions if q.id == answer.question_id), None)
            requirement = answer.requirement or (question.requirement if question is not None else "")
            if not requirement:
                continue
            facts.append(UserFact(requirement=requirement, content=answer.answer))
            new_fact_requirements.append(requirement)
            if question is not None:
                question.status = "answered"

        review_items = self._refresh_review_items(session, facts)
        pending_questions = [item.question for item in review_items if item.question and item.question.status == "pending"]
        proposals = build_proposals(review_items)
        state = self._resolve_state(pending_questions, proposals)
        summary = self._build_followup_summary(new_fact_requirements, pending_questions, proposals)

        session.facts = facts
        session.review_items = review_items
        session.pending_questions = pending_questions[:5]
        session.proposals = proposals
        session.state = state
        session.summary = summary
        session.updated_at = now

        if summary:
            self.repository.add_message(
                session_id,
                AgentMessage(role=MessageRole.AGENT, content=summary, created_at=now),
            )

        session.messages = self.repository.list_messages(session_id)
        self.repository.save_session(session)
        return session

    async def apply_decision(self, session_id: str, payload: ResumeAgentDecisionCreate) -> ResumeAgentSession:
        session = self.get_session(session_id)
        target = next((proposal for proposal in session.proposals if proposal.id == payload.proposal_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="Proposal not found in session.")

        target.status = payload.decision
        session.updated_at = datetime.now(UTC)

        if payload.note.strip():
            self.repository.add_message(
                session_id,
                AgentMessage(role=MessageRole.USER, content=payload.note.strip(), created_at=session.updated_at),
            )

        accepted = any(proposal.status == ProposalStatus.ACCEPTED for proposal in session.proposals)
        session.state = ResumeAgentState.COMPLETED if accepted else ResumeAgentState.AWAITING_USER_CHOICE
        session.summary = "已记录你的选择，可以继续采纳更多建议或结束本轮优化。"
        self.repository.add_message(
            session_id,
            AgentMessage(role=MessageRole.AGENT, content=session.summary, created_at=session.updated_at),
        )
        session.messages = self.repository.list_messages(session_id)
        self.repository.save_session(session)
        return session

    def _refresh_review_items(self, session: ResumeAgentSession, facts: list[UserFact]) -> list[ReviewItem]:
        return review_requirements(session.analysis.requirement_analysis, facts=facts)

    def _resolve_state(self, pending_questions, proposals) -> ResumeAgentState:
        if pending_questions:
            return ResumeAgentState.NEEDS_CLARIFICATION
        if proposals:
            return ResumeAgentState.AWAITING_USER_CHOICE
        return ResumeAgentState.PROPOSAL_READY

    def _build_summary(self, review_items: list[ReviewItem], pending_questions, proposals) -> str:
        direct_count = sum(1 for item in review_items if item.disposition.value == "direct_optimize")
        if pending_questions:
            return f"我先完成了简历审查，发现 {direct_count} 个可直接优化点，同时还有 {len(pending_questions)} 个问题需要你补充确认。"
        if proposals:
            return f"我已经整理出 {len(proposals)} 条可供你选择的改写建议，可以开始逐条确认。"
        return "我已经完成了初步审查，但当前还没有足够信息生成改写建议。"

    def _build_followup_summary(self, new_fact_requirements: list[str], pending_questions, proposals) -> str:
        if new_fact_requirements:
            requirements = "、".join(dict.fromkeys(new_fact_requirements))
            if proposals:
                return f"已根据你补充的 {requirements} 信息更新建议，目前有 {len(proposals)} 条候选改写可供确认。"
            return f"已记录你补充的 {requirements} 信息，我还需要继续确认剩余问题。"
        if pending_questions:
            return f"我已记录这轮对话，但还有 {len(pending_questions)} 个关键信息点需要确认。"
        if proposals:
            return f"当前已有 {len(proposals)} 条候选改写，欢迎继续选择或补充细节。"
        return "我已记录这轮信息，但暂时还没有新的可执行建议。"


@lru_cache(maxsize=1)
def get_resume_agent_orchestrator() -> ResumeAgentOrchestrator:
    repository = ResumeAgentRepository(settings.resume_agent_db_path)
    return ResumeAgentOrchestrator(repository)
