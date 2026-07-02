"""Schemas for Resume Agent V1 session workflow."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from app.schemas.jobfit import AnalysisOverview, JobFitAnalysis


class ResumeAgentState(StrEnum):
    INTAKE = "intake"
    REVIEWING = "reviewing"
    NEEDS_CLARIFICATION = "needs_clarification"
    PROPOSAL_READY = "proposal_ready"
    AWAITING_USER_CHOICE = "awaiting_user_choice"
    COMPLETED = "completed"


class MessageRole(StrEnum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class ProposalStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVISED = "revised"


class ReviewDisposition(StrEnum):
    DIRECT_OPTIMIZE = "direct_optimize"
    CLARIFY = "clarify"
    DO_NOT_WRITE = "do_not_write"


class ClarifyingQuestion(BaseModel):
    id: str = Field(default_factory=lambda: f"q_{uuid4().hex[:10]}")
    requirement: str
    question: str
    rationale: str = ""
    status: str = "pending"


class UserFact(BaseModel):
    id: str = Field(default_factory=lambda: f"fact_{uuid4().hex[:10]}")
    requirement: str
    content: str
    source: str = "user"
    confirmed: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RewriteProposal(BaseModel):
    id: str = Field(default_factory=lambda: f"proposal_{uuid4().hex[:10]}")
    requirement: str
    source_section: str = "experience"
    before: str = ""
    after: str = ""
    reason: str = ""
    evidence_basis: str = ""
    needs_user_confirmation: bool = True
    status: ProposalStatus = ProposalStatus.PROPOSED


class ReviewItem(BaseModel):
    requirement: str
    disposition: ReviewDisposition
    reason: str
    evidence: str = ""
    question: ClarifyingQuestion | None = None


class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex[:10]}")
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class QuestionAnswer(BaseModel):
    question_id: str
    requirement: str | None = None
    answer: str


class ResumeAgentSessionCreate(BaseModel):
    resume_text: str = Field(min_length=30)
    jd_text: str = Field(min_length=20)


class ResumeAgentMessageCreate(BaseModel):
    content: str = ""
    answers: list[QuestionAnswer] = Field(default_factory=list)


class ResumeAgentDecisionCreate(BaseModel):
    proposal_id: str
    decision: ProposalStatus
    note: str = ""


class ResumeAgentSession(BaseModel):
    id: str
    state: ResumeAgentState
    summary: str = ""
    resume_text: str
    jd_text: str
    analysis: JobFitAnalysis
    analysis_overview: AnalysisOverview = Field(default_factory=AnalysisOverview)
    review_items: list[ReviewItem] = Field(default_factory=list)
    pending_questions: list[ClarifyingQuestion] = Field(default_factory=list)
    facts: list[UserFact] = Field(default_factory=list)
    proposals: list[RewriteProposal] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
