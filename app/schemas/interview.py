"""Persistable interview turns with explicit question and revision guards."""

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class PracticeQuestion(BaseModel):
    id: str = Field(default_factory=lambda: f"iq_{uuid4().hex[:12]}")
    requirement_id: str
    requirement: str
    text: str
    focus: str
    kind: Literal["main", "followup"] = "main"


class AnswerFeedback(BaseModel):
    score: int | None = Field(default=None, ge=0, le=100)
    evidence: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    needs_followup: bool = False
    feedback_available: bool = True


class PracticeTurn(BaseModel):
    question: PracticeQuestion
    answer: str
    skipped: bool = False
    feedback: AnswerFeedback


class PracticeReport(BaseModel):
    summary: str
    average_score: int | None = None
    assessed_count: int = 0
    answered_count: int = 0
    skipped_count: int = 0
    improvements: list[str] = Field(default_factory=list)


class InterviewRun(BaseModel):
    id: str = Field(default_factory=lambda: f"ir_{uuid4().hex[:12]}")
    revision: int = 0
    state: Literal["active", "paused", "completed"] = "active"
    source_document_revision: int
    resume_text: str
    jd_text: str
    plan: list[PracticeQuestion]
    position: int = 0
    current_question: PracticeQuestion | None = None
    turns: list[PracticeTurn] = Field(default_factory=list)
    report: PracticeReport | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InterviewStart(BaseModel):
    question_count: int = Field(default=3, ge=1, le=8)


class InterviewAnswer(BaseModel):
    run_id: str
    expected_revision: int = Field(ge=0)
    question_id: str
    answer: str = Field(default="", max_length=10000)
    skip: bool = False


class InterviewControl(BaseModel):
    run_id: str
    expected_revision: int = Field(ge=0)
    action: Literal["pause", "resume", "finish"]
