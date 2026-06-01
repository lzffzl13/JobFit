from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    source: str
    chunk_id: int
    text: str
    score: float = Field(ge=0)


class Requirement(BaseModel):
    name: str
    category: str = "general"
    evidence: str | None = None
    type: str = "single"
    options: list[str] = Field(default_factory=list)
    required_count: int = 1
    priority: str = "core"
    weight: float = 1.0


class MatchItem(BaseModel):
    requirement: str
    resume_evidence: str
    score: int = Field(ge=0, le=100)


class GapItem(BaseModel):
    requirement: str
    suggestion: str


class ResumeRewrite(BaseModel):
    before: str
    after: str
    reason: str


class InterviewQuestion(BaseModel):
    question: str
    focus: str
    difficulty: str = "medium"


class ScoreBreakdown(BaseModel):
    core_matched: int = 0
    core_total: int = 0
    bonus_matched: int = 0
    bonus_total: int = 0
    core_ratio: float = 0
    bonus_ratio: float = 0
    core_score: float = 0
    bonus_jd: float = 0
    bonus_extra: float = 0
    raw_score: float = 0
    final_cap: int = 100
    cap_reasons: list[str] = Field(default_factory=list)
    core_detail: dict[str, str] = Field(default_factory=dict)
    evidence_notes: list[str] = Field(default_factory=list)
    core_points: int = 0
    core_points_total: int = 70
    bonus_points: int = 0
    bonus_points_total: int = 15
    extra_points: int = 0
    extra_points_total: int = 15
    extra_matched: int = 0
    extra_total: int = 5


class JobFitAnalysis(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    match_score: int = Field(ge=0, le=100)
    summary: str
    jd_requirements: list[Requirement] = Field(default_factory=list)
    matched_strengths: list[MatchItem] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    resume_rewrites: list[ResumeRewrite] = Field(default_factory=list)
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    core_requirements: list[str] = Field(default_factory=list)
    bonus_requirements: list[str] = Field(default_factory=list)
    extra_strengths: list[str] = Field(default_factory=list)
    risk_items: list[str] = Field(default_factory=list)
    model_used: str
    fallback_used: bool = False
