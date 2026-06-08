"""JobFit schemas — structured data models for the three-step pipeline.

Step 1 (LLM): ResumeProfile / JDProfile — extracted from raw text
Step 2 (Program): MatchResult — deterministic matching calculation
Step 3 (LLM): Suggestions embedded in final JobFitAnalysis output
"""

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Step 1: LLM extraction schemas
# ---------------------------------------------------------------------------


class SkillsBlock(BaseModel):
    """Flat skill lists — enables set operations for matching."""

    hard: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)


class ProjectBlock(BaseModel):
    """Project with context — preserves semantic structure for embedding."""

    name: str = ""
    tech: list[str] = Field(default_factory=list)
    desc: str = ""
    highlights: list[str] = Field(default_factory=list)


class EducationBlock(BaseModel):
    degree: str = ""
    major: str = ""
    school: str = ""


class ResumeProfile(BaseModel):
    """Structured data extracted from a resume by LLM."""

    skills: SkillsBlock = Field(default_factory=SkillsBlock)
    experience_years: dict[str, float] = Field(default_factory=dict)
    projects: list[ProjectBlock] = Field(default_factory=list)
    education: EducationBlock = Field(default_factory=EducationBlock)
    certifications: list[str] = Field(default_factory=list)


class JDRequirement(BaseModel):
    """Single requirement extracted from a JD."""

    name: str
    category: str = "skill"             # skill / experience / education / soft
    level: str = "required"             # required / preferred / nice-to-have
    description: str = ""
    alternatives: list[str] = Field(default_factory=list)


class JDProfile(BaseModel):
    """Structured data extracted from a JD by LLM."""

    requirements: list[JDRequirement] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 2: Program matching schemas
# ---------------------------------------------------------------------------


class MatchDetail(BaseModel):
    """Per-requirement match result."""

    requirement: str
    category: str = "skill"
    level: str = "required"
    matched: bool = False
    match_score: float = Field(ge=0, le=1, default=0)
    evidence: str = ""                  # what in the resume matched
    method: str = ""                    # synonym / embedding / exact / rule


class GapDetail(BaseModel):
    """Unmatched or weak requirement."""

    requirement: str
    category: str = "skill"
    level: str = "required"
    current_score: float = 0
    suggestion: str = ""                # program-generated hint


class ScoreBreakdown(BaseModel):
    """Weighted score breakdown by category."""

    skill_score: int = 0
    experience_score: int = 0
    project_score: int = 0
    education_score: int = 0
    total_score: int = 0
    skill_total: int = 0
    experience_total: int = 0
    project_total: int = 0
    education_total: int = 0


class MatchResult(BaseModel):
    """Deterministic matching result — no LLM involved."""

    total_score: int = Field(ge=0, le=100, default=0)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    matched: list[MatchDetail] = Field(default_factory=list)
    gaps: list[GapDetail] = Field(default_factory=list)
    core_requirements: list[str] = Field(default_factory=list)
    bonus_requirements: list[str] = Field(default_factory=list)
    risk_items: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 3: Final API output (preserves frontend compatibility)
# ---------------------------------------------------------------------------


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


class Evidence(BaseModel):
    """Kept for compatibility — now represents match evidence instead of RAG chunks."""

    source: str
    chunk_id: int = 0
    text: str
    score: float = Field(ge=0, default=0)


class JobFitAnalysis(BaseModel):
    """Final API response — combines match result + LLM suggestions."""

    model_config = ConfigDict(protected_namespaces=())

    match_score: int = Field(ge=0, le=100)
    summary: str
    matched_strengths: list[MatchItem] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    resume_rewrites: list[ResumeRewrite] = Field(default_factory=list)
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    core_requirements: list[str] = Field(default_factory=list)
    bonus_requirements: list[str] = Field(default_factory=list)
    risk_items: list[str] = Field(default_factory=list)
    model_used: str = ""
    fallback_used: bool = False
