"""Explicit, source-bound edits and immutable document snapshots."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ResumeField(BaseModel):
    id: str
    text: str
    evidence: dict[str, str] = Field(default_factory=dict)


class ResumeVersion(BaseModel):
    revision: int
    label: str
    fields: list[ResumeField]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResumeDocument(BaseModel):
    revision: int = 0
    fields: list[ResumeField] = Field(default_factory=list)
    versions: list[ResumeVersion] = Field(default_factory=list)
    outdated_requirements: list[str] = Field(default_factory=list)


class FieldChange(BaseModel):
    proposal_ids: list[str]
    field_id: str
    label: str = "简历段落"
    path: str
    operation: Literal["replace", "add"]
    before: str
    after: str
    evidence_basis: list[str]
    source_ids: list[str]
    checks: list[str]


class ResumePreview(BaseModel):
    token: str
    base_revision: int
    changes: list[FieldChange]
    text: str
    fields: list[ResumeField]


class DocumentApply(BaseModel):
    token: str = Field(min_length=64, max_length=64)
    expected_revision: int = Field(ge=0)


class DocumentEdit(BaseModel):
    text: str = Field(min_length=30, max_length=100000)
    expected_revision: int = Field(ge=0)


class DocumentRestore(BaseModel):
    revision: int = Field(ge=0)
    expected_revision: int = Field(ge=0)
