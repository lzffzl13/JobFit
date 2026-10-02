"""Build a reviewed preview; apply only the exact source and revision it binds."""

import hashlib
import json
import re

from fastapi import HTTPException

from app.schemas.resume_document import (
    FieldChange,
    ResumeDocument,
    ResumeField,
    ResumePreview,
    ResumeVersion,
)
from app.services.source_rewrite import rewrite_source_passage


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def split_fields(text: str) -> list[ResumeField]:
    # Preserve every character, including blank lines. Fields are text paragraphs, not LLM extractions.
    return [ResumeField(id=f"fld_{digest([index, part])[:16]}", text=part)
            for index, part in enumerate(re.split(r"(\r?\n[ \t]*\r?\n)", text)) if part]


def document_text(fields: list[ResumeField]) -> str:
    return "".join(field.text for field in fields)


def initialize_document(session) -> bool:
    if session.document is not None:
        refresh_document_validity(session)
        return False
    fields = split_fields(session.resume_text)
    session.document = ResumeDocument(fields=fields, versions=[
        ResumeVersion(revision=0, label="原始简历", fields=[f.model_copy(deep=True) for f in fields])
    ])
    return True


def source_for(session, requirement_id):
    fact = next((f for f in reversed(session.facts) if f.requirement_id == requirement_id), None)
    if fact:
        return fact.content if fact.confirmed and fact.answer_type == "details" else "", fact.id
    return session.resume_text, f"resume:{session.id}"


def evidence_stamp(session, requirement_id) -> str:
    assessment = session.assessments.get(requirement_id)
    source, source_id = source_for(session, requirement_id)
    return digest([assessment.model_dump() if assessment else None, source, source_id])


def refresh_document_validity(session) -> None:
    if session.document is None:
        return
    session.document.outdated_requirements = sorted({
        key for field in session.document.fields for key, stamp in field.evidence.items()
        if stamp != evidence_stamp(session, key)
    })


def make_preview(session) -> ResumePreview:
    initialize_document(session)
    if session.pending_questions:
        raise HTTPException(status_code=409, detail="请先完成事实确认，再预览简历修改。")
    selected = [p for p in session.proposals if p.status == "accepted"]
    if not selected:
        raise HTTPException(status_code=400, detail="请先选择至少一条候选改写。")
    fields = [field.model_copy(deep=True) for field in session.document.fields]
    changes: dict[str, FieldChange] = {}
    for proposal in selected:
        source, source_id = source_for(session, proposal.requirement_id)
        assessment = session.assessments.get(proposal.requirement_id)
        if (not assessment or assessment.outcome != "supported" or not proposal.before
                or proposal.before != proposal.evidence_basis
                or proposal.evidence_basis != assessment.evidence
                or proposal.evidence_basis not in source
                or proposal.after != rewrite_source_passage(proposal.evidence_basis)):
            raise HTTPException(status_code=409, detail=f"“{proposal.requirement}”的事实或改写未通过校验，请重新确认。")
        stamp = evidence_stamp(session, proposal.requirement_id)
        matches = [f for f in fields if proposal.before in f.text]
        if len(matches) > 1 or (matches and matches[0].text.count(proposal.before) != 1):
            raise HTTPException(status_code=409, detail="原文出现多次，无法唯一定位修改字段，请先手动整理简历。")
        if matches:
            target = matches[0]
            if target.evidence.get(proposal.requirement_id) == stamp and proposal.before == proposal.after:
                continue
            before = target.text
            target.text = before.replace(proposal.before, proposal.after, 1)
            operation = "replace"
        else:
            # A previously applied rewrite can be selected again without duplicating it.
            applied = [f for f in fields if f.evidence.get(proposal.requirement_id) == stamp
                       and proposal.after in f.text]
            if len(applied) == 1:
                continue
            # Merge identical source passages already changed by another selected requirement.
            shared = [f for f in fields if f.id in changes
                      and proposal.before in changes[f.id].evidence_basis and proposal.after in f.text]
            if len(shared) == 1:
                target = shared[0]
                before = changes[target.id].before
                operation = changes[target.id].operation
            elif source_id.startswith("fact_"):
                previous = [f for f in fields if proposal.requirement_id in f.evidence]
                if len(previous) == 1 and previous[0].id.startswith("add_"):
                    target = previous[0]
                    before = target.text
                    target.text = "\n\n" + proposal.after
                    operation = "replace"
                else:
                    target = ResumeField(id=f"add_{digest([proposal.requirement_id, source_id])[:16]}",
                                         text="\n\n" + proposal.after)
                    fields.append(target)
                    before, operation = "", "add"
            else:
                raise HTTPException(status_code=409, detail="目标原文已经改变，旧提案不能覆盖当前简历。")
        target.evidence[proposal.requirement_id] = stamp
        if target.id not in changes:
            changes[target.id] = FieldChange(
                proposal_ids=[], field_id=target.id, path=f"/fields/{target.id}/text",
                label=("已确认的补充经历" if target.id.startswith("add_") else
                       f"原文第 {sum(bool(f.text.strip()) for f in fields[:fields.index(target) + 1])} 段"),
                operation=operation, before=before, after=target.text,
                evidence_basis=[], source_ids=[],
                checks=["原文与事实来源一致", "未新增技能、指标或职责", "保留原文限定范围"],
            )
        change = changes[target.id]
        change.after = target.text
        change.proposal_ids.append(proposal.id)
        change.evidence_basis.append(proposal.evidence_basis)
        change.source_ids.append(source_id)
    if not changes:
        raise HTTPException(status_code=400, detail="所选表达已经应用，无需重复写入。")
    bound = {
        "session": session.id, "jd": session.jd_text, "resume": session.resume_text,
        "revision": session.document.revision,
        "current": [f.model_dump() for f in session.document.fields],
        "proposals": [p.model_dump(mode="json") for p in selected],
        "facts": [f.model_dump(mode="json") for f in session.facts],
        "changes": [c.model_dump() for c in changes.values()],
    }
    return ResumePreview(token=digest(bound), base_revision=session.document.revision,
                         changes=list(changes.values()), fields=fields, text=document_text(fields))


def check_revision(session, expected: int) -> None:
    initialize_document(session)
    if session.document.revision != expected:
        raise HTTPException(status_code=409, detail="简历版本已更新，请刷新并重新预览。")


def save_version(session, fields: list[ResumeField], label: str) -> None:
    document = session.document
    document.revision += 1
    document.fields = [f.model_copy(deep=True) for f in fields]
    document.versions.append(ResumeVersion(
        revision=document.revision, label=label, fields=[f.model_copy(deep=True) for f in fields],
    ))
    session.preview = None
    refresh_document_validity(session)
