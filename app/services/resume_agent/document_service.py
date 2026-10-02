"""Document mutations persisted together with their version history."""

from datetime import UTC, datetime

from fastapi import HTTPException

from app.services.resume_agent.document import (
    check_revision,
    document_text,
    make_preview,
    save_version,
    split_fields,
)


class ResumeDocumentService:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def persist(self, session):
        session.updated_at = datetime.now(UTC)
        self.orchestrator.repository.save_session(session)
        return session

    def preview(self, session_id):
        session = self.orchestrator.get_session(session_id)
        session.preview = make_preview(session)
        return self.persist(session)

    def apply(self, session_id, payload):
        session = self.orchestrator.get_session(session_id)
        check_revision(session, payload.expected_revision)
        if session.preview is None or session.preview.token != payload.token:
            raise HTTPException(status_code=409, detail="预览已失效，请重新生成并核对修改。")
        current = make_preview(session)
        if current.token != payload.token:
            raise HTTPException(status_code=409, detail="事实、岗位或提案已经变化，请重新预览。")
        save_version(session, current.fields, "应用已确认的改写")
        session.summary = f"已生成简历第 {session.document.revision} 版，可以查看、导出或恢复历史版本。"
        return self.persist(session)

    def edit(self, session_id, payload):
        session = self.orchestrator.get_session(session_id)
        check_revision(session, payload.expected_revision)
        if payload.text != document_text(session.document.fields):
            save_version(session, split_fields(payload.text), "用户手动编辑")
        return self.persist(session)

    def restore(self, session_id, payload):
        session = self.orchestrator.get_session(session_id)
        check_revision(session, payload.expected_revision)
        version = next((v for v in session.document.versions if v.revision == payload.revision), None)
        if version is None:
            raise HTTPException(status_code=404, detail="该简历版本不存在。")
        save_version(session, version.fields, f"恢复第 {version.revision} 版")
        return self.persist(session)

    def export(self, session_id):
        session = self.orchestrator.get_session(session_id)
        if session.document.outdated_requirements:
            raise HTTPException(status_code=409, detail="已有表达的事实依据已更正，请更新或手动核对简历后再导出。")
        return document_text(session.document.fields)
