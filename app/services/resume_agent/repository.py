"""SQLite-backed repository for Resume Agent V1 sessions."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from app.schemas.resume_agent import AgentMessage, ResumeAgentSession


class ResumeAgentRepository:
    """Persist resume agent sessions and messages in SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_database()

    def save_session(self, session: ResumeAgentSession) -> None:
        expected = session.storage_revision
        payload = session.model_dump(mode="json")
        payload["storage_revision"] = expected + 1
        with self._connect() as conn:
            exists = conn.execute("SELECT id FROM resume_agent_sessions WHERE id = ?", (session.id,)).fetchone()
            if exists:
                updated = conn.execute(
                    "UPDATE resume_agent_sessions SET state=?, payload_json=?, updated_at=?, storage_revision=? "
                    "WHERE id=? AND storage_revision=?",
                    (session.state.value, json.dumps(payload, ensure_ascii=False),
                     session.updated_at.isoformat(), expected + 1, session.id, expected),
                )
                if updated.rowcount != 1:
                    raise HTTPException(status_code=409, detail="会话已在其他页面更新，请刷新后重试。")
            else:
                conn.execute(
                    """
                    INSERT INTO resume_agent_sessions (id, state, payload_json, created_at, updated_at, storage_revision)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.id,
                        session.state.value,
                        json.dumps(payload, ensure_ascii=False),
                        session.created_at.isoformat(),
                        session.updated_at.isoformat(),
                        expected + 1,
                    ),
                )
            for message in session.messages:
                conn.execute(
                    "INSERT OR IGNORE INTO resume_agent_messages (id, session_id, role, content, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (message.id, session.id, message.role.value, message.content, message.created_at.isoformat()),
                )
            conn.commit()
        session.storage_revision = expected + 1

    def get_session(self, session_id: str) -> ResumeAgentSession | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, storage_revision FROM resume_agent_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload["storage_revision"] = row["storage_revision"]
        return ResumeAgentSession.model_validate(payload)

    def add_message(self, session_id: str, message: AgentMessage) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO resume_agent_messages (id, session_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    session_id,
                    message.role.value,
                    message.content,
                    message.created_at.isoformat(),
                ),
            )
            conn.commit()

    def list_messages(self, session_id: str) -> list[AgentMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM resume_agent_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            AgentMessage(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_database(self) -> None:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_agent_sessions (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(resume_agent_sessions)")}
            if "storage_revision" not in columns:
                conn.execute("ALTER TABLE resume_agent_sessions ADD COLUMN storage_revision INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_agent_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
