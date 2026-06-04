from fastapi.testclient import TestClient

from app.main import app


def test_analyze_endpoint_with_pasted_resume_text():
    client = TestClient(app)
    response = client.post(
        "/jobfit/analyze",
        data={
            "resume_text": (
                "我做过 Python FastAPI 后端项目，使用 Redis 缓存，"
                "也做过一个支持 RAG 检索和 DeepSeek 调用的求职分析系统。"
            ),
            "jd_text": (
                "要求熟悉 Python、FastAPI、Redis，了解 RAG、LLM 和 Agent，"
                "具备 Docker 部署和接口开发经验。"
            ),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["match_score"] >= 0
    assert payload["evidence"]
    assert "score_breakdown" in payload
    assert "match_score" in payload["score_breakdown"]
    assert "core_requirements" in payload
    assert "risk_items" in payload


def test_llm_ratios_drive_score(monkeypatch):
    """LLM's per-requirement evidence_ratio drives the final score via validator."""

    class FakeLLMClient:
        async def analyze(self, system_prompt: str, user_prompt: str):
            return {
                "match_score": 76,
                "bonus_score": 10,
                "extra_score": 5,
                "requirements": [
                    {"name": "Python 基础", "priority": "core", "evidence_ratio": 0.9, "confidence": 0.85, "reasoning": "strong Python", "evidence_quote": "Python"},
                    {"name": "Python Web框架", "priority": "core", "evidence_ratio": 0.8, "confidence": 0.8, "reasoning": "FastAPI project", "evidence_quote": "FastAPI"},
                    {"name": "MySQL/SQL", "priority": "core", "evidence_ratio": 0.7, "confidence": 0.75, "reasoning": "MySQL usage", "evidence_quote": "MySQL"},
                    {"name": "接口开发能力", "priority": "core", "evidence_ratio": 0.85, "confidence": 0.9, "reasoning": "RESTful API", "evidence_quote": "RESTful"},
                ],
                "summary": "较强的后端匹配。",
                "gaps": [],
                "resume_rewrites": [],
                "interview_questions": [],
            }

    monkeypatch.setattr("app.services.jobfit.get_llm_client", lambda: FakeLLMClient())

    client = TestClient(app)
    response = client.post(
        "/jobfit/analyze",
        data={
            "resume_text": (
                "我使用 Python、FastAPI、MySQL 和 SQLAlchemy 开发博客后端项目，"
                "完成用户、文章、评论 RESTful API 和 CRUD 接口。"
            ),
            "jd_text": (
                "掌握 Python 基础语法，了解 Python Web 框架 Django/Flask/FastAPI，"
                "熟悉 MySQL 和 SQL，能够参与后端接口开发。"
            ),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback_used"] is False
    # LLM outputs match_score=76 directly; validator only clamps
    assert payload["match_score"] >= 60
    assert payload["score_breakdown"]["core_matched"] > 0


def test_analyze_endpoint_requires_resume_text_or_file():
    client = TestClient(app)
    response = client.post(
        "/jobfit/analyze",
        data={"jd_text": "要求熟悉 Python FastAPI Redis，并了解 RAG 和 Agent。"},
    )

    assert response.status_code == 400
    assert "Provide a resume file or paste resume text" in response.json()["detail"]
