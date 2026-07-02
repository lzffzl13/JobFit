"""Integration tests for the analyze endpoint with pasted text."""

from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Fake LLM client — returns appropriate responses for each extraction step
# ---------------------------------------------------------------------------


_FAKE_RESUME_RESPONSE = {
    "skills": {"hard": ["Python", "FastAPI", "Redis"], "soft": ["沟通能力"]},
    "experience_years": {"total": 2, "backend": 2},
    "projects": [{"name": "求职分析系统", "tech": ["FastAPI", "Redis"], "desc": "RAG检索系统"}],
    "education": {"degree": "本科", "major": "计算机科学", "school": "XX大学"},
    "certifications": [],
}

_FAKE_JD_RESPONSE = {
    "requirements": [
        {"name": "Python", "category": "skill", "level": "required", "description": "熟悉Python"},
        {"name": "FastAPI", "category": "skill", "level": "required", "description": "有Web框架经验"},
        {"name": "Redis", "category": "skill", "level": "preferred", "description": "了解缓存"},
        {"name": "Docker", "category": "skill", "level": "preferred", "description": "容器部署"},
    ]
}

_FAKE_SUGGESTION_RESPONSE = {
    "summary": "匹配度较高，后端技术栈基本匹配。",
    "resume_rewrites": [
        {"before": "做过Python项目", "after": "使用Python FastAPI开发高性能后端服务", "reason": "更具体"},
    ],
    "interview_questions": [
        {"question": "请介绍Redis缓存策略", "focus": "缓存设计", "difficulty": "medium"},
    ],
}


class FakeLLMClient:
    """Fake LLM client that returns different responses based on prompt content."""

    async def analyze(self, system_prompt: str, user_prompt: str) -> dict:
        if "简历" in user_prompt and "提取" in user_prompt:
            return _FAKE_RESUME_RESPONSE
        if "职位描述" in user_prompt or "JD" in user_prompt:
            return _FAKE_JD_RESPONSE
        if "匹配分析结果" in user_prompt or "建议" in user_prompt:
            return _FAKE_SUGGESTION_RESPONSE
        return _FAKE_RESUME_RESPONSE


def _mock_batch_embedding_similarity(query: str, candidates: list[str]) -> list[float]:
    """Mock embedding: always returns 0.0 (no match). Tests rely on synonym matching only."""
    return [0.0] * len(candidates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_analyze_endpoint_with_pasted_resume_text(monkeypatch):
    monkeypatch.setattr("app.services.jobfit.get_llm_client", lambda: FakeLLMClient())
    monkeypatch.setattr("app.services.matcher._batch_embedding_similarity", _mock_batch_embedding_similarity)

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
    assert "score_breakdown" in payload
    assert "core_requirements" in payload
    assert "risk_items" in payload
    assert "requirement_analysis" in payload
    assert "analysis_overview" in payload
    assert "risk_details" in payload
    assert payload["fallback_used"] is False


def test_match_score_reflects_skills(monkeypatch):
    """Match score should be driven by program matching, not LLM."""
    monkeypatch.setattr("app.services.jobfit.get_llm_client", lambda: FakeLLMClient())
    monkeypatch.setattr("app.services.matcher._batch_embedding_similarity", _mock_batch_embedding_similarity)

    client = TestClient(app)
    response = client.post(
        "/jobfit/analyze",
        data={
            "resume_text": "我使用 Python 和 FastAPI 开发过后端服务，有 Redis 缓存使用经验。",
            "jd_text": "要求熟悉 Python、FastAPI、Redis、Docker 容器化部署经验。",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    # Python + FastAPI + Redis matched via synonym, Docker not in resume
    assert payload["match_score"] > 0
    assert len(payload["matched_strengths"]) > 0
    analyses = {item["requirement"]: item for item in payload["requirement_analysis"]}
    assert analyses["Python"]["status"] == "strong_match"
    assert analyses["Docker"]["status"] == "gap"
    assert analyses["Docker"]["suggestion"] != ""


def test_analyze_endpoint_requires_resume_text_or_file():
    client = TestClient(app)
    response = client.post(
        "/jobfit/analyze",
        data={"jd_text": "要求熟悉 Python FastAPI Redis，并了解 RAG 和 Agent 技术。"},
    )

    assert response.status_code == 400
    assert "Provide a resume file or paste resume text" in response.json()["detail"]


def test_suggestions_populated(monkeypatch):
    """LLM suggestions (rewrites, questions) should be in the response."""
    monkeypatch.setattr("app.services.jobfit.get_llm_client", lambda: FakeLLMClient())
    monkeypatch.setattr("app.services.matcher._batch_embedding_similarity", _mock_batch_embedding_similarity)

    client = TestClient(app)
    response = client.post(
        "/jobfit/analyze",
        data={
            "resume_text": "我使用 Python 和 FastAPI 开发过后端服务项目，有 Redis 缓存经验。",
            "jd_text": "要求熟悉 Python、FastAPI、Redis、Docker 容器化部署经验。",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["resume_rewrites"]) > 0
    assert len(payload["interview_questions"]) > 0
    assert payload["summary"] != ""
    assert isinstance(payload["analysis_overview"]["strong_match_count"], int)
    assert isinstance(payload["risk_details"], list)
