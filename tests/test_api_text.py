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
    assert "core_points" in payload["score_breakdown"]
    assert "bonus_points" in payload["score_breakdown"]
    assert "extra_points" in payload["score_breakdown"]
    assert "core_requirements" in payload
    assert "extra_strengths" in payload
    assert "risk_items" in payload


def test_deepseek_output_cannot_override_deterministic_score(monkeypatch):
    class FakeDeepSeekClient:
        async def analyze(self, resume_context: str, jd_context: str, jd_semantics: str):
            return {
                "match_score": 1,
                "summary": "模型故意给出很低分。",
                "jd_requirements": ["FastAPI/Flask/Django"],
                "matched_strengths": ["模型认为只命中一点"],
                "gaps": ["Flask"],
                "resume_rewrites": ["补充项目证据"],
                "interview_questions": ["请介绍 FastAPI 项目"],
            }

    monkeypatch.setattr("app.services.jobfit.DeepSeekClient", FakeDeepSeekClient)

    client = TestClient(app)
    response = client.post(
        "/jobfit/analyze",
        data={
            "resume_text": (
                "我使用 Python、FastAPI、MySQL 和 SQLAlchemy 开发博客后端项目，"
                "完成用户、文章、评论 RESTful API 和 CRUD 接口，使用 Docker 部署，"
                "pytest 编写接口测试，Git 协作维护文档，获得竞赛二等奖。"
            ),
            "jd_text": (
                "掌握 Python/JAVA 基础语法，了解至少一种 Python Web 框架 Django/Flask/FastAPI，"
                "熟悉 MySQL 和 SQL，能够参与后端接口开发。Spring Boot、MyBatis 为加分项。"
            ),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback_used"] is False
    assert payload["match_score"] >= 60
    assert payload["match_score"] != 1
    assert payload["score_breakdown"]["core_points"] > 0
    assert all(gap["requirement"] != "Flask" for gap in payload["gaps"])


def test_analyze_endpoint_requires_resume_text_or_file():
    client = TestClient(app)
    response = client.post(
        "/jobfit/analyze",
        data={"jd_text": "要求熟悉 Python FastAPI Redis，并了解 RAG 和 Agent。"},
    )

    assert response.status_code == 400
    assert "Provide a resume file or paste resume text" in response.json()["detail"]
