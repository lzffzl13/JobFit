"""Integration tests for Resume Agent V1 session workflow."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.jobfit import AnalysisOverview, JobFitAnalysis, ScoreBreakdown
from app.schemas.resume_agent import ProposalStatus


def _fake_jobfit_analysis() -> JobFitAnalysis:
    return JobFitAnalysis(
        match_score=66,
        summary="分析完成",
        matched_strengths=[],
        gaps=[],
        resume_rewrites=[],
        interview_questions=[],
        evidence=[],
        score_breakdown=ScoreBreakdown(total_score=66),
        requirement_analysis=[
            {
                "requirement": "Python",
                "category": "skill",
                "level": "required",
                "matched": True,
                "score": 100,
                "status": "strong_match",
                "evidence": "Python 后端服务开发",
                "method": "exact",
                "explanation": "已命中",
                "suggestion": "",
            },
            {
                "requirement": "Docker",
                "category": "skill",
                "level": "required",
                "matched": False,
                "score": 0,
                "status": "gap",
                "evidence": "",
                "method": "",
                "explanation": "缺少直接证据",
                "suggestion": "补充 Docker 相关经历",
            },
        ],
        analysis_overview=AnalysisOverview(strong_match_count=1, gap_count=1),
        core_requirements=["Python", "Docker"],
        bonus_requirements=[],
        risk_items=["Docker: 0%"],
        risk_details=[],
        model_used="fake",
        fallback_used=False,
    )


def test_create_resume_agent_session(monkeypatch, tmp_path):
    db_path = tmp_path / "resume-agent.db"
    monkeypatch.setattr("app.services.resume_agent.orchestrator.settings.resume_agent_db_path", str(db_path))
    monkeypatch.setattr("app.services.resume_agent.orchestrator.analyze_job_fit", _fake_analyze_job_fit)
    from app.services.resume_agent.orchestrator import get_resume_agent_orchestrator

    get_resume_agent_orchestrator.cache_clear()
    client = TestClient(app)

    response = client.post(
        "/resume-agent/sessions",
        json={
            "resume_text": "我使用 Python 开发过后端服务，并参与过接口开发和缓存设计。",
            "jd_text": "需要 Python 和 Docker 能力，能支持服务部署与工程化工作。",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "needs_clarification"
    assert len(payload["pending_questions"]) == 1
    assert len(payload["proposals"]) == 1
    review_by_requirement = {item["requirement"]: item for item in payload["review_items"]}
    assert review_by_requirement["Python"]["write_policy"] == "safe_rewrite"
    assert review_by_requirement["Python"]["confidence"] >= 0.8
    assert review_by_requirement["Docker"]["write_policy"] == "ask_for_facts"
    assert review_by_requirement["Docker"]["missing_info"]
    assert payload["pending_questions"][0]["expected_evidence"]
    assert "真实" in payload["pending_questions"][0]["question"]
    assert payload["proposals"][0]["confidence"] >= 0.8
    assert payload["proposals"][0]["tone"] == "strong"
    assert payload["proposals"][0]["safety_notes"]


def test_resume_agent_message_updates_session(monkeypatch, tmp_path):
    db_path = tmp_path / "resume-agent.db"
    monkeypatch.setattr("app.services.resume_agent.orchestrator.settings.resume_agent_db_path", str(db_path))
    monkeypatch.setattr("app.services.resume_agent.orchestrator.analyze_job_fit", _fake_analyze_job_fit)
    from app.services.resume_agent.orchestrator import get_resume_agent_orchestrator

    get_resume_agent_orchestrator.cache_clear()
    client = TestClient(app)

    create_response = client.post(
        "/resume-agent/sessions",
        json={
            "resume_text": "我使用 Python 开发过后端服务，并参与过接口开发和缓存设计。",
            "jd_text": "需要 Python 和 Docker 能力，能支持服务部署与工程化工作。",
        },
    )
    session = create_response.json()
    question = session["pending_questions"][0]

    response = client.post(
        f"/resume-agent/sessions/{session['id']}/messages",
        json={
            "content": "我补充一下部署相关信息。",
            "answers": [
                {
                    "question_id": question["id"],
                    "requirement": "Docker",
                    "answer": "我在个人项目中用 Docker 打包过 FastAPI 服务，并自己写过 docker-compose。",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "awaiting_user_choice"
    assert len(payload["pending_questions"]) == 0
    docker_proposal = next(item for item in payload["proposals"] if item["requirement"] == "Docker")
    assert docker_proposal["source_section"] == "skills_or_projects"
    assert docker_proposal["tone"] == "conservative"
    assert "Docker" in docker_proposal["after"]
    assert docker_proposal["safety_notes"]


def test_resume_agent_decision_marks_completion(monkeypatch, tmp_path):
    db_path = tmp_path / "resume-agent.db"
    monkeypatch.setattr("app.services.resume_agent.orchestrator.settings.resume_agent_db_path", str(db_path))
    monkeypatch.setattr("app.services.resume_agent.orchestrator.analyze_job_fit", _fake_analyze_job_fit)
    from app.services.resume_agent.orchestrator import get_resume_agent_orchestrator

    get_resume_agent_orchestrator.cache_clear()
    client = TestClient(app)

    create_response = client.post(
        "/resume-agent/sessions",
        json={
            "resume_text": "我使用 Python 开发过后端服务，并参与过接口开发和缓存设计。",
            "jd_text": "需要 Python 和 Docker 能力，能支持服务部署与工程化工作。",
        },
    )
    session = create_response.json()
    proposal = session["proposals"][0]

    response = client.post(
        f"/resume-agent/sessions/{session['id']}/decisions",
        json={
            "proposal_id": proposal["id"],
            "decision": ProposalStatus.ACCEPTED,
            "note": "这条建议我先采纳。",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "completed"
    selected = next(item for item in payload["proposals"] if item["id"] == proposal["id"])
    assert selected["status"] == "accepted"


async def _fake_analyze_job_fit(resume_text: str, jd_text: str) -> JobFitAnalysis:
    return _fake_jobfit_analysis()
