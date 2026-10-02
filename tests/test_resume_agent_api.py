"""Regression tests for fact confirmation, evidence updates and proposal consent."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.jobfit import JobFitAnalysis
from app.schemas.resume_agent import ProposalStatus, ResumeAgentState
from app.services.resume_agent.orchestrator import get_resume_agent_orchestrator

RESUME = "我在博客项目中使用 Python 实现登录接口，并用 pytest 编写接口测试。"
DOCKER = "我在个人项目中使用 Docker 打包 FastAPI 服务，并编写 docker-compose 配置。"
DOCKER_NEW = "我在博客项目中使用 Docker 部署测试环境，并编写健康检查脚本。"
JD = "招聘后端开发，要求使用 Python 开发接口，并有 Docker 容器打包和配置经验。"


def fake_analysis():
    return JobFitAnalysis(match_score=50, summary="初步分析", requirement_analysis=[
        {"requirement": "Python", "category": "skill", "level": "required",
         "matched": True, "score": 100, "status": "strong_match", "evidence": "Python"},
        {"requirement": "Docker", "category": "skill", "level": "required",
         "matched": False, "score": 0, "status": "gap", "evidence": ""},
    ])


class FakeEvidenceClient:
    """Only model responses are stubbed; source checks and workflow run normally."""

    async def analyze(self, system_prompt, user_prompt):
        items = []
        for request in json.loads(user_prompt)["items"]:
            source = request["source"]
            evidence = ""
            if request["requirement"] == "Python" and RESUME in source:
                evidence = RESUME
                context, action = "博客项目", "使用 Python 实现登录接口"
            elif request["requirement"] == "Docker" and DOCKER_NEW in source:
                evidence = DOCKER_NEW
                context, action = "博客项目", "使用 Docker 部署测试环境"
            elif request["requirement"] == "Docker" and DOCKER in source:
                evidence = DOCKER
                context, action = "个人项目", "使用 Docker 打包 FastAPI 服务"
            if evidence:
                items.append({"key": request["key"], "outcome": "supported", "coverage": "full",
                              "evidence": evidence, "context": context, "action": action})
            else:
                items.append({"key": request["key"], "outcome": "insufficient",
                              "followup": "请补充真实项目场景和你完成的具体动作。"})
        return {"items": items}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.resume_agent.orchestrator.settings.resume_agent_db_path",
                        str(tmp_path / "agent.db"))

    async def analyze(resume_text, jd_text, *, include_suggestions=True):
        assert include_suggestions is False, "Agent must not generate legacy one-shot rewrites"
        return fake_analysis()

    monkeypatch.setattr("app.services.resume_agent.orchestrator.analyze_job_fit", analyze)
    monkeypatch.setattr("app.services.resume_agent.evidence.get_llm_client", lambda: FakeEvidenceClient())
    get_resume_agent_orchestrator.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_resume_agent_orchestrator.cache_clear()


def create(client, resume=RESUME):
    response = client.post("/resume-agent/sessions", json={"resume_text": resume, "jd_text": JD})
    assert response.status_code == 200, response.text
    return response.json()


def answer(client, session, text="", answer_type="details", requirement="Docker"):
    question = next((q for q in session["pending_questions"] if q["requirement"] == requirement), None)
    return client.post(f"/resume-agent/sessions/{session['id']}/messages", json={"answers": [{
        "question_id": question["id"] if question else "", "requirement": requirement,
        "answer": text, "answer_type": answer_type,
    }]})


def choose(client, session, proposal, decision="accepted", version=None):
    return client.post(f"/resume-agent/sessions/{session['id']}/decisions", json={
        "proposal_id": proposal["id"], "decision": decision,
        "version": proposal["version"] if version is None else version,
    })


def proposal(session, requirement):
    return next(p for p in session["proposals"] if p["requirement"] == requirement)


def test_initial_questions_block_all_proposals_and_decisions(client):
    session = create(client)
    assert session["state"] == "needs_clarification"
    assert [q["requirement"] for q in session["pending_questions"]] == ["Docker"]
    assert session["proposals"] == []
    assert session["analysis"]["resume_rewrites"] == []
    assert session["analysis"]["interview_questions"] == []
    denied = client.post(f"/resume-agent/sessions/{session['id']}/decisions",
                         json={"proposal_id": "old-proposal", "decision": "accepted"})
    assert denied.status_code == 409


def test_uploaded_document_uses_same_confirmation_flow(client):
    response = client.post("/resume-agent/sessions/from-document", data={"jd_text": JD},
                           files={"resume": ("resume.txt", RESUME.encode(), "text/plain")})
    assert response.status_code == 200
    assert response.json()["resume_text"] == RESUME
    assert response.json()["proposals"] == []


@pytest.mark.parametrize("text,kind", [
    ("没有用过 Docker，也没有任何容器部署经历。", "details"),
    ("", "no_experience"), ("", "skip"), ("暂不回答", "details"),
])
def test_denied_or_skipped_experience_is_never_rewritten(client, text, kind):
    session = create(client)
    response = answer(client, session, text, kind)
    assert response.status_code == 200
    session = response.json()
    assert session["pending_questions"] == []
    assert [p["requirement"] for p in session["proposals"]] == ["Python"]
    assert session["analysis"]["match_score"] == 50
    assert [g["requirement"] for g in session["analysis"]["gaps"]] == ["Docker"]
    assert session["review_items"][1]["write_policy"] == "do_not_claim"


@pytest.mark.parametrize("text,kind", [
    ("不知道", "details"), ("不确定", "details"), ("会一点", "details"),
    ("Docker Docker Docker Docker Docker", "details"), ("", "unsure"),
])
def test_insufficient_answer_does_not_close_question(client, text, kind):
    session = create(client)
    question_id = session["pending_questions"][0]["id"]
    response = answer(client, session, text, kind)
    assert response.status_code == 200
    updated = response.json()
    assert updated["state"] == "needs_clarification"
    assert updated["proposals"] == []
    assert updated["pending_questions"][0]["id"] == question_id
    assert updated["facts"][-1]["confirmed"] is False


def test_blank_answer_is_rejected_without_saving_a_fact(client):
    session = create(client)
    assert answer(client, session, "   ").status_code == 400
    stored = client.get(f"/resume-agent/sessions/{session['id']}").json()
    assert stored["facts"] == []
    assert stored["pending_questions"] == session["pending_questions"]


def test_confirmed_details_refresh_all_analysis_and_preserve_initial_snapshot(client):
    session = create(client)
    updated = answer(client, session, DOCKER).json()
    assert updated["state"] == "awaiting_user_choice"
    assert not updated["pending_questions"]
    assert updated["facts"][-1]["confirmed"] is True
    assert updated["initial_analysis"]["match_score"] == 50
    analysis = updated["analysis"]
    assert analysis["match_score"] == analysis["score_breakdown"]["total_score"] == 100
    assert analysis["gaps"] == analysis["risk_items"] == analysis["risk_details"] == []
    assert analysis["analysis_overview"]["strong_match_count"] == 2
    assert updated["analysis_overview"] == analysis["analysis_overview"]
    assert any(e["text"] == DOCKER for e in analysis["evidence"])
    docker = proposal(updated, "Docker")
    assert "docker-compose 配置" in docker["after"]
    assert "结合用户补充事实" not in docker["after"]
    assert docker["evidence_basis"] == DOCKER
    stored = client.get(f"/resume-agent/sessions/{session['id']}").json()
    assert stored["initial_analysis"] == updated["initial_analysis"]
    assert stored["analysis"] == analysis


def test_sufficient_original_resume_does_not_require_unnecessary_questions(client):
    session = create(client, RESUME + "\n" + DOCKER)
    assert session["pending_questions"] == []
    assert session["state"] == "awaiting_user_choice"
    assert len(session["proposals"]) == 2


def test_unchanged_proposal_keeps_decision_when_other_fact_changes(client):
    session = answer(client, create(client), DOCKER).json()
    python = proposal(session, "Python")
    session = choose(client, session, python).json()
    session = answer(client, session, DOCKER_NEW).json()
    kept = proposal(session, "Python")
    assert (kept["id"], kept["version"], kept["status"]) == (python["id"], 1, "accepted")


def test_changed_proposal_requires_new_consent_and_rejects_stale_requests(client):
    session = answer(client, create(client), DOCKER).json()
    old = proposal(session, "Docker")
    session = choose(client, session, old).json()
    session = answer(client, session, DOCKER_NEW).json()
    new = proposal(session, "Docker")
    assert new["status"] == "proposed"
    assert new["version"] == 2
    assert new["id"] != old["id"]
    assert "健康检查脚本" in new["after"]
    assert choose(client, session, old).status_code == 409
    assert choose(client, session, new, version=1).status_code == 409
    assert choose(client, session, new).status_code == 200


def test_revoking_then_restoring_evidence_does_not_restore_old_consent(client):
    session = answer(client, create(client), DOCKER).json()
    old = proposal(session, "Docker")
    session = choose(client, session, old).json()
    session = answer(client, session, "", "no_experience").json()
    assert all(p["requirement"] != "Docker" for p in session["proposals"])
    assert session["analysis"]["match_score"] == 50
    session = answer(client, session, DOCKER).json()
    restored = proposal(session, "Docker")
    assert restored["status"] == "proposed"
    assert restored["id"] != old["id"]
    assert restored["version"] > old["version"]


def test_uncertain_correction_reopens_clarification_and_blocks_old_acceptance(client):
    session = answer(client, create(client), DOCKER).json()
    old = proposal(session, "Docker")
    session = answer(client, session, "不确定").json()
    assert session["state"] == "needs_clarification"
    assert session["proposals"] == []
    assert choose(client, session, old).status_code == 409


def test_completion_requires_every_proposal_to_be_decided(client):
    session = answer(client, create(client), DOCKER).json()
    session = choose(client, session, proposal(session, "Python")).json()
    assert session["state"] == "awaiting_user_choice"
    session = choose(client, session, proposal(session, "Docker"), "rejected").json()
    assert session["state"] == "completed"
    assert [p["requirement"] for p in session["proposals"] if p["status"] == "accepted"] == ["Python"]


def test_all_explicitly_absent_completes_without_fabricated_proposals(client):
    session = create(client, "我正在学习计算机基础课程，目前还没有参与实际的后端开发项目或部署工作。")
    for requirement in ("Python", "Docker"):
        response = answer(client, session, "", "no_experience", requirement)
        assert response.status_code == 200
        session = response.json()
    assert session["state"] == "completed"
    assert session["proposals"] == []
    assert session["analysis"]["match_score"] == 0


def test_model_failure_keeps_clarification_open(client, monkeypatch):
    class FailedClient:
        async def analyze(self, *args):
            raise RuntimeError("model unavailable")
    monkeypatch.setattr("app.services.resume_agent.evidence.get_llm_client", lambda: FailedClient())
    session = create(client)
    assert session["state"] == "needs_clarification"
    assert len(session["pending_questions"]) == 2
    assert session["proposals"] == []


def test_legacy_sessions_cannot_expose_or_accept_unverified_proposals(client):
    from app.schemas.resume_agent import RewriteProposal

    session = create(client)
    repository = get_resume_agent_orchestrator().repository
    old = repository.get_session(session["id"])
    old.initial_analysis = None
    old.proposals = [RewriteProposal(requirement="Docker", after="部署了服务", status=ProposalStatus.PROPOSED)]
    old.state = ResumeAgentState.AWAITING_USER_CHOICE
    old.pending_questions = []
    repository.save_session(old)
    restored = client.get(f"/resume-agent/sessions/{old.id}").json()
    assert restored["state"] == "needs_clarification"
    assert restored["proposals"] == []
    assert client.post(f"/resume-agent/sessions/{old.id}/decisions", json={
        "proposal_id": old.proposals[0].id, "decision": "accepted"
    }).status_code == 409


def duplicate_requirements(monkeypatch, category="project"):
    async def analyze(*args, **kwargs):
        return JobFitAnalysis(match_score=0, summary="", requirement_analysis=[
            {"requirement": "Docker", "category": "skill", "level": "required"},
            {"requirement": "Docker", "category": category, "level": "required"},
        ])
    monkeypatch.setattr("app.services.resume_agent.orchestrator.analyze_job_fit", analyze)


def answer_id(client, session, requirement_id, text=DOCKER):
    question = next((q for q in session["pending_questions"] if q["requirement_id"] == requirement_id), None)
    return client.post(f"/resume-agent/sessions/{session['id']}/messages", json={"answers": [{
        "question_id": question["id"] if question else "", "requirement_id": requirement_id,
        "answer": text,
    }]})


@pytest.mark.parametrize("category", ["project", "skill"])
def test_same_name_requirements_have_distinct_persistent_identity(client, monkeypatch, category):
    duplicate_requirements(monkeypatch, category)
    session = create(client)
    ids = [item["requirement_id"] for item in session["initial_analysis"]["requirement_analysis"]]
    assert len(set(ids)) == 2
    assert len({q["id"] for q in session["pending_questions"]}) == 2
    updated = answer_id(client, session, ids[0]).json()
    assert [q["requirement_id"] for q in updated["pending_questions"]] == [ids[1]]
    assert updated["proposals"] == []
    assert set(updated["assessments"]) == set(ids)
    assert updated["facts"][0]["requirement_id"] == ids[0]
    restored = client.get(f"/resume-agent/sessions/{session['id']}").json()
    assert restored["pending_questions"] == updated["pending_questions"]
    assert [i["requirement_id"] for i in restored["analysis"]["requirement_analysis"]] == ids


def test_same_name_decisions_and_versions_do_not_leak_to_each_other(client, monkeypatch):
    duplicate_requirements(monkeypatch)
    session = create(client)
    skill_id, project_id = [item["requirement_id"] for item in session["initial_analysis"]["requirement_analysis"]]
    session = answer_id(client, session, skill_id).json()
    session = answer_id(client, session, project_id).json()
    assert len(session["proposals"]) == 2
    skill = next(p for p in session["proposals"] if p["requirement_id"] == skill_id)
    session = choose(client, session, skill).json()
    session = answer_id(client, session, project_id, DOCKER_NEW).json()
    skill = next(p for p in session["proposals"] if p["requirement_id"] == skill_id)
    project = next(p for p in session["proposals"] if p["requirement_id"] == project_id)
    assert skill["status"] == "accepted" and skill["version"] == 1
    assert project["status"] == "proposed" and project["version"] == 2
    assert len(session["proposal_history"]) == 2


def test_ambiguous_name_only_correction_is_rejected(client, monkeypatch):
    duplicate_requirements(monkeypatch)
    session = create(client)
    response = client.post(f"/resume-agent/sessions/{session['id']}/messages", json={"answers": [{
        "question_id": "", "requirement": "Docker", "answer": DOCKER,
    }]})
    assert response.status_code == 400
    assert client.get(f"/resume-agent/sessions/{session['id']}").json()["facts"] == []


def test_question_id_cannot_be_used_to_answer_another_same_name_requirement(client, monkeypatch):
    duplicate_requirements(monkeypatch)
    session = create(client)
    first, second = session["pending_questions"]
    response = client.post(f"/resume-agent/sessions/{session['id']}/messages", json={"answers": [{
        "question_id": first["id"], "requirement_id": second["requirement_id"], "answer": DOCKER,
    }]})
    assert response.status_code == 400


def strip_requirement_ids(session):
    id_to_name = {item.requirement_id: item.requirement for item in session.initial_analysis.requirement_analysis}
    session.assessments = {id_to_name[key]: value for key, value in session.assessments.items()}
    for analysis in (session.initial_analysis, session.analysis):
        for item in analysis.requirement_analysis:
            item.requirement_id = ""
    for attribute in ("facts", "proposals", "proposal_history", "review_items", "pending_questions"):
        for item in getattr(session, attribute):
            item.requirement_id = ""


def test_unique_legacy_keys_migrate_once_without_losing_decisions(client):
    session = answer(client, create(client), DOCKER).json()
    session = choose(client, session, proposal(session, "Python")).json()
    repository = get_resume_agent_orchestrator().repository
    legacy = repository.get_session(session["id"])
    strip_requirement_ids(legacy)
    repository.save_session(legacy)
    migrated = client.get(f"/resume-agent/sessions/{session['id']}").json()
    assert proposal(migrated, "Python")["status"] == "accepted"
    assert proposal(migrated, "Python")["id"] == proposal(session, "Python")["id"]
    assert migrated["facts"][0]["requirement_id"] == proposal(migrated, "Docker")["requirement_id"]
    repeated = client.get(f"/resume-agent/sessions/{session['id']}").json()
    assert repeated["initial_analysis"] == migrated["initial_analysis"]
    assert repeated["proposals"] == migrated["proposals"]
    assert set(repository.get_session(session["id"]).assessments) == set(migrated["assessments"])


def test_collided_legacy_fact_is_not_assigned_to_both_requirements(client, monkeypatch):
    duplicate_requirements(monkeypatch)
    session = create(client)
    ids = [q["requirement_id"] for q in session["pending_questions"]]
    session = answer_id(client, session, ids[0]).json()
    session = answer_id(client, session, ids[1]).json()
    repository = get_resume_agent_orchestrator().repository
    legacy = repository.get_session(session["id"])
    strip_requirement_ids(legacy)
    repository.save_session(legacy)
    migrated = client.get(f"/resume-agent/sessions/{session['id']}").json()
    assert migrated["proposals"] == []
    assert len(migrated["pending_questions"]) == 2
    assert migrated["assessments"] == {}
    assert all(not fact["confirmed"] for fact in migrated["facts"])
    assert len({q["requirement_id"] for q in migrated["pending_questions"]}) == 2
