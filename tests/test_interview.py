import json

import pytest

from app.services.resume_agent.orchestrator import get_resume_agent_orchestrator
from tests.test_resume_agent_api import RESUME, create
from tests.test_resume_agent_api import client as client


class FeedbackClient:
    def __init__(self, followup=False, bad_source=False):
        self.followup, self.bad_source = followup, bad_source

    async def analyze(self, system_prompt, user_prompt):
        data = json.loads(user_prompt)
        assert data["resume"] == RESUME
        return {"score": 65, "evidence": ["不存在的回答" if self.bad_source else data["answer"]],
                "strengths": ["说明了当前思路"], "improvements": ["补充方案的验证方法"],
                "needs_followup": self.followup}


def url(session, suffix=""):
    return f"/resume-agent/sessions/{session['id']}/interview{suffix}"


def run(session):
    return next(r for r in session["interviews"] if r["id"] == session["active_interview_id"])


def start(client, count=2):
    response = client.post(url(create(client)), json={"question_count": count})
    assert response.status_code == 200, response.text
    return response.json()


def guard(session):
    current = run(session)
    return {"run_id": current["id"], "expected_revision": current["revision"]}


def answer(client, session, skip=False):
    return client.post(url(session, "/answers"), json={**guard(session),
        "question_id": run(session)["current_question"]["id"],
        "answer": "我会先拆分需求，再验证接口与异常路径。", "skip": skip,
    })


def control(client, session, action):
    return client.post(url(session, "/control"), json={**guard(session), "action": action})


def test_plan_prioritizes_gaps_and_resume_restores_same_question(client):
    session = start(client)
    current = run(session)
    assert current["plan"][0]["requirement"] == "Docker"
    assert current["source_document_revision"] == 0
    assert len(current["plan"]) == 2
    paused = control(client, session, "pause").json()
    assert run(paused)["state"] == "paused"
    assert answer(client, paused).status_code == 409
    get_resume_agent_orchestrator.cache_clear()
    reloaded = client.get(f"/resume-agent/sessions/{session['id']}").json()
    assert run(reloaded) == run(paused)
    resumed = control(client, reloaded, "resume").json()
    assert run(resumed)["current_question"]["id"] == current["current_question"]["id"]
    assert client.post(url(resumed), json={}).status_code == 409


def test_full_loop_evaluates_advances_reports_and_keeps_resume_facts_separate(client, monkeypatch):
    monkeypatch.setattr("app.services.interview.get_llm_client", lambda: FeedbackClient())
    session = start(client)
    original_facts = session["facts"]
    stale = session
    session = answer(client, session).json()
    assert run(session)["current_question"]["requirement"] == "Python"
    assert answer(client, stale).status_code == 409
    session = answer(client, session).json()
    assert run(session)["state"] == "completed"
    assert run(session)["current_question"] is None
    report = run(session)["report"]
    assert report["average_score"] == 65
    assert report["assessed_count"] == report["answered_count"] == 2
    assert report["improvements"]
    assert session["facts"] == original_facts
    assert session["document"]["revision"] == 0
    assert control(client, session, "resume").status_code == 409
    another = client.post(url(session), json={"question_count": 1}).json()
    assert len(another["interviews"]) == 2
    assert another["interviews"][0]["report"] == report
    assert control(client, session, "finish").status_code == 409


def test_followup_is_limited_to_one_and_original_question_keeps_its_identity(client, monkeypatch):
    monkeypatch.setattr("app.services.interview.get_llm_client", lambda: FeedbackClient(followup=True))
    session = start(client, 1)
    first_id = run(session)["current_question"]["id"]
    session = answer(client, session).json()
    assert run(session)["current_question"]["kind"] == "followup"
    assert run(session)["current_question"]["id"] != first_id
    assert run(session)["position"] == 0
    session = answer(client, session).json()
    assert run(session)["state"] == "completed"
    assert len(run(session)["turns"]) == 2


@pytest.mark.parametrize("bad", ["source", "network"])
def test_invalid_or_failed_evaluation_preserves_answer_without_inventing_a_score(client, monkeypatch, bad):
    class FailedClient:
        async def analyze(self, *_):
            raise RuntimeError("offline")
    monkeypatch.setattr("app.services.interview.get_llm_client", lambda:
                        FailedClient() if bad == "network" else FeedbackClient(bad_source=True))
    session = answer(client, start(client, 1)).json()
    current = run(session)
    assert current["state"] == "completed"
    assert current["turns"][0]["answer"]
    assert current["turns"][0]["feedback"]["score"] is None
    assert current["report"]["average_score"] is None
    assert current["report"]["assessed_count"] == 0


def test_skip_finish_blank_and_stale_question_guards(client):
    session = start(client)
    assert client.post(url(session, "/answers"), json={**guard(session),
        "question_id": run(session)["current_question"]["id"], "answer": " "}).status_code == 400
    assert client.post(url(session, "/answers"), json={**guard(session),
        "question_id": "stale", "answer": "回答"}).status_code == 409
    session = answer(client, session, True).json()
    assert run(session)["position"] == 1
    session = control(client, session, "finish").json()
    assert run(session)["report"]["skipped_count"] == 1
    assert run(session)["report"]["answered_count"] == 0
    assert run(session)["report"]["average_score"] is None


def test_interview_material_remains_bound_to_starting_document_version(client, monkeypatch):
    monkeypatch.setattr("app.services.interview.get_llm_client", lambda: FeedbackClient())
    session = start(client, 1)
    edit_url = f"/resume-agent/sessions/{session['id']}/document"
    client.put(edit_url, json={"text": RESUME + "\n新的手动简历版本。", "expected_revision": 0})
    session = answer(client, session).json()
    assert session["document"]["revision"] == 1
    assert run(session)["source_document_revision"] == 0
    assert run(session)["resume_text"] == RESUME


def test_preferred_gap_is_planned_before_a_fully_supported_required_item(client, monkeypatch):
    from tests.test_resume_agent_api import fake_analysis

    async def analyze(*_, **__):
        analysis = fake_analysis()
        item = analysis.requirement_analysis[1].model_copy(deep=True)
        item.requirement, item.level = "Redis", "preferred"
        analysis.requirement_analysis.append(item)
        return analysis

    monkeypatch.setattr("app.services.resume_agent.orchestrator.analyze_job_fit", analyze)
    session = start(client, 2)
    assert [q["requirement"] for q in run(session)["plan"]] == ["Docker", "Redis"]
