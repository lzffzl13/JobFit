import pytest

from app.schemas.resume_agent import AgentMessage, MessageRole
from app.services.resume_agent.orchestrator import get_resume_agent_orchestrator
from tests.test_resume_agent_api import (
    DOCKER,
    DOCKER_NEW,
    RESUME,
    answer,
    choose,
    create,
)
from tests.test_resume_agent_api import (
    client as client,
)


def ready(client):
    session = answer(client, create(client), DOCKER).json()
    for proposal in session["proposals"]:
        session = choose(client, session, proposal).json()
    return session


def endpoint(session, suffix=""):
    return f"/resume-agent/sessions/{session['id']}/document{suffix}"


def preview(client, session):
    response = client.post(endpoint(session, "/preview"))
    assert response.status_code == 200, response.text
    return response.json()


def apply(client, session):
    return client.post(endpoint(session, "/apply"), json={
        "token": session["preview"]["token"], "expected_revision": session["document"]["revision"],
    })


def text(session):
    return "".join(f["text"] for f in session["document"]["fields"])


def test_selection_preview_apply_export_and_reload_are_distinct(client):
    session = ready(client)
    assert text(session) == RESUME
    assert session["document"]["revision"] == 0
    session = preview(client, session)
    assert text(session) == RESUME
    assert {c["operation"] for c in session["preview"]["changes"]} == {"replace", "add"}
    assert all(c["source_ids"] and c["checks"] and c["path"].startswith("/fields/")
               for c in session["preview"]["changes"])
    expected = RESUME[1:] + "\n\n" + DOCKER[1:]
    assert session["preview"]["text"] == expected
    updated = apply(client, session)
    assert updated.status_code == 200
    result = updated.json()
    assert text(result) == expected
    assert result["preview"] is None
    assert [v["revision"] for v in result["document"]["versions"]] == [0, 1]
    assert apply(client, session).status_code == 409
    export = client.get(endpoint(session, "/export"))
    assert export.text == expected
    assert "attachment" in export.headers["content-disposition"]
    reloaded = client.get(f"/resume-agent/sessions/{session['id']}").json()
    assert reloaded["document"] == result["document"]
    assert client.post(endpoint(session, "/preview")).status_code == 400


def test_token_is_bound_to_session_jd_and_proposal_choice(client):
    first = preview(client, ready(client))
    second = preview(client, ready(client))
    assert client.post(endpoint(second, "/apply"), json={
        "token": first["preview"]["token"], "expected_revision": 0,
    }).status_code == 409
    rejected = choose(client, first, first["proposals"][0], "rejected").json()
    assert rejected["preview"] is None
    assert apply(client, first).status_code == 409
    service = get_resume_agent_orchestrator()
    stored = service.get_session(second["id"])
    stored.jd_text += " 岗位要求已更正。"
    service.repository.save_session(stored)
    assert apply(client, second).status_code == 409


def test_manual_edit_and_restore_preserve_history_and_reject_old_preview(client):
    session = preview(client, ready(client))
    manual = RESUME + "\r\n\r\n" + "用户手动补充：仅进行本地测试。"
    edited = client.put(endpoint(session), json={"text": manual, "expected_revision": 0}).json()
    assert text(edited) == manual
    assert edited["document"]["revision"] == 1
    assert apply(client, session).status_code == 409
    assert client.put(endpoint(session), json={"text": RESUME, "expected_revision": 0}).status_code == 409
    restored = client.post(endpoint(session, "/restore"), json={"revision": 0, "expected_revision": 1}).json()
    assert text(restored) == RESUME
    assert [v["revision"] for v in restored["document"]["versions"]] == [0, 1, 2]
    assert "".join(f["text"] for f in restored["document"]["versions"][1]["fields"]) == manual
    assert client.post(endpoint(session, "/restore"), json={"revision": 90, "expected_revision": 2}).status_code == 404


@pytest.mark.parametrize("addition", ["，使用 Kubernetes", "，提升性能 90%", "，主导整个研发团队"])
def test_program_validator_blocks_fabricated_skills_metrics_and_roles(client, addition):
    session = ready(client)
    service = get_resume_agent_orchestrator()
    stored = service.get_session(session["id"])
    stored.proposals[0].after += addition
    service.repository.save_session(stored)
    assert client.post(endpoint(session, "/preview")).status_code == 409
    assert client.get(endpoint(session, "/export")).text == RESUME


def test_ambiguous_source_and_changed_original_are_not_overwritten(client):
    session = answer(client, create(client, RESUME + "\n\n" + RESUME), "", "no_experience").json()
    session = choose(client, session, session["proposals"][0]).json()
    assert client.post(endpoint(session, "/preview")).status_code == 409
    session = ready(client)
    client.put(endpoint(session), json={"text": "手动修改后的完整简历。" * 4, "expected_revision": 0})
    assert client.post(endpoint(session, "/preview")).status_code == 409


def test_fact_correction_marks_applied_text_outdated_and_replaces_appended_field(client):
    session = apply(client, preview(client, ready(client))).json()
    changed = answer(client, session, DOCKER_NEW).json()
    assert changed["document"]["outdated_requirements"]
    assert client.get(endpoint(session, "/export")).status_code == 409
    docker = next(p for p in changed["proposals"] if p["requirement"] == "Docker")
    changed = choose(client, changed, docker).json()
    changed = apply(client, preview(client, changed)).json()
    assert DOCKER_NEW[1:] in text(changed)
    assert DOCKER[1:] not in text(changed)
    assert changed["document"]["outdated_requirements"] == []
    assert client.get(endpoint(changed, "/export")).status_code == 200


def test_revoked_facts_block_export_until_original_is_restored(client):
    session = apply(client, preview(client, ready(client))).json()
    revoked = answer(client, session, "", "no_experience").json()
    assert client.get(endpoint(revoked, "/export")).status_code == 409
    restored = client.post(endpoint(session, "/restore"), json={"revision": 0, "expected_revision": 1}).json()
    assert restored["document"]["outdated_requirements"] == []
    assert client.get(endpoint(restored, "/export")).text == RESUME


def test_same_source_for_two_requirements_is_one_field_change(client):
    session = ready(client)
    service = get_resume_agent_orchestrator()
    stored = service.get_session(session["id"])
    original = stored.proposals[0]
    other = original.model_copy(deep=True)
    other.id, other.requirement_id = "proposal_second", "req_second"
    stored.proposals = [original, other]
    stored.assessments[other.requirement_id] = stored.assessments[original.requirement_id].model_copy()
    service.repository.save_session(stored)
    result = preview(client, session)["preview"]
    assert len(result["changes"]) == 1
    assert len(result["changes"][0]["proposal_ids"]) == 2
    assert result["text"] == RESUME[1:]


def test_repository_conflict_does_not_save_messages_or_document_snapshot(client):
    session = ready(client)
    service = get_resume_agent_orchestrator()
    first, stale = service.get_session(session["id"]), service.get_session(session["id"])
    first.summary = "first writer"
    service.repository.save_session(first)
    stale.messages.append(AgentMessage(role=MessageRole.USER, content="stale unsaved message"))
    stale.document.revision = 100
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as conflict:
        service.repository.save_session(stale)
    assert conflict.value.status_code == 409
    loaded = service.get_session(session["id"])
    assert loaded.summary == "first writer"
    assert loaded.document.revision == 0
    assert all(m.content != "stale unsaved message" for m in loaded.messages)
