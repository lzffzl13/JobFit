from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_analyze_endpoint_with_text_resume():
    client = TestClient(app)
    resume_path = Path("samples/resume.txt")
    jd_text = Path("samples/jd_ai_app.txt").read_text(encoding="utf-8")

    with resume_path.open("rb") as resume_file:
        response = client.post(
            "/jobfit/analyze",
            files={"resume": ("resume.txt", resume_file, "text/plain")},
            data={"jd_text": jd_text},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["match_score"] >= 0
    assert isinstance(payload["fallback_used"], bool)
    assert payload["model_used"]
    assert payload["evidence"]
