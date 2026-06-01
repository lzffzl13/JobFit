from app.services.llm import local_fallback_analysis

BACKEND_JD = (
    "Need Python/Java basics. Need one of Django/Flask/FastAPI. "
    "Need backend API development. Need MySQL SQL CRUD and Redis cache."
)


def test_medium_backend_resume_is_not_inflated_to_top_score():
    analysis = local_fallback_analysis(
        resume_text=(
            "Skills: Python, Flask, MySQL, Redis, Docker, Git, pytest. "
            "Project: Python Flask blog backend system, built RESTful APIs for users and posts, "
            "used MySQL CRUD and Redis cache. Used Docker deployment and pytest API tests. "
            "Reduced API latency by 20%."
        ),
        jd_text=BACKEND_JD,
    )

    assert 75 <= analysis.match_score <= 88
    assert analysis.score_breakdown.core_points == 70
    assert analysis.score_breakdown.raw_score < 90


def test_low_backend_resume_hits_no_backend_project_cap():
    analysis = local_fallback_analysis(
        resume_text=(
            "Skills: Python, MySQL, FastAPI. "
            "Testing intern, wrote API test cases and test reports. "
            "No backend project experience."
        ),
        jd_text=BACKEND_JD,
    )

    assert analysis.match_score <= 55
    assert analysis.score_breakdown.final_cap == 55
    assert any("无后端/Web/接口开发证据" in reason for reason in analysis.score_breakdown.cap_reasons)


def test_operations_resume_cannot_score_like_technical_resume():
    analysis = local_fallback_analysis(
        resume_text=(
            "Operations specialist. Managed campaigns, user growth, content operation, "
            "conversion rate analysis, community activity planning, sales support. "
            "Awarded marketing excellence prize."
        ),
        jd_text=BACKEND_JD,
    )

    assert 0 <= analysis.match_score <= 20
    assert analysis.score_breakdown.final_cap == 45
    assert analysis.score_breakdown.bonus_extra == 0
    assert any("无编程语言证据" in reason for reason in analysis.score_breakdown.cap_reasons)


def test_keyword_stuffing_without_project_is_capped():
    analysis = local_fallback_analysis(
        resume_text=(
            "Skills: Python, FastAPI, Flask, MySQL, Redis, Docker, Git, Linux, pytest. "
            "Self evaluation: familiar with backend technology stack."
        ),
        jd_text=BACKEND_JD,
    )

    assert analysis.match_score <= 70
    assert analysis.score_breakdown.final_cap <= 70
    assert any("技能栏" in reason for reason in analysis.score_breakdown.cap_reasons)
