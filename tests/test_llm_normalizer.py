from app.schemas.jobfit import JobFitAnalysis
from app.services.llm import normalize_llm_payload


def test_normalize_llm_payload_accepts_string_lists_and_requirement_keys():
    raw = {
        "score": "82",
        "summary": "匹配度较高",
        "jd_requirements": [{"requirement": "Python"}, "MySQL"],
        "matched_strengths": ["熟悉 Python", {"requirement": "SQL", "evidence": "写过 SQL"}],
        "gaps": ["缺少 Spring Boot"],
        "resume_rewrites": ["补充接口开发经历"],
        "interview_questions": ["请介绍 Python 项目"],
    }

    normalized = normalize_llm_payload(raw)
    analysis = JobFitAnalysis.model_validate(
        {
            **normalized,
            "model_used": "deepseek-chat",
            "fallback_used": False,
        }
    )

    assert analysis.match_score == 82
    assert analysis.jd_requirements[0].name == "Python"
    assert analysis.matched_strengths[0].requirement == "熟悉 Python"
    assert analysis.gaps[0].requirement == "缺少 Spring Boot"
    assert analysis.resume_rewrites[0].after == "补充接口开发经历"
    assert analysis.interview_questions[0].question == "请介绍 Python 项目"
