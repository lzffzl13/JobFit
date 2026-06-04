from app.schemas.jobfit import JobFitAnalysis
from app.services.llm import normalize_llm_payload


def test_normalize_llm_payload_accepts_string_lists_and_requirement_keys():
    raw = {
        "match_score": "82",
        "bonus_score": "10",
        "extra_score": "5",
        "summary": "匹配度较高",
        "requirements": [
            {"name": "Python", "evidence_ratio": 0.85, "confidence": 0.9},
            "MySQL",
        ],
        "gaps": ["缺少 Spring Boot"],
        "resume_rewrites": ["补充接口开发经历"],
        "interview_questions": ["请介绍 Python 项目"],
    }

    normalized = normalize_llm_payload(raw)

    assert normalized["match_score"] == 82
    assert normalized["bonus_score"] == 10
    assert normalized["extra_score"] == 5
    assert normalized["summary"] == "匹配度较高"
    assert normalized["requirements"][0]["name"] == "Python"
    assert normalized["requirements"][0]["evidence_ratio"] == 0.85
    assert normalized["requirements"][1]["name"] == "MySQL"
    assert normalized["gaps"][0]["requirement"] == "缺少 Spring Boot"
    assert normalized["resume_rewrites"][0]["after"] == "补充接口开发经历"
    assert normalized["interview_questions"][0]["question"] == "请介绍 Python 项目"


def test_normalize_handles_missing_fields():
    raw = {"match_score": 75}
    normalized = normalize_llm_payload(raw)

    assert normalized["match_score"] == 75
    assert normalized["bonus_score"] == 0
    assert normalized["extra_score"] == 0
    assert normalized["summary"] == ""
    assert normalized["requirements"] == []
    assert normalized["gaps"] == []


def test_normalize_clamps_score():
    assert normalize_llm_payload({"match_score": 150})["match_score"] == 100
    assert normalize_llm_payload({"match_score": -5})["match_score"] == 0
    assert normalize_llm_payload({"match_score": "abc"})["match_score"] == 0


def test_normalize_accepts_score_alias():
    """LLM might return 'score' instead of 'match_score'."""
    assert normalize_llm_payload({"score": 80})["match_score"] == 80


def test_normalize_accepts_conclusion_alias():
    """LLM might return 'conclusion' or 'analysis' instead of 'summary'."""
    assert normalize_llm_payload({"conclusion": "匹配度高"})["summary"] == "匹配度高"
    assert normalize_llm_payload({"analysis": "一般"})["summary"] == "一般"


def test_normalize_requirement_field_aliases():
    """LLM might use 'requirement'/'skill' instead of 'name', 'ratio' instead of 'evidence_ratio'."""
    raw = {
        "requirements": [
            {"requirement": "Python", "ratio": 0.9, "reason": "strong"},
            {"skill": "MySQL", "evidence": "used in project"},
        ]
    }
    norm = normalize_llm_payload(raw)
    assert norm["requirements"][0]["name"] == "Python"
    assert norm["requirements"][0]["evidence_ratio"] == 0.9
    assert norm["requirements"][0]["reasoning"] == "strong"
    assert norm["requirements"][1]["name"] == "MySQL"
    assert norm["requirements"][1]["evidence_quote"] == "used in project"


def test_normalize_gap_field_aliases():
    """LLM might use 'name'/'skill'/'advice' instead of standard fields."""
    raw = {"gaps": [{"name": "Docker", "advice": "学习容器化"}]}
    norm = normalize_llm_payload(raw)
    assert norm["gaps"][0]["requirement"] == "Docker"
    assert norm["gaps"][0]["suggestion"] == "学习容器化"


def test_normalize_empty_raw():
    """Completely empty input should return valid defaults."""
    norm = normalize_llm_payload({})
    assert norm["match_score"] == 0
    assert norm["requirements"] == []
    assert norm["gaps"] == []
    assert norm["summary"] == ""


def test_normalize_none_requirements():
    """None requirements should become empty list, not crash."""
    norm = normalize_llm_payload({"requirements": None})
    assert norm["requirements"] == []


def test_normalize_question_field_aliases():
    """LLM might use 'content'/'title' instead of 'question'."""
    raw = {"interview_questions": [{"content": "介绍项目", "topic": "技术"}]}
    norm = normalize_llm_payload(raw)
    assert norm["interview_questions"][0]["question"] == "介绍项目"
    assert norm["interview_questions"][0]["focus"] == "技术"
