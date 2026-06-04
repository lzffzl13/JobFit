"""Unit tests for validator.py — clamp, count, confidence warnings."""

from app.services.validator import validate_and_score


def _req(name, priority="core", evidence_ratio=0.8, confidence=0.9):
    return {
        "name": name,
        "priority": priority,
        "evidence_ratio": evidence_ratio,
        "confidence": confidence,
        "reasoning": "test",
        "evidence_quote": "",
    }


def test_clamp_match_score_to_0_100():
    result = validate_and_score(
        llm_requirements=[],
        resume_text="",
        llm_match_score=150,
    )
    assert result["score"] == 100

    result = validate_and_score(
        llm_requirements=[],
        resume_text="",
        llm_match_score=-10,
    )
    assert result["score"] == 0


def test_clamp_bonus_and_extra_score():
    result = validate_and_score(
        llm_requirements=[],
        resume_text="",
        llm_match_score=50,
        llm_bonus_score=20,
        llm_extra_score=-5,
    )
    assert result["score_breakdown"].bonus_score == 15
    assert result["score_breakdown"].extra_score == 0


def test_core_bonus_counts():
    reqs = [
        _req("Python", "core", 0.9),
        _req("MySQL", "core", 0.0),
        _req("Docker", "bonus", 0.5),
        _req("K8s", "bonus", 0.0),
    ]
    result = validate_and_score(
        llm_requirements=reqs,
        resume_text="",
        llm_match_score=70,
    )
    sb = result["score_breakdown"]
    assert sb.core_matched == 1
    assert sb.core_total == 2
    assert sb.bonus_matched == 1
    assert sb.bonus_total == 2


def test_low_confidence_generates_warning():
    reqs = [_req("Python", "core", 0.8, confidence=0.1)]
    result = validate_and_score(
        llm_requirements=reqs,
        resume_text="",
        llm_match_score=70,
    )
    assert len(result["warnings"]) == 1
    assert "低置信度" in result["warnings"][0]
    assert len(result["score_breakdown"].evidence_notes) == 1


def test_high_confidence_no_warning():
    reqs = [_req("Python", "core", 0.8, confidence=0.9)]
    result = validate_and_score(
        llm_requirements=reqs,
        resume_text="",
        llm_match_score=70,
    )
    assert result["warnings"] == []


def test_risk_items_from_low_evidence_core():
    reqs = [
        _req("Python", "core", 0.9),
        _req("MySQL", "core", 0.3),
        _req("Docker", "bonus", 0.1),  # bonus, not core
    ]
    result = validate_and_score(
        llm_requirements=reqs,
        resume_text="",
        llm_match_score=60,
    )
    # MySQL has ratio < 0.85 and is core → should be in risk_items
    assert any("MySQL" in r for r in result["risk_items"])
    # Docker is bonus → not in risk_items
    assert not any("Docker" in r for r in result["risk_items"])


def test_risk_items_capped_at_6():
    reqs = [_req(f"Skill{i}", "core", 0.1) for i in range(10)]
    result = validate_and_score(
        llm_requirements=reqs,
        resume_text="",
        llm_match_score=30,
    )
    assert len(result["risk_items"]) <= 6


def test_evidence_ratio_clamped():
    reqs = [_req("Python", "core", evidence_ratio=1.5, confidence=0.9)]
    result = validate_and_score(
        llm_requirements=reqs,
        resume_text="",
        llm_match_score=70,
    )
    validated = result["validated_requirements"][0]
    assert validated["evidence_ratio"] == 1.0


def test_missing_requirement_fields():
    """Validator should handle incomplete requirement dicts gracefully."""
    reqs = [{"name": "Python"}]
    result = validate_and_score(
        llm_requirements=reqs,
        resume_text="",
        llm_match_score=70,
    )
    # No priority → not counted as core or bonus
    assert result["score_breakdown"].core_total == 0
    assert result["validated_requirements"][0]["evidence_ratio"] == 0.0
    assert result["validated_requirements"][0]["confidence"] == 0.0
