"""Source and semantic boundary regressions; no model/network access."""

import asyncio

import pytest

from app.schemas.jobfit import MatchDetail, RequirementAnalysis
from app.schemas.resume_agent import EvidenceAssessment, UserFact
from app.services.matcher import summarize_matches
from app.services.resume_agent.evidence import (
    assess_requirements,
    explicit_outcome,
    validate_assessment,
)
from app.services.resume_agent.proposal_generator import build_proposals
from app.services.resume_agent.reviewer import review_requirements


def supported(evidence, **updates):
    result = {"outcome": "supported", "coverage": "full", "evidence": evidence,
              "context": "个人项目", "action": "使用 Docker 部署测试环境"}
    result.update(updates)
    return result


def test_concrete_source_is_usable_without_inventing_metrics():
    source = "我在个人项目中使用 Docker 部署测试环境。"
    assessment = validate_assessment(supported(source), source, "skill")
    assert assessment.outcome == "supported"
    assert assessment.result == ""
    review = review_requirements([RequirementAnalysis(requirement="Docker")],
                                 assessments={"Docker": assessment})
    draft = build_proposals(review)[0]
    assert draft.after == "在个人项目中使用 Docker 部署测试环境。"
    assert "生产" not in draft.after
    assert "主导" not in draft.after


def test_full_evidence_line_after_heading_is_accepted():
    evidence = "我在个人项目中使用 Docker 部署测试环境。"
    source = "项目经历\n  " + evidence + "\n技能"
    assert validate_assessment(supported(evidence), source, "skill").outcome == "supported"


@pytest.mark.parametrize("source,raw", [
    ("了解 Docker", supported("在公司项目中使用 Docker 部署生产环境。")),
    ("Docker", supported("Docker", context="Docker", action="Docker")),
    ("我只了解 Docker，没有部署过任何项目。", supported("我只了解 Docker，没有部署过任何项目。")),
    ("我在个人项目中使用 Docker 部署测试环境。",
     supported("我在个人项目中使用 Docker 部署测试环境。", result="性能提升 50%")),
    ("我在个人项目中使用 Docker 部署测试环境。",
     supported("我在个人项目中使用 Docker 部署测试环境。", action="主导生产环境部署")),
    ("I have never used Docker.", supported("I have never used Docker.")),
    ("没有在个人项目中使用 Docker 部署测试环境。", supported("在个人项目中使用 Docker 部署测试环境。")),
    ("如果在个人项目中使用 Docker 部署测试环境。", supported("如果在个人项目中使用 Docker 部署测试环境。")),
])
def test_untraceable_or_weak_evidence_cannot_become_a_proposal(source, raw):
    assessment = validate_assessment(raw, source, "skill")
    assert assessment.outcome == "insufficient"
    reviews = review_requirements([RequirementAnalysis(requirement="Docker", status="strong_match")],
                                  assessments={"Docker": assessment})
    assert reviews[0].question is not None
    assert build_proposals(reviews) == []


def test_model_cannot_claim_user_skipped_or_denied_without_source():
    for raw in [{"outcome": "skipped"}, {"outcome": "no_experience"},
                {"outcome": "no_experience", "evidence": "Docker"}]:
        assert validate_assessment(raw, "Docker", "skill").outcome == "insufficient"


@pytest.mark.parametrize("text", ["不知道", "不确定", "", "I don't know"])
def test_uncertainty_never_confirms_a_fact(text):
    assert explicit_outcome(UserFact(requirement="Docker", content=text)).outcome == "insufficient"


def test_mixed_negative_and_positive_account_requires_semantic_review():
    text = "没有生产部署经验，但是在个人项目里用过 Docker。"
    assert explicit_outcome(UserFact(requirement="Docker", content=text)) is None


def test_model_failure_or_malformed_result_fails_closed(monkeypatch):
    class InvalidClient:
        async def analyze(self, *args):
            return {"items": [{"key": "0", "outcome": "supported", "evidence": 12}]}
    monkeypatch.setattr("app.services.resume_agent.evidence.get_llm_client", lambda: InvalidClient())
    result = asyncio.run(assess_requirements(
        [RequirementAnalysis(requirement="Docker")], "只了解 Docker", "需要 Docker", []
    ))
    assert result["Docker"].outcome == "insufficient"


def test_multiple_model_records_for_one_requirement_cannot_confirm_it(monkeypatch):
    class DuplicateClient:
        async def analyze(self, *args):
            return {"items": [{"key": "0", "outcome": "no_experience"}] * 2}
    monkeypatch.setattr("app.services.resume_agent.evidence.get_llm_client", lambda: DuplicateClient())
    result = asyncio.run(assess_requirements(
        [RequirementAnalysis(requirement="Docker")], "只了解 Docker", "需要 Docker", []
    ))
    assert result["Docker"].outcome == "insufficient"


def test_only_active_jd_dimensions_contribute_to_total():
    result = summarize_matches([MatchDetail(
        requirement="Python", category="skill", level="required",
        matched=True, match_score=1, evidence="Python",
    )])
    assert result.total_score == 100
    assert result.score_breakdown.experience_total == 0


def test_missing_required_dimension_still_counts_as_zero():
    result = summarize_matches([
        MatchDetail(requirement="Python", category="skill", matched=True, match_score=1),
        MatchDetail(requirement="本科", category="education", matched=False, match_score=0),
    ])
    # Skill weight 4, education weight 2. An actual JD requirement cannot be omitted.
    assert result.total_score == 67


def test_no_jd_requirements_returns_zero():
    assert summarize_matches([]).total_score == 0


def test_partial_evidence_stays_partial_in_updated_report():
    from app.schemas.jobfit import JobFitAnalysis
    from app.services.resume_agent.analysis import refresh_analysis

    source = "我在个人项目中使用 Docker 部署测试环境。"
    original = JobFitAnalysis(match_score=0, summary="", requirement_analysis=[
        RequirementAnalysis(requirement="生产部署", category="project")
    ])
    updated = refresh_analysis(original, {"生产部署": EvidenceAssessment(
        outcome="supported", coverage="partial", evidence=source,
    )}, updated=True)
    assert updated.match_score == 50
    assert updated.requirement_analysis[0].status == "partial_match"
    assert len(updated.gaps) == len(updated.risk_details) == 1
