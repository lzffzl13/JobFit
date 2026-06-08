"""Unit tests for matcher.py — deterministic matching engine."""

import pytest

from app.schemas.jobfit import (
    EducationBlock,
    JDProfile,
    JDRequirement,
    ProjectBlock,
    ResumeProfile,
    SkillsBlock,
)
from app.services.matcher import (
    _calculate_breakdown,
    _embedding_match,
    _match_education,
    _match_experience,
    _match_skill,
    _match_soft,
    _normalize_text,
    _synonym_match,
    calculate_match,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_resume(
    hard=None, soft=None, exp=None, projects=None, degree="", major="", school=""
):
    return ResumeProfile(
        skills=SkillsBlock(hard=hard or [], soft=soft or []),
        experience_years=exp or {},
        projects=[
            ProjectBlock(name=p["name"], tech=p.get("tech", []), desc=p.get("desc", ""), highlights=p.get("highlights", []))
            for p in (projects or [])
        ],
        education=EducationBlock(degree=degree, major=major, school=school),
    )


def _make_jd(requirements):
    return JDProfile(
        requirements=[
            JDRequirement(
                name=r["name"],
                category=r.get("category", "skill"),
                level=r.get("level", "required"),
                description=r.get("description", ""),
                alternatives=r.get("alternatives", []),
            )
            for r in requirements
        ]
    )


# ---------------------------------------------------------------------------
# Synonym matching
# ---------------------------------------------------------------------------


class TestSynonymMatch:
    def test_exact_match(self):
        matched, evidence, method = _synonym_match("Python", ["Python", "Java"])
        assert matched is True
        assert evidence == "Python"
        assert method == "exact"

    def test_case_insensitive(self):
        matched, evidence, method = _synonym_match("python", ["Python"])
        assert matched is True

    def test_synonym_table_match(self):
        matched, evidence, method = _synonym_match("缓存", ["Redis"])
        assert matched is True
        assert evidence == "Redis"
        assert method == "synonym"

    def test_reverse_synonym(self):
        # Resume has "Redis", JD asks for "redis" via synonym table
        matched, evidence, method = _synonym_match("redis", ["缓存技术", "Redis"])
        assert matched is True

    def test_no_match(self):
        matched, _, _ = _synonym_match("Rust", ["Python", "Java"])
        assert matched is False

    def test_empty_resume(self):
        matched, _, _ = _synonym_match("Python", [])
        assert matched is False


class TestNormalizeText:
    def test_lowercase(self):
        assert _normalize_text("Python") == "python"

    def test_strip(self):
        assert _normalize_text("  Python  ") == "python"

    def test_extra_spaces(self):
        assert _normalize_text("Fast  API") == "fast api"


# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------


class TestMatchSkill:
    def test_direct_match(self):
        detail = _match_skill("Python", ["Python", "FastAPI"], [])
        assert detail.matched is True
        assert detail.match_score == 1.0
        assert detail.method == "exact"

    def test_synonym_match(self):
        detail = _match_skill("缓存", ["Redis", "MySQL"], [])
        assert detail.matched is True
        assert detail.method == "synonym"

    def test_alternative_match(self):
        # JD says "Redis or Memcached", resume has Memcached
        detail = _match_skill("Redis", ["Memcached", "MySQL"], ["Memcached"])
        assert detail.matched is True

    def test_no_match(self):
        detail = _match_skill("Rust", ["Python", "Java"], [])
        assert detail.matched is False

    def test_empty_skills(self):
        detail = _match_skill("Python", [], [])
        assert detail.matched is False


# ---------------------------------------------------------------------------
# Experience matching
# ---------------------------------------------------------------------------


class TestMatchExperience:
    def test_years_met(self):
        detail = _match_experience("后端开发", "3年以上后端经验", {"total": 5, "backend": 4})
        assert detail.matched is True
        assert detail.match_score == 1.0

    def test_years_partial(self):
        # "后端" matches the key "后端" in experience_years
        detail = _match_experience("后端", "3年以上后端经验", {"total": 1, "后端": 2})
        assert detail.matched is True
        assert detail.match_score == pytest.approx(2 / 3, rel=0.01)

    def test_years_not_met(self):
        detail = _match_experience("后端开发", "5年以上经验", {"total": 2})
        assert detail.matched is True  # has some experience
        assert detail.match_score == pytest.approx(0.4, rel=0.01)

    def test_no_experience(self):
        detail = _match_experience("后端开发", "3年经验", {})
        assert detail.matched is False


# ---------------------------------------------------------------------------
# Education matching
# ---------------------------------------------------------------------------


class TestMatchEducation:
    def test_degree_met(self):
        edu = EducationBlock(degree="本科", major="计算机科学", school="XX大学")
        detail = _match_education("计算机相关专业", "本科及以上", edu)
        assert detail.matched is True

    def test_degree_higher(self):
        edu = EducationBlock(degree="硕士", major="计算机", school="XX大学")
        detail = _match_education("学历要求", "本科", edu)
        assert detail.matched is True

    def test_degree_not_met(self):
        edu = EducationBlock(degree="大专", major="计算机", school="XX学校")
        detail = _match_education("学历要求", "本科及以上学历", edu)
        assert detail.match_score < 1.0

    def test_no_education(self):
        edu = EducationBlock()
        detail = _match_education("学历要求", "本科", edu)
        assert detail.matched is False


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestCalculateMatch:
    def test_perfect_match(self):
        resume = _make_resume(
            hard=["Python", "FastAPI", "Redis", "MySQL"],
            exp={"total": 5, "后端": 4},
            projects=[{"name": "用户系统", "tech": ["FastAPI", "Redis"], "desc": "高性能用户系统"}],
            degree="本科", major="计算机科学",
        )
        jd = _make_jd([
            {"name": "Python", "category": "skill", "level": "required"},
            {"name": "FastAPI", "category": "skill", "level": "required"},
            {"name": "Redis", "category": "skill", "level": "preferred"},
            {"name": "后端", "category": "experience", "level": "required", "description": "3年以上"},
        ])
        result = calculate_match(resume, jd)
        # All 4 requirements matched, score should be high
        assert result.total_score >= 50
        assert len(result.matched) == 4
        assert len(result.gaps) == 0

    def test_no_match(self):
        resume = _make_resume(hard=["Python"])
        jd = _make_jd([
            {"name": "Rust", "category": "skill", "level": "required"},
            {"name": "Go", "category": "skill", "level": "required"},
        ])
        result = calculate_match(resume, jd)
        assert result.total_score < 50
        assert len(result.gaps) >= 1

    def test_partial_match(self):
        resume = _make_resume(
            hard=["Python", "MySQL"],
            exp={"total": 2},
        )
        jd = _make_jd([
            {"name": "Python", "category": "skill", "level": "required"},
            {"name": "Redis", "category": "skill", "level": "required"},
            {"name": "后端经验", "category": "experience", "level": "required", "description": "3年"},
        ])
        result = calculate_match(resume, jd)
        assert 0 < result.total_score < 100
        assert len(result.matched) >= 1
        assert len(result.gaps) >= 1

    def test_empty_resume(self):
        resume = _make_resume()
        jd = _make_jd([{"name": "Python", "category": "skill", "level": "required"}])
        result = calculate_match(resume, jd)
        assert result.total_score == 0 or result.total_score < 30

    def test_empty_jd(self):
        resume = _make_resume(hard=["Python"])
        jd = _make_jd([])
        result = calculate_match(resume, jd)
        assert result.total_score == 0

    def test_score_breakdown_populated(self):
        resume = _make_resume(
            hard=["Python"],
            exp={"total": 3},
            degree="本科", major="计算机",
        )
        jd = _make_jd([
            {"name": "Python", "category": "skill", "level": "required"},
            {"name": "经验", "category": "experience", "level": "required", "description": "2年"},
            {"name": "学历", "category": "education", "level": "required", "description": "本科"},
        ])
        result = calculate_match(resume, jd)
        sb = result.score_breakdown
        assert sb.total_score >= 0
        assert sb.skill_score >= 0
        assert sb.experience_score >= 0
        assert sb.education_score >= 0

    def test_core_vs_bonus_requirements(self):
        resume = _make_resume(hard=["Python", "Docker"])
        jd = _make_jd([
            {"name": "Python", "category": "skill", "level": "required"},
            {"name": "Docker", "category": "skill", "level": "preferred"},
        ])
        result = calculate_match(resume, jd)
        assert "Python" in result.core_requirements
        assert "Docker" in result.bonus_requirements

    def test_risk_items_from_weak_required(self):
        resume = _make_resume(hard=[])
        jd = _make_jd([
            {"name": "Rust", "category": "skill", "level": "required"},
        ])
        result = calculate_match(resume, jd)
        assert len(result.risk_items) >= 1
        assert "Rust" in result.risk_items[0]

    def test_alternatives_in_matching(self):
        resume = _make_resume(hard=["Memcached"])
        jd = _make_jd([
            {"name": "Redis", "category": "skill", "level": "required", "alternatives": ["Memcached"]},
        ])
        result = calculate_match(resume, jd)
        assert len(result.matched) == 1
        assert result.matched[0].matched is True
