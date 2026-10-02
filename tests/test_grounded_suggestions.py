"""The legacy suggestion endpoint receives source facts and cannot invent a rewrite."""

import asyncio
import json

from app.schemas.jobfit import JDProfile, JDRequirement, MatchResult, ProjectBlock, ResumeProfile
from app.services.llm import generate_suggestions


def test_legacy_prompt_contains_real_material_and_unsupported_claims_are_removed():
    source = "我在博客项目中使用 Python 实现登录接口，了解 Docker。"
    resume = ResumeProfile(projects=[ProjectBlock(name="博客项目", desc=source, tech=["Python"])])
    jd = JDProfile(requirements=[JDRequirement(name="Docker", description="容器部署经验")])

    class Client:
        async def analyze(self, system_prompt, user_prompt):
            context = json.JSONDecoder().raw_decode(user_prompt[user_prompt.index("{"):])[0]
            assert context["简历原文"] == source
            assert context["结构化简历"]["projects"][0]["name"] == "博客项目"
            assert context["岗位要求"]["requirements"][0]["description"] == "容器部署经验"
            return {"summary": "需要补充部署经历", "resume_rewrites": [
                {"before": source, "after": "主导生产部署，性能提升 50%", "evidence_basis": source},
                {"before": "负责生产环境部署", "after": "主导平台建设", "evidence_basis": "不存在的经历"},
            ], "interview_questions": []}

    result = asyncio.run(generate_suggestions(MatchResult(), resume, jd, Client(), resume_text=source))
    assert len(result["resume_rewrites"]) == 1
    rewrite = result["resume_rewrites"][0]
    assert rewrite["after"] == "在博客项目中使用 Python 实现登录接口，了解 Docker。"
    assert rewrite["evidence_basis"] == source
    assert "50%" not in rewrite["after"]
    assert "主导" not in rewrite["after"]


def test_untraceable_before_or_basis_is_rejected():
    class Client:
        async def analyze(self, *args):
            return {"resume_rewrites": [
                {"before": "Python", "evidence_basis": "生产部署经历"},
                {"before": "Java", "evidence_basis": "Python"},
                {"before": "Python", "evidence_basis": "Redis"},
                {"before": "", "after": "新经历"},
            ]}
    result = asyncio.run(generate_suggestions(
        MatchResult(), ResumeProfile(), JDProfile(), Client(), resume_text="只了解 Python。学习 Redis。"
    ))
    assert result["resume_rewrites"] == []


def test_legacy_model_failure_still_returns_valid_empty_suggestions():
    class Client:
        async def analyze(self, *args):
            raise RuntimeError("offline")
    result = asyncio.run(generate_suggestions(MatchResult(), ResumeProfile(), JDProfile(), Client()))
    assert result == {"summary": "", "resume_rewrites": [], "interview_questions": []}
