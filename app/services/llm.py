"""LLM interaction — extraction + suggestion generation.

Three independent LLM calls:
  1. extract_resume(text) -> ResumeProfile
  2. extract_jd(text) -> JDProfile
  3. generate_suggestions(match_result, resume, jd) -> dict

Each call has 4-layer error handling:
  Layer 1: Strong-constraint prompt (strict JSON)
  Layer 2: Auto-retry with error feedback (max 2 retries)
  Layer 3: Field fallback (missing fields get defaults)
  Layer 4: Pydantic validation
"""

import json
import logging
import re
from typing import Any

from app.schemas.jobfit import (
    GapDetail,
    MatchDetail,
    MatchResult,
    ResumeProfile,
    JDProfile,
    JDRequirement,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt: Resume extraction
# ---------------------------------------------------------------------------

RESUME_SYSTEM_PROMPT = """你是简历信息提取专家。从简历文本中提取结构化信息。

## 输出要求
- 只输出严格JSON，禁止解释、禁止markdown、禁止多余字符
- 技能要拆到具体技术名（FastAPI不是"Web框架"）
- experience_years 用数字，单位年
- projects 提取所有有技术细节的项目

## JSON格式
```json
{
  "skills": {
    "hard": ["具体技术名1", "具体技术名2"],
    "soft": ["沟通能力", "团队协作"]
  },
  "experience_years": {
    "total": 3,
    "backend": 2,
    "frontend": 1
  },
  "projects": [
    {
      "name": "项目名称",
      "tech": ["技术1", "技术2"],
      "desc": "项目描述",
      "highlights": ["亮点1", "亮点2"]
    }
  ],
  "education": {
    "degree": "本科",
    "major": "计算机科学",
    "school": "XX大学"
  },
  "certifications": ["证书1"]
}
```

注意：
- skills.hard 必须是具体技术栈名称，不要写"后端开发"这种笼统的
- 如果简历没有某类信息，返回空列表/空对象，不要编造
- experience_years.total 是总工作年限，其他按技术方向分类
- highlights 提取量化成果（QPS、性能提升百分比、用户数等）"""

# ---------------------------------------------------------------------------
# Prompt: JD extraction
# ---------------------------------------------------------------------------

JD_SYSTEM_PROMPT = """你是职位描述分析专家。从JD中提取所有岗位要求。

## 输出要求
- 只输出严格JSON，禁止解释、禁止markdown、禁止多余字符
- 识别所有要求项，包括隐含的
- level 三级：required（必须）、preferred（优先）、nice-to-have（加分）
- alternatives: 可替代选项，如"Redis或Memcached"

## JSON格式
```json
{
  "requirements": [
    {
      "name": "Python开发",
      "category": "skill",
      "level": "required",
      "description": "熟悉Python，有Web框架经验",
      "alternatives": []
    },
    {
      "name": "微服务经验",
      "category": "experience",
      "level": "preferred",
      "description": "有分布式系统或微服务架构经验",
      "alternatives": []
    },
    {
      "name": "计算机相关专业",
      "category": "education",
      "level": "required",
      "description": "本科及以上学历",
      "alternatives": []
    }
  ]
}
```

注意：
- category 必须是: skill / experience / education / project / soft
- level 必须是: required / preferred / nice-to-have
- JD写"必须/必备"→required，写"优先/加分/bonus"→preferred，写"了解/熟悉最好"→nice-to-have
- 技能要求拆到具体技术名
- alternatives 记录可互换的选项"""

# ---------------------------------------------------------------------------
# Prompt: Suggestion generation
# ---------------------------------------------------------------------------

SUGGESTION_SYSTEM_PROMPT = """你是求职顾问。基于匹配分析结果，为求职者生成实用建议。

## 输出要求
- 只输出严格JSON，禁止解释、禁止markdown
- 建议要具体、可操作，不要泛泛而谈
- 简历改写要给出具体的前后对比
- 面试题要针对薄弱环节

## JSON格式
```json
{
  "summary": "总体分析摘要（中文，200字以内）",
  "resume_rewrites": [
    {
      "before": "当前简历表述",
      "after": "改进后的表述",
      "reason": "改进原因"
    }
  ],
  "interview_questions": [
    {
      "question": "面试问题",
      "focus": "考察重点",
      "difficulty": "easy/medium/hard"
    }
  ]
}
```"""

# ---------------------------------------------------------------------------
# JSON parsing with retry
# ---------------------------------------------------------------------------


def _parse_json_strict(text: str) -> dict[str, Any]:
    """Try strict JSON parse first, then regex fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first JSON object
    brace_match = re.search(r"\{.*\}", text, re.S)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from LLM output: {text[:200]}")


async def _llm_call_with_retry(
    client,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Call LLM with auto-retry on JSON parse failure.

    Layer 1: Strong-constraint prompt
    Layer 2: Auto-retry with error feedback
    """
    current_prompt = user_prompt

    for attempt in range(max_retries + 1):
        try:
            raw = await client.analyze(system_prompt, current_prompt)
            if isinstance(raw, dict):
                return raw
            return _parse_json_strict(str(raw))
        except Exception as e:
            if attempt < max_retries:
                logger.warning("LLM JSON parse failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
                current_prompt = (
                    f"你上次输出的不是合法JSON，解析失败: {e}\n\n"
                    f"请严格输出合法JSON，不要包含任何其他文字。\n\n"
                    f"原始请求:\n{user_prompt}"
                )
            else:
                logger.error("LLM extraction failed after %d retries: %s", max_retries + 1, e)
                raise

    raise RuntimeError("LLM extraction failed")


# ---------------------------------------------------------------------------
# Field fallback helpers
# ---------------------------------------------------------------------------


def _safe_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return {}


# ---------------------------------------------------------------------------
# Public API: Extraction
# ---------------------------------------------------------------------------


async def extract_resume(text: str, client) -> ResumeProfile:
    """Extract structured resume data from raw text.

    Layer 3: Field fallback (missing fields get defaults)
    Layer 4: Pydantic validation (via ResumeProfile constructor)
    """
    user_prompt = f"请提取以下简历的结构化信息：\n\n{text}"
    raw = await _llm_call_with_retry(client, RESUME_SYSTEM_PROMPT, user_prompt)

    # Layer 3: Field fallback
    skills_raw = _safe_dict(raw.get("skills"))
    skills = {
        "hard": [_safe_str(s) for s in _safe_list(skills_raw.get("hard")) if s],
        "soft": [_safe_str(s) for s in _safe_list(skills_raw.get("soft")) if s],
    }

    exp_raw = _safe_dict(raw.get("experience_years"))
    experience_years = {}
    for k, v in exp_raw.items():
        fv = _safe_float(v)
        if fv > 0:
            experience_years[k] = fv

    projects_raw = _safe_list(raw.get("projects"))
    projects = []
    for p in projects_raw:
        if not isinstance(p, dict):
            continue
        projects.append({
            "name": _safe_str(p.get("name")),
            "tech": [_safe_str(t) for t in _safe_list(p.get("tech")) if t],
            "desc": _safe_str(p.get("desc")),
            "highlights": [_safe_str(h) for h in _safe_list(p.get("highlights")) if h],
        })

    edu_raw = _safe_dict(raw.get("education"))
    education = {
        "degree": _safe_str(edu_raw.get("degree")),
        "major": _safe_str(edu_raw.get("major")),
        "school": _safe_str(edu_raw.get("school")),
    }

    certifications = [_safe_str(c) for c in _safe_list(raw.get("certifications")) if c]

    # Layer 4: Pydantic validation
    return ResumeProfile(
        skills=skills,
        experience_years=experience_years,
        projects=projects,
        education=education,
        certifications=certifications,
    )


async def extract_jd(text: str, client) -> JDProfile:
    """Extract structured JD requirements from raw text.

    Layer 3: Field fallback
    Layer 4: Pydantic validation
    """
    user_prompt = f"请提取以下职位描述的所有要求：\n\n{text}"
    raw = await _llm_call_with_retry(client, JD_SYSTEM_PROMPT, user_prompt)

    reqs_raw = _safe_list(raw.get("requirements"))
    requirements = []
    for r in reqs_raw:
        if not isinstance(r, dict):
            if isinstance(r, str) and r:
                requirements.append({"name": r, "category": "skill", "level": "required"})
            continue
        requirements.append({
            "name": _safe_str(r.get("name"), "unknown"),
            "category": _safe_str(r.get("category"), "skill"),
            "level": _safe_str(r.get("level"), "required"),
            "description": _safe_str(r.get("description")),
            "alternatives": [_safe_str(a) for a in _safe_list(r.get("alternatives")) if a],
        })

    # Validate category values
    valid_categories = {"skill", "experience", "education", "project", "soft"}
    valid_levels = {"required", "preferred", "nice-to-have"}
    for r in requirements:
        if r["category"] not in valid_categories:
            r["category"] = "skill"
        if r["level"] not in valid_levels:
            r["level"] = "required"

    return JDProfile(requirements=requirements)


async def generate_suggestions(
    match_result: MatchResult,
    resume: ResumeProfile,
    jd: JDProfile,
    client,
) -> dict[str, Any]:
    """Generate human-readable suggestions based on match result.

    This is the only step where LLM does "creative" work.
    """
    # Build concise context for the LLM
    matched_names = [d.requirement for d in match_result.matched]
    gap_names = [d.requirement for d in match_result.gaps]
    risk_text = "\n".join(f"- {r}" for r in match_result.risk_items) or "无"

    user_prompt = f"""匹配分析结果：

总分: {match_result.total_score}/100
技能分: {match_result.score_breakdown.skill_score}
经验分: {match_result.score_breakdown.experience_score}
项目分: {match_result.score_breakdown.project_score}
学历分: {match_result.score_breakdown.education_score}

已匹配: {', '.join(matched_names) or '无'}
未匹配: {', '.join(gap_names) or '无'}
风险项:
{risk_text}

请基于以上分析结果，生成改进建议和面试准备问题。"""

    try:
        raw = await _llm_call_with_retry(client, SUGGESTION_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        logger.error("Suggestion generation failed: %s", e)
        # Graceful degradation — return empty suggestions
        return {"summary": "", "resume_rewrites": [], "interview_questions": []}

    return {
        "summary": _safe_str(raw.get("summary")),
        "resume_rewrites": [
            {
                "before": _safe_str(r.get("before")),
                "after": _safe_str(r.get("after"), "补充更贴合JD的项目表达"),
                "reason": _safe_str(r.get("reason"), "根据岗位要求优化表达"),
            }
            for r in _safe_list(raw.get("resume_rewrites"))
            if isinstance(r, dict)
        ],
        "interview_questions": [
            {
                "question": _safe_str(q.get("question"), "请介绍一个相关项目"),
                "focus": _safe_str(q.get("focus"), "技术能力"),
                "difficulty": _safe_str(q.get("difficulty"), "medium"),
            }
            for q in _safe_list(raw.get("interview_questions"))
            if isinstance(q, dict)
        ],
    }
