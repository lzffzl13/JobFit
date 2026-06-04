"""LLM prompt building and output parsing.

The LLM is responsible for:
1. Extracting requirements from JD (any position, no predefined strategies)
2. Evaluating evidence_ratio + confidence for each requirement
3. Generating summary, gaps, rewrites, interview questions

Code only does: defensive parsing, clamp, format repair.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert job-fit analyst. Your task is to analyze the match between a resume and a job description.

## Your Responsibilities

1. **Extract requirements from the JD** — identify core skills, qualifications, and bonus items. You can handle ANY position (backend, frontend, data, product, design, etc.) without predefined categories.

2. **Evaluate each requirement against the resume** — for each requirement, provide:
   - `evidence_ratio`: float 0.0~1.0 (continuous, NOT discrete)
     - 0.0 = no evidence at all
     - 0.1~0.3 = only mentioned in skills list, no project context
     - 0.3~0.6 = some learning/demo/coursework evidence
     - 0.6~0.85 = project experience but missing details
     - 0.85~1.0 = strong project evidence with technical details
   - `confidence`: how confident you are in this judgment (0.0~1.0)
   - `reasoning`: brief explanation
   - `evidence_quote`: exact text from resume (empty string if none)

3. **Assess bonus and extra competitiveness** — bonus items from JD, and extra strengths (quantified results, awards, open source, complex system design).

## Rules

- Never invent resume experience. Only cite evidence from the provided resume.
- Base all conclusions on retrieved evidence chunks.
- Return valid JSON only.
- Treat A/B/C, one of, 任一, 至少一种 as alternative options unless the JD clearly says all are required.
- If a candidate matches one option in an alternative group, do not create a gap for other options.
- Focus on evidence quality, not just keyword presence.
- The application will verify your scores, so be honest about uncertainty — low confidence is better than a wrong high-confidence answer.
"""


def build_user_prompt(resume_context: str, jd_context: str) -> str:
    """Build the user prompt for LLM analysis.

    The LLM extracts requirements from JD and evaluates each against the resume.
    No predefined job type strategies needed.
    """
    return f"""Analyze the job fit between this resume and job description.

Step 1: Read the JD and extract all requirements (core skills, qualifications, bonus items).
Step 2: For each requirement, evaluate how well the resume matches it.
Step 3: Provide overall scores (match_score, bonus_score, extra_score).

Return JSON with this exact structure:
{{
  "match_score": 75,
  "bonus_score": 10,
  "extra_score": 5,
  "summary": "总体分析摘要（中文）",
  "requirements": [
    {{
      "name": "需求名称",
      "priority": "core",
      "evidence_ratio": 0.85,
      "confidence": 0.9,
      "reasoning": "判断依据",
      "evidence_quote": "简历中的原文，没有则为空字符串"
    }}
  ],
  "gaps": [
    {{
      "requirement": "缺失的需求名称",
      "suggestion": "改进建议（中文）"
    }}
  ],
  "resume_rewrites": [
    {{
      "before": "当前简历表述",
      "after": "改进后的表述",
      "reason": "改进原因"
    }}
  ],
  "interview_questions": [
    {{
      "question": "面试问题",
      "focus": "考察重点",
      "difficulty": "easy/medium/hard"
    }}
  ]
}}

Important:
- match_score is your overall match score (0-100). This is YOUR judgment — be honest and precise.
- bonus_score is the score for JD bonus items (0-15).
- extra_score is the score for resume extra competitiveness like quantified results, awards, open source (0-15).
- match_score should equal approximately core_score + bonus_score + extra_score, where core_score is derived from per-requirement evidence_ratio.
- evidence_ratio is a CONTINUOUS float (0.0~1.0), NOT discrete values
- priority is "core" or "bonus"
- If a candidate matches one option in an alternative group, do not create gaps for other options
- Be specific about evidence — quote exact resume text when possible

Resume evidence (retrieved chunks):
{resume_context}

JD evidence (retrieved chunks):
{jd_context}
"""


def normalize_llm_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM output with defensive parsing."""
    return {
        "requirements": [
            _normalize_requirement(item) for item in _as_list(raw.get("requirements"))
        ],
        "match_score": _coerce_score(raw.get("match_score") or raw.get("score")),
        "bonus_score": _coerce_score(raw.get("bonus_score")),
        "extra_score": _coerce_score(raw.get("extra_score")),
        "summary": _coerce_text(raw.get("summary") or raw.get("conclusion") or raw.get("analysis")),
        "gaps": [_normalize_gap_item(item) for item in _as_list(raw.get("gaps"))],
        "resume_rewrites": [
            _normalize_rewrite_item(item) for item in _as_list(raw.get("resume_rewrites"))
        ],
        "interview_questions": [
            _normalize_question_item(item) for item in _as_list(raw.get("interview_questions"))
        ],
    }


# ---------------------------------------------------------------------------
# Coerce helpers — defensive parsing for unpredictable LLM JSON output
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _coerce_score(value: Any) -> int:
    try:
        return max(0, min(int(float(value)), 100))
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_ratio(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Normalizers — handle LLM field name variations
# ---------------------------------------------------------------------------

def _normalize_requirement(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return {
            "name": _coerce_text(
                item.get("name") or item.get("requirement") or item.get("skill"), "unknown"
            ),
            "priority": _coerce_text(item.get("priority"), "core"),
            "evidence_ratio": _clamp_ratio(item.get("evidence_ratio") or item.get("ratio")),
            "confidence": _clamp_ratio(item.get("confidence")),
            "reasoning": _coerce_text(item.get("reasoning") or item.get("reason"), ""),
            "evidence_quote": _coerce_text(
                item.get("evidence_quote") or item.get("evidence"), ""
            ),
        }
    if isinstance(item, str):
        return {
            "name": item,
            "priority": "core",
            "evidence_ratio": 0.0,
            "confidence": 0.0,
            "reasoning": "",
            "evidence_quote": "",
        }
    return {
        "name": _coerce_text(item, "unknown"),
        "priority": "core",
        "evidence_ratio": 0.0,
        "confidence": 0.0,
        "reasoning": "",
        "evidence_quote": "",
    }


def _normalize_gap_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"requirement": item, "suggestion": f"补充与 {item} 相关的经历或项目细节。"}
    if isinstance(item, dict):
        requirement = item.get("requirement") or item.get("name") or item.get("skill")
        suggestion = item.get("suggestion") or item.get("advice") or item.get("reason")
        return {
            "requirement": _coerce_text(requirement, "gap"),
            "suggestion": _coerce_text(suggestion, "补充相关经历或项目细节。"),
        }
    text = _coerce_text(item, "gap")
    return {"requirement": text, "suggestion": "补充相关经历或项目细节。"}


def _normalize_rewrite_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"before": "", "after": item, "reason": "根据岗位要求优化表达。"}
    if isinstance(item, dict):
        after = item.get("after") or item.get("rewrite") or item.get("suggestion")
        return {
            "before": _coerce_text(item.get("before")),
            "after": _coerce_text(after, "补充更贴合 JD 的项目表达。"),
            "reason": _coerce_text(item.get("reason") or item.get("why"), "根据岗位要求优化表达。"),
        }
    return {"before": "", "after": _coerce_text(item), "reason": "根据岗位要求优化表达。"}


def _normalize_question_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"question": item, "focus": "interview", "difficulty": "medium"}
    if isinstance(item, dict):
        question = item.get("question") or item.get("content") or item.get("title")
        return {
            "question": _coerce_text(question, "请介绍一个相关项目。"),
            "focus": _coerce_text(item.get("focus") or item.get("topic"), "interview"),
            "difficulty": _coerce_text(item.get("difficulty"), "medium"),
        }
    return {"question": _coerce_text(item, "请介绍一个相关项目。"), "focus": "interview", "difficulty": "medium"}
