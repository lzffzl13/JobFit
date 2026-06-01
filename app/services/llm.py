import json
import re
from typing import Any

import httpx

from app.core.config import settings
from app.schemas.jobfit import JobFitAnalysis
from app.services import scoring as deterministic_scoring

SkillRequirement = deterministic_scoring.SkillRequirement
EvaluationResult = deterministic_scoring.EvaluationResult

SYSTEM_PROMPT = """You are a RAG agent for internship job-fit analysis.
Rules:
1. Never invent resume experience.
2. Base conclusions on retrieved evidence.
3. Return valid JSON only.
4. Treat A/B/C, one of, 任一, 至少一种 as alternative options unless the JD clearly says all are required.
5. Distinguish core requirements from bonus items. Missing bonus items should not heavily reduce the score.
6. If a candidate already matches one option in an alternative group, do not create a gap for the other options.
7. The final score is calculated by deterministic local rules. Use match_score only as an advisory field.
"""


def build_user_prompt(resume_context: str, jd_context: str, jd_semantics: str) -> str:
    return f"""Analyze the job fit and return JSON with these fields:
- match_score
- summary
- jd_requirements
- matched_strengths
- gaps
- resume_rewrites
- interview_questions

Important: do not treat missing alternative options as gaps when one option is already matched.
The application will overwrite match_score with local deterministic scoring, so focus on evidence and useful explanations.

JD semantics:
{jd_semantics}

Resume evidence:
{resume_context}

JD evidence:
{jd_context}
"""


class DeepSeekClient:
    async def analyze(
        self,
        resume_context: str,
        jd_context: str,
        jd_semantics: str,
    ) -> dict[str, Any]:
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

        payload = {
            "model": settings.deepseek_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(resume_context, jd_context, jd_semantics),
                },
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {settings.deepseek_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(
                f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return _parse_json_content(data["choices"][0]["message"]["content"])


def _parse_json_content(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_llm_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_score": _coerce_score(raw.get("match_score") or raw.get("score")),
        "summary": _coerce_text(raw.get("summary") or raw.get("conclusion") or raw.get("analysis")),
        "jd_requirements": [
            _normalize_requirement(item) for item in _as_list(raw.get("jd_requirements"))
        ],
        "matched_strengths": [
            _normalize_match_item(item) for item in _as_list(raw.get("matched_strengths"))
        ],
        "gaps": [_normalize_gap_item(item) for item in _as_list(raw.get("gaps"))],
        "resume_rewrites": [
            _normalize_rewrite_item(item) for item in _as_list(raw.get("resume_rewrites"))
        ],
        "interview_questions": [
            _normalize_question_item(item) for item in _as_list(raw.get("interview_questions"))
        ],
    }


def summarize_jd_semantics(jd_text: str) -> str:
    return deterministic_scoring.summarize_jd_semantics(jd_text)


def local_fallback_analysis(resume_text: str, jd_text: str) -> JobFitAnalysis:
    return deterministic_scoring.local_fallback_analysis(resume_text, jd_text)


def extract_skill_requirements(jd_text: str) -> list[SkillRequirement]:
    return deterministic_scoring.extract_skill_requirements(jd_text)


def evaluate_resume_against_requirements(
    requirements: list[SkillRequirement],
    resume_text: str,
) -> EvaluationResult:
    return deterministic_scoring.evaluate_resume_against_requirements(requirements, resume_text)


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


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_requirement(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "name": item,
            "category": "general",
            "evidence": item,
            "type": "single",
            "options": [],
            "required_count": 1,
            "priority": "core",
            "weight": 1.0,
        }
    if isinstance(item, dict):
        name = item.get("name") or item.get("requirement") or item.get("skill") or item.get("title")
        return {
            "name": _coerce_text(name, "unknown"),
            "category": _coerce_text(item.get("category") or item.get("type"), "general"),
            "evidence": _coerce_text(item.get("evidence") or item.get("reason") or name),
            "type": _coerce_text(item.get("type") or item.get("requirement_type"), "single"),
            "options": [str(option) for option in _as_list(item.get("options"))],
            "required_count": _coerce_int(item.get("required_count"), 1),
            "priority": _coerce_text(item.get("priority"), "core"),
            "weight": _coerce_float(item.get("weight"), 1.0),
        }
    return {
        "name": _coerce_text(item, "unknown"),
        "category": "general",
        "evidence": None,
        "type": "single",
        "options": [],
        "required_count": 1,
        "priority": "core",
        "weight": 1.0,
    }


def _normalize_match_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"requirement": item, "resume_evidence": item, "score": 80}
    if isinstance(item, dict):
        requirement = item.get("requirement") or item.get("name") or item.get("skill") or item.get("title")
        evidence = (
            item.get("resume_evidence")
            or item.get("evidence")
            or item.get("reason")
            or item.get("matched_evidence")
            or requirement
        )
        return {
            "requirement": _coerce_text(requirement, "matched requirement"),
            "resume_evidence": _coerce_text(evidence, "matched in resume"),
            "score": _coerce_score(item.get("score") or item.get("match_score") or 80),
        }
    text = _coerce_text(item, "matched requirement")
    return {"requirement": text, "resume_evidence": text, "score": 80}


def _normalize_gap_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"requirement": item, "suggestion": f"补充与 {item} 相关的经历或项目细节。"}
    if isinstance(item, dict):
        requirement = item.get("requirement") or item.get("name") or item.get("skill") or item.get("title")
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
        after = item.get("after") or item.get("rewrite") or item.get("suggestion") or item.get("content")
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
