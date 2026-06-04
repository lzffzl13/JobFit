"""Lightweight validator — clamp + confidence warning.

The LLM does all the heavy lifting. This module only ensures output is sane:
- Clamp scores to valid ranges
- Flag low-confidence requirements (warning only, no rule engine override)
- Build score_breakdown from LLM's data (counts only, no recomputation)
"""

import logging

from app.schemas.jobfit import ScoreBreakdown

logger = logging.getLogger(__name__)

CONFIDENCE_LOW_THRESHOLD = 0.3


def validate_and_score(
    llm_requirements: list[dict],
    resume_text: str,
    llm_match_score: int,
    llm_bonus_score: int = 0,
    llm_extra_score: int = 0,
) -> dict:
    """Validate LLM output and build final result.

    Uses LLM's scores directly (clamped). Does NOT recompute scores
    from evidence_ratio — that's LLM's job.
    """
    warnings = []

    # Validate each requirement — flag low confidence
    validated_reqs = []
    for req in llm_requirements:
        ratio = _clamp_ratio(req.get("evidence_ratio", 0))
        confidence = _clamp_ratio(req.get("confidence", 0))

        if confidence < CONFIDENCE_LOW_THRESHOLD:
            warnings.append(
                f"{req.get('name')}: 低置信度(confidence={confidence:.2f}), 结果可能不准确"
            )

        validated_reqs.append({
            **req,
            "evidence_ratio": ratio,
            "confidence": confidence,
        })

    # Separate core and bonus
    core_reqs = [r for r in validated_reqs if r.get("priority") == "core"]
    bonus_reqs = [r for r in validated_reqs if r.get("priority") == "bonus"]

    # Clamp LLM scores
    match_score = max(0, min(100, llm_match_score))
    bonus_score = max(0, min(15, llm_bonus_score))
    extra_score = max(0, min(15, llm_extra_score))

    # Counts only — no recomputation
    core_matched = sum(1 for r in core_reqs if r["evidence_ratio"] > 0)
    core_total = len(core_reqs)
    bonus_matched = sum(1 for r in bonus_reqs if r["evidence_ratio"] > 0)
    bonus_total = len(bonus_reqs)

    score_breakdown = ScoreBreakdown(
        core_matched=core_matched,
        core_total=core_total,
        bonus_matched=bonus_matched,
        bonus_total=bonus_total,
        match_score=match_score,
        bonus_score=bonus_score,
        extra_score=extra_score,
        core_detail={r["name"]: _format_detail(r) for r in core_reqs},
        evidence_notes=warnings,
    )

    return {
        "score": match_score,
        "score_breakdown": score_breakdown,
        "validated_requirements": validated_reqs,
        "core_requirements": [r["name"] for r in core_reqs],
        "bonus_requirements": [r["name"] for r in bonus_reqs],
        "matched": [r["name"] for r in validated_reqs if r["evidence_ratio"] > 0],
        "missing": [r["name"] for r in validated_reqs if r["evidence_ratio"] == 0],
        "risk_items": [
            f"{r['name']}: ratio={r['evidence_ratio']:.2f}"
            for r in core_reqs if r["evidence_ratio"] < 0.85
        ][:6],
        "warnings": warnings,
    }


def _clamp_ratio(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _format_detail(req: dict) -> str:
    ratio = req["evidence_ratio"]
    if ratio == 0:
        return "0% (无证据)"
    if ratio <= 0.3:
        return f"{ratio*100:.0f}% (低证据)"
    if ratio <= 0.6:
        return f"{ratio*100:.0f}% (学习/demo)"
    if ratio <= 0.85:
        return f"{ratio*100:.0f}% (项目经验)"
    return f"{ratio*100:.0f}% (充分证据)"
