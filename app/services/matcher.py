"""Deterministic matching engine — no LLM, pure program logic.

Matching strategy:
  1st pass: synonym lookup (high precision, fast)
  2nd pass: embedding similarity (semantic fallback, threshold 0.8)

Scoring:
  Program defines weights by requirement level.
  LLM only labels level (required/preferred/nice-to-have).
"""

import logging
import re
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from app.core.config import settings
from app.schemas.jobfit import (
    AnalysisOverview,
    GapDetail,
    MatchDetail,
    MatchResult,
    RequirementAnalysis,
    ResumeProfile,
    RiskItemDetail,
    ScoreBreakdown,
    JDProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_SIMILARITY_THRESHOLD = 0.8
STRONG_MATCH_THRESHOLD = 0.85

# Weights by requirement level
LEVEL_WEIGHTS: dict[str, int] = {
    "required": 5,
    "preferred": 2,
    "nice-to-have": 1,
}

# Category weights for total score calculation
CATEGORY_WEIGHTS: dict[str, int] = {
    "skill": 4,
    "experience": 3,
    "project": 3,
    "education": 2,
    "soft": 1,
}

# ---------------------------------------------------------------------------
# Synonym table — high-frequency tech mappings (~80 entries)
# Manual maintenance, covers ~80% of common cases.
# Fallback to embedding for anything not here.
# ---------------------------------------------------------------------------

SYNONYM_MAP: dict[str, list[str]] = {
    # Python ecosystem
    "python": ["python", "python3"],
    "python生态": ["python", "fastapi", "flask", "django", "sqlalchemy", "celery", "pydantic"],
    "fastapi": ["fastapi", "fast api"],
    "flask": ["flask"],
    "django": ["django"],
    "sqlalchemy": ["sqlalchemy", "sql alchemy"],
    "celery": ["celery"],
    # JS ecosystem
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "node": ["node", "nodejs", "node.js", "node js"],
    "react": ["react", "reactjs", "react.js"],
    "vue": ["vue", "vuejs", "vue.js"],
    "angular": ["angular", "angularjs"],
    "next.js": ["next.js", "nextjs", "next"],
    "nuxt": ["nuxt", "nuxtjs", "nuxt.js"],
    # Java ecosystem
    "java": ["java"],
    "spring": ["spring", "spring boot", "springboot", "spring mvc"],
    "spring boot": ["spring boot", "springboot", "spring-boot"],
    "mybatis": ["mybatis", "mybatis-plus", "mybatis plus"],
    # Go
    "go": ["go", "golang"],
    "gin": ["gin"],
    # Rust
    "rust": ["rust"],
    # Database
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres", "pg"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "memcached": ["memcached", "memcache"],
    "elasticsearch": ["elasticsearch", "es", "elastic search"],
    # Cache
    "缓存": ["redis", "memcached", "cache"],
    # Queue / MQ
    "消息队列": ["kafka", "rabbitmq", "rocketmq", "mq", "消息中间件"],
    "kafka": ["kafka"],
    "rabbitmq": ["rabbitmq", "rabbit mq"],
    # Container / Cloud
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "容器": ["docker", "kubernetes", "k8s", "container"],
    "aws": ["aws", "amazon web services"],
    "阿里云": ["阿里云", "aliyun", "alicloud"],
    # Frontend
    "前端框架": ["react", "vue", "angular", "svelte"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "scss", "sass", "less"],
    "tailwind": ["tailwind", "tailwindcss", "tailwind css"],
    # API
    "rest api": ["rest", "restful", "rest api", "restful api"],
    "graphql": ["graphql", "graph ql"],
    "grpc": ["grpc", "g rpc"],
    # DevOps
    "ci/cd": ["ci/cd", "cicd", "ci cd", "jenkins", "github actions", "gitlab ci"],
    "jenkins": ["jenkins"],
    "nginx": ["nginx"],
    # Testing
    "单元测试": ["pytest", "unittest", "jest", "junit", "单元测试", "unit test"],
    # Version control
    "git": ["git", "github", "gitlab"],
    # Architecture
    "微服务": ["微服务", "microservice", "micro-service", "微服务架构"],
    "分布式": ["分布式", "distributed"],
    # AI / ML
    "机器学习": ["机器学习", "machine learning", "ml"],
    "深度学习": ["深度学习", "deep learning", "dl"],
    "pytorch": ["pytorch", "py torch"],
    "tensorflow": ["tensorflow", "tf"],
    "llm": ["llm", "大模型", "大语言模型", "大语言模型"],
    "rag": ["rag", "retrieval augmented generation"],
    # OS
    "linux": ["linux"],
    # Education
    "计算机": ["计算机", "computer science", "cs", "软件工程", "信息技术"],
    "本科": ["本科", "bachelor", "学士"],
    "硕士": ["硕士", "master", "研究生"],
    "博士": ["博士", "phd", "doctorate"],
}


# ---------------------------------------------------------------------------
# Embedding model (lazy load, cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def _embedding_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using BGE embeddings."""
    model = _get_embedding_model()
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    score = cos_sim(embeddings[0], embeddings[1]).item()
    return max(0.0, min(1.0, score))


def _batch_embedding_similarity(query: str, candidates: list[str]) -> list[float]:
    """Compute similarity between one query and multiple candidates."""
    if not candidates:
        return []
    model = _get_embedding_model()
    all_texts = [query, *candidates]
    embeddings = model.encode(all_texts, normalize_embeddings=True)
    query_emb = embeddings[0]
    candidate_embs = embeddings[1:]
    scores = cos_sim(query_emb, candidate_embs).tolist()[0]
    return [max(0.0, min(1.0, s)) for s in scores]


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove extra spaces."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _synonym_match(requirement_name: str, resume_items: list[str]) -> tuple[bool, str, str]:
    """First pass: check synonym table for exact mapping.

    Returns (matched, matched_item, method).
    """
    req_lower = _normalize_text(requirement_name)
    resume_lower = [_normalize_text(s) for s in resume_items]

    # Direct match
    for i, item in enumerate(resume_lower):
        if req_lower == item or req_lower in item or item in req_lower:
            return True, resume_items[i], "exact"

    # Synonym match
    if req_lower in SYNONYM_MAP:
        synonyms = [s.lower() for s in SYNONYM_MAP[req_lower]]
        for i, item in enumerate(resume_lower):
            if item in synonyms:
                return True, resume_items[i], "synonym"

    # Reverse: check if any resume item is a synonym of the requirement
    for key, synonyms in SYNONYM_MAP.items():
        synonyms_lower = [s.lower() for s in synonyms]
        if req_lower in synonyms_lower:
            for i, item in enumerate(resume_lower):
                if item == key:
                    return True, resume_items[i], "synonym"

    return False, "", ""


def _embedding_match(
    requirement_name: str, resume_items: list[str], threshold: float = EMBEDDING_SIMILARITY_THRESHOLD
) -> tuple[bool, str, float, str]:
    """Second pass: embedding similarity fallback.

    Returns (matched, best_item, score, method).
    """
    if not resume_items:
        return False, "", 0.0, ""

    scores = _batch_embedding_similarity(requirement_name, resume_items)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]

    if best_score >= threshold:
        return True, resume_items[best_idx], best_score, "embedding"

    return False, resume_items[best_idx], best_score, ""


def _match_skill(
    req_name: str, resume_skills: list[str], alternatives: list[str]
) -> MatchDetail:
    """Match a single skill requirement against resume skills."""
    search_names = [req_name] + alternatives if alternatives else [req_name]

    best_match = MatchDetail(requirement=req_name, category="skill")

    for name in search_names:
        # Pass 1: synonym
        matched, evidence, method = _synonym_match(name, resume_skills)
        if matched:
            return MatchDetail(
                requirement=req_name,
                category="skill",
                matched=True,
                match_score=1.0,
                evidence=evidence,
                method=method,
            )

        # Pass 2: embedding
        emb_matched, emb_evidence, emb_score, emb_method = _embedding_match(name, resume_skills)
        if emb_matched and emb_score > best_match.match_score:
            best_match = MatchDetail(
                requirement=req_name,
                category="skill",
                matched=True,
                match_score=emb_score,
                evidence=emb_evidence,
                method=emb_method,
            )

    return best_match


def _match_experience(req_name: str, req_desc: str, experience_years: dict[str, float]) -> MatchDetail:
    """Match experience requirement against resume experience years."""
    # Try to extract years from requirement description
    years_match = re.search(r"(\d+)\s*[年+]", req_desc)
    required_years = float(years_match.group(1)) if years_match else 0

    # Check total years
    total = experience_years.get("total", 0)

    # Check category-specific years
    req_lower = _normalize_text(req_name)
    category_years = 0
    for key, val in experience_years.items():
        if key == "total":
            continue
        if _normalize_text(key) in req_lower or req_lower in _normalize_text(key):
            category_years = max(category_years, val)

    best_years = max(total, category_years)

    if required_years <= 0:
        # No specific years required, just check if has experience
        if best_years > 0:
            return MatchDetail(
                requirement=req_name, category="experience",
                matched=True, match_score=min(1.0, best_years / 3),
                evidence=f"{best_years}年", method="rule",
            )
        return MatchDetail(requirement=req_name, category="experience")

    ratio = min(1.0, best_years / required_years)
    return MatchDetail(
        requirement=req_name, category="experience",
        matched=ratio > 0,
        match_score=ratio,
        evidence=f"{best_years}年 (要求{required_years}年)",
        method="rule",
    )


def _match_project(req_name: str, req_desc: str, projects: list) -> MatchDetail:
    """Match project requirement using embedding similarity on project descriptions."""
    if not projects:
        return MatchDetail(requirement=req_name, category="project")

    # Combine project info into searchable text
    project_texts = []
    for p in projects:
        parts = [p.name, p.desc, " ".join(p.tech), " ".join(p.highlights)]
        project_texts.append(" ".join(filter(None, parts)))

    query = f"{req_name} {req_desc}".strip()
    emb_matched, best_text, score, method = _embedding_match(query, project_texts)

    if emb_matched:
        return MatchDetail(
            requirement=req_name, category="project",
            matched=True, match_score=score,
            evidence=best_text[:200], method=method,
        )

    return MatchDetail(
        requirement=req_name, category="project",
        match_score=score, evidence=best_text[:200] if best_text else "",
    )


def _match_education(req_name: str, req_desc: str, education) -> MatchDetail:
    """Match education requirement using rule-based logic."""
    edu_text = f"{education.degree} {education.major} {education.school}".strip()
    if not edu_text:
        return MatchDetail(requirement=req_name, category="education")

    req_lower = _normalize_text(f"{req_name} {req_desc}")
    edu_lower = _normalize_text(edu_text)

    # Degree matching
    degree_order = {"博士": 4, "硕士": 3, "本科": 2, "大专": 1}
    required_degree = 0
    user_degree = 0
    for deg, level in degree_order.items():
        if deg in req_lower:
            required_degree = max(required_degree, level)
        if deg in edu_lower:
            user_degree = max(user_degree, level)

    if required_degree > 0:
        ratio = min(1.0, user_degree / required_degree) if user_degree > 0 else 0
        return MatchDetail(
            requirement=req_name, category="education",
            matched=ratio >= 1.0, match_score=ratio,
            evidence=edu_text, method="rule",
        )

    # Major relevance
    matched, evidence, method = _synonym_match(req_name, [edu_text])
    if matched:
        return MatchDetail(
            requirement=req_name, category="education",
            matched=True, match_score=0.9, evidence=evidence, method=method,
        )

    # Fallback: embedding
    score = _embedding_similarity(req_lower, edu_lower)
    return MatchDetail(
        requirement=req_name, category="education",
        matched=score >= 0.7, match_score=score,
        evidence=edu_text, method="embedding" if score >= 0.7 else "",
    )


def _match_soft(req_name: str, req_desc: str, resume_skills: list[str]) -> MatchDetail:
    """Match soft skill — mostly embedding based, lower threshold."""
    soft_skills = resume_skills
    if not soft_skills:
        return MatchDetail(requirement=req_name, category="soft")

    query = f"{req_name} {req_desc}".strip()
    emb_matched, evidence, score, method = _embedding_match(query, soft_skills, threshold=0.7)

    return MatchDetail(
        requirement=req_name, category="soft",
        matched=emb_matched, match_score=score,
        evidence=evidence, method=method,
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def calculate_match(resume: ResumeProfile, jd: JDProfile) -> MatchResult:
    """Calculate match between resume and JD profiles.

    Pure deterministic logic — no LLM involved.
    """
    all_resume_skills = resume.skills.hard + resume.skills.soft
    project_texts = [
        f"{p.name} {' '.join(p.tech)} {p.desc} {' '.join(p.highlights)}"
        for p in resume.projects
    ]

    match_details: list[MatchDetail] = []

    for req in jd.requirements:
        level = req.level  # required / preferred / nice-to-have
        category = req.category

        if category == "skill":
            detail = _match_skill(req.name, all_resume_skills, req.alternatives)
        elif category == "experience":
            detail = _match_experience(req.name, req.description, resume.experience_years)
        elif category == "project":
            detail = _match_project(req.name, req.description, resume.projects)
        elif category == "education":
            detail = _match_education(req.name, req.description, resume.education)
        elif category == "soft":
            detail = _match_soft(req.name, req.description, resume.skills.soft)
        else:
            # Unknown category → try skill matching as fallback
            detail = _match_skill(req.name, all_resume_skills, req.alternatives)

        detail.level = level
        match_details.append(detail)

    # Build score breakdown
    breakdown = _calculate_breakdown(match_details)

    # Build gaps (unmatched or low-score required items)
    gaps = [
        GapDetail(
            requirement=d.requirement,
            category=d.category,
            level=d.level,
            current_score=d.match_score,
            suggestion=_gap_suggestion(d),
        )
        for d in match_details
        if _is_gap_detail(d) and d.level in ("required", "preferred")
    ]

    requirement_analyses = [_build_requirement_analysis(d) for d in match_details]

    # Risk items: required skills with low match
    risk_details = [
        _build_risk_detail(d)
        for d in match_details
        if d.level == "required" and d.match_score < STRONG_MATCH_THRESHOLD
    ][:6]
    risk_items = [
        f"{item.requirement}: {item.score}%"
        for item in risk_details
    ]

    return MatchResult(
        total_score=breakdown.total_score,
        score_breakdown=breakdown,
        matched=[d for d in match_details if d.matched],
        gaps=gaps,
        requirement_analyses=requirement_analyses,
        analysis_overview=_build_analysis_overview(requirement_analyses, risk_details),
        core_requirements=[d.requirement for d in match_details if d.level == "required"],
        bonus_requirements=[d.requirement for d in match_details if d.level != "required"],
        risk_items=risk_items,
        risk_details=risk_details,
    )


def _calculate_breakdown(details: list[MatchDetail]) -> ScoreBreakdown:
    """Calculate weighted score breakdown by category."""
    category_scores: dict[str, list[tuple[float, int]]] = {
        "skill": [], "experience": [], "project": [], "education": [], "soft": [],
    }

    for d in details:
        cat = d.category if d.category in category_scores else "skill"
        weight = LEVEL_WEIGHTS.get(d.level, 1)
        category_scores[cat].append((d.match_score, weight))

    def _weighted_avg(items: list[tuple[float, int]]) -> tuple[int, int]:
        if not items:
            return 0, 0
        total_weight = sum(w for _, w in items)
        weighted_sum = sum(s * w for s, w in items)
        return round(weighted_sum / total_weight * 100) if total_weight > 0 else 0, total_weight

    skill_score, skill_total = _weighted_avg(category_scores["skill"])
    exp_score, exp_total = _weighted_avg(category_scores["experience"])
    proj_score, proj_total = _weighted_avg(category_scores["project"])
    edu_score, edu_total = _weighted_avg(category_scores["education"])
    soft_score, _ = _weighted_avg(category_scores["soft"])

    # Total: weighted by category importance
    cat_weighted_sum = (
        skill_score * CATEGORY_WEIGHTS["skill"]
        + exp_score * CATEGORY_WEIGHTS["experience"]
        + proj_score * CATEGORY_WEIGHTS["project"]
        + edu_score * CATEGORY_WEIGHTS["education"]
        + soft_score * CATEGORY_WEIGHTS["soft"]
    )
    cat_total_weight = sum(CATEGORY_WEIGHTS.values())
    total = round(cat_weighted_sum / cat_total_weight) if cat_total_weight > 0 else 0

    return ScoreBreakdown(
        skill_score=skill_score,
        experience_score=exp_score,
        project_score=proj_score,
        education_score=edu_score,
        total_score=min(100, max(0, total)),
        skill_total=skill_total,
        experience_total=exp_total,
        project_total=proj_total,
        education_total=edu_total,
    )


def _gap_suggestion(detail: MatchDetail) -> str:
    """Generate a hint for an unmatched requirement."""
    if detail.category == "skill":
        if detail.evidence:
            return f"补充 {detail.requirement} 的直接使用场景，并把相关项目结果写清楚"
        return f"补充 {detail.requirement} 相关的项目经历或学习经验"
    if detail.category == "experience":
        if detail.evidence:
            return f"突出与 {detail.requirement} 对应的年限、职责范围和业务场景"
        return f"补充 {detail.requirement} 相关的工作经历"
    if detail.category == "project":
        if detail.evidence:
            return f"强化项目中与 {detail.requirement} 直接相关的职责、技术和结果"
        return f"补充与 {detail.requirement} 相关的项目描述"
    if detail.category == "education":
        if detail.evidence:
            return "明确学历层次、专业名称和学校信息"
        return f"学历要求: {detail.requirement}"
    if detail.category == "soft":
        return f"补充能体现 {detail.requirement} 的具体案例或协作结果"
    return f"补充 {detail.requirement} 相关经历"


def _classify_status(detail: MatchDetail) -> str:
    """Classify a requirement match into strong/partial/gap."""
    if detail.match_score >= STRONG_MATCH_THRESHOLD and detail.matched:
        return "strong_match"
    if detail.match_score > 0 or detail.matched:
        return "partial_match"
    return "gap"


def _is_gap_detail(detail: MatchDetail) -> bool:
    """Whether a requirement should appear in gap output."""
    return _classify_status(detail) != "strong_match"


def _build_requirement_analysis(detail: MatchDetail) -> RequirementAnalysis:
    """Convert a raw match detail into a richer requirement analysis item."""
    return RequirementAnalysis(
        requirement=detail.requirement,
        category=detail.category,
        level=detail.level,
        matched=detail.matched,
        score=int(round(detail.match_score * 100)),
        status=_classify_status(detail),
        evidence=detail.evidence,
        method=detail.method,
        explanation=_build_explanation(detail),
        suggestion=_gap_suggestion(detail) if _classify_status(detail) != "strong_match" else "",
    )


def _build_explanation(detail: MatchDetail) -> str:
    """Generate a human-readable explanation for a requirement match."""
    status = _classify_status(detail)

    if detail.category == "experience":
        if status == "strong_match":
            return f"相关经验满足要求，当前证据为 {detail.evidence or '已识别到相关经验'}。"
        if status == "partial_match":
            return f"已有部分相关经验，但覆盖度不足，当前证据为 {detail.evidence or '经验信息有限'}。"
        return f"未找到与 {detail.requirement} 对应的明确经验年限或场景。"

    if detail.category == "education":
        if status == "strong_match":
            return f"学历或专业要求已覆盖，匹配证据为 {detail.evidence or '教育背景已命中'}。"
        if status == "partial_match":
            return f"教育背景与要求有一定相关性，但仍存在差距，当前证据为 {detail.evidence or '教育信息有限'}。"
        return f"当前教育信息不足以支撑 {detail.requirement} 要求。"

    if detail.category == "project":
        if status == "strong_match":
            return f"项目经历中存在直接相关内容，证据为 {detail.evidence or '相关项目已命中'}。"
        if status == "partial_match":
            return f"项目经历与该要求有一定关联，但缺少更直接的证明，当前证据为 {detail.evidence or '项目关联度有限'}。"
        return f"项目经历中缺少与 {detail.requirement} 直接相关的内容。"

    if detail.category == "soft":
        if status == "strong_match":
            return f"软技能要求已有明确佐证，证据为 {detail.evidence or '软技能已命中'}。"
        if status == "partial_match":
            return f"检测到一定相关软技能，但表达还不够具体，当前证据为 {detail.evidence or '软技能证据有限'}。"
        return f"缺少能体现 {detail.requirement} 的明确案例。"

    if status == "strong_match":
        return f"已找到与 {detail.requirement} 直接相关的证据，匹配方式为 {detail.method or '规则匹配'}。"
    if status == "partial_match":
        if detail.evidence:
            return f"与 {detail.requirement} 有一定相关性，但证据强度不足，当前命中 {detail.evidence}。"
        return f"与 {detail.requirement} 存在部分相关性，但缺少直接证据。"
    return f"当前简历中未找到 {detail.requirement} 的直接证据。"


def _build_risk_detail(detail: MatchDetail) -> RiskItemDetail:
    """Build structured risk detail for weak required requirements."""
    score = int(round(detail.match_score * 100))
    if score < 40:
        severity = "high"
    elif score < 70:
        severity = "medium"
    else:
        severity = "low"

    return RiskItemDetail(
        requirement=detail.requirement,
        category=detail.category,
        level=detail.level,
        score=score,
        severity=severity,
        reason=_build_explanation(detail),
    )


def _build_analysis_overview(
    requirement_analyses: list[RequirementAnalysis], risk_details: list[RiskItemDetail]
) -> AnalysisOverview:
    """Aggregate requirement analyses into a compact overview."""
    return AnalysisOverview(
        strong_match_count=sum(1 for item in requirement_analyses if item.status == "strong_match"),
        partial_match_count=sum(1 for item in requirement_analyses if item.status == "partial_match"),
        gap_count=sum(1 for item in requirement_analyses if item.status == "gap"),
        high_risk_count=sum(1 for item in risk_details if item.severity == "high"),
    )
