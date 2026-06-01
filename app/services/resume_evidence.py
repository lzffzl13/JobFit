import re
from dataclasses import dataclass

from app.services.jd_parser import (
    SKILL_ALIASES,
    SkillRequirement,
    clean_text,
    contains_any_alias,
)

SKILL_ONLY_RATIO = 0.30
LEARNING_RATIO = 0.60
PROJECT_RATIO = 0.85
PROJECT_DETAIL_RATIO = 1.00

SKILL_SECTION_MARKERS = ("技能", "技术栈", "专业能力", "自评", "掌握", "熟悉", "了解", "skills", "technical skills")
PROJECT_SECTION_MARKERS = ("项目", "项目经历", "项目经验", "作品", "系统")
WORK_SECTION_MARKERS = ("工作经历", "实习经历", "任职", "工作经验")
AWARD_SECTION_MARKERS = ("奖项", "获奖", "竞赛", "荣誉", "奖学金")
NON_TECH_TERMS = (
    "运营",
    "市场",
    "销售",
    "投放",
    "活动策划",
    "用户增长",
    "转化率",
    "社群",
    "内容运营",
    "直播",
    "客服",
    "设计",
    "产品经理",
)
DEV_TERMS = (
    "开发",
    "后端",
    "接口",
    "api",
    "restful",
    "数据库",
    "服务",
    "系统",
    "部署",
    "测试",
    "代码",
    "工程",
)
PROJECT_DETAIL_TERMS: dict[str, tuple[str, ...]] = {
    "language_basic": ("fastapi", "flask", "django", "spring boot", "接口", "api", "数据库", "脚本", "服务"),
    "web_framework": ("接口", "api", "restful", "jwt", "路由", "分页", "中间件", "crud", "认证", "权限"),
    "database_sql": ("sqlalchemy", "orm", "crud", "查询", "联查", "索引", "事务", "表设计", "聚合", "优化"),
    "cache_message": ("缓存", "限流", "redis", "过期", "队列", "session", "热点"),
    "api_development": ("restful", "jwt", "crud", "用户", "文章", "评论", "分页", "认证", "权限", "异常"),
    "data_algorithm": ("leetcode", "复杂度", "排序", "查找", "动态规划", "哈希", "竞赛"),
}


@dataclass(frozen=True)
class ResumeLine:
    text: str
    section: str


@dataclass(frozen=True)
class SkillEvidence:
    requirement: SkillRequirement
    ratio: float
    level: str
    evidence: str
    matched_option: str | None

    @property
    def matched(self) -> bool:
        return self.ratio > 0


@dataclass(frozen=True)
class ExtraEvidence:
    name: str
    points: float
    evidence: str


@dataclass(frozen=True)
class ResumeEvidenceProfile:
    lines: list[ResumeLine]
    text: str
    is_non_technical: bool
    has_technical_project: bool
    has_backend_evidence: bool
    has_database_evidence: bool
    has_language_evidence: bool
    all_work_non_dev_without_backend_project: bool


def analyze_resume(resume_text: str) -> ResumeEvidenceProfile:
    lines = _parse_lines(resume_text)
    normalized = clean_text(resume_text)
    is_non_technical = _looks_non_technical(normalized)
    has_technical_project = any(line.section != "skills" and _is_technical_project_line(line) for line in lines)
    has_backend_evidence = any(
        line.section != "skills" and _is_backend_line(line) and _is_project_or_work_line(line)
        for line in lines
    )
    has_database_evidence = contains_any_alias(normalized, ("mysql", "postgresql", "sql", "sqlalchemy", "database", "sql crud"))
    has_language_evidence = contains_any_alias(normalized, ("python", "java"))
    work_lines = [line for line in lines if line.section == "work"]
    all_work_non_dev = bool(work_lines) and all(_is_non_dev_work_line(line.text) for line in work_lines)
    return ResumeEvidenceProfile(
        lines=lines,
        text=normalized,
        is_non_technical=is_non_technical,
        has_technical_project=has_technical_project,
        has_backend_evidence=has_backend_evidence,
        has_database_evidence=has_database_evidence,
        has_language_evidence=has_language_evidence,
        all_work_non_dev_without_backend_project=all_work_non_dev and not has_backend_evidence,
    )


def evaluate_requirement(
    requirement: SkillRequirement,
    profile: ResumeEvidenceProfile,
) -> SkillEvidence:
    best = SkillEvidence(requirement, 0.0, "0% (无证据)", "", None)
    for option in requirement.options:
        aliases = _evidence_aliases(requirement, option)
        for line in profile.lines:
            if not any(_alias_in_text(alias, line.text) for alias in aliases):
                continue
            candidate = _score_line(requirement, option, line)
            if candidate.ratio > best.ratio:
                best = candidate
    return best


def _evidence_aliases(requirement: SkillRequirement, option: str) -> tuple[str, ...]:
    aliases = SKILL_ALIASES.get(option, (option,))
    if requirement.dimension != "language_basic":
        if requirement.dimension == "database_sql":
            return (*aliases, "sqlalchemy", "orm", "crud", "数据库")
        if requirement.dimension == "cache_message":
            return (*aliases, "缓存")
        return aliases
    if option == "python":
        return (*aliases, "fastapi", "flask", "django", "sqlalchemy")
    if option == "java":
        return (*aliases, "spring boot", "springboot", "mybatis")
    return aliases


def score_explicit_bonus(
    bonus_requirements: list[SkillRequirement],
    profile: ResumeEvidenceProfile,
) -> tuple[float, int, list[str]]:
    if not bonus_requirements:
        return 0.0, 0, []
    matched = 0
    notes = []
    for requirement in bonus_requirements:
        evidence = evaluate_requirement(requirement, profile)
        if evidence.ratio > 0:
            matched += 1
            notes.append(f"{requirement.label}: {evidence.level}")
    return round(matched / len(bonus_requirements) * 15, 2), matched, notes


def score_extension_bonus(
    profile: ResumeEvidenceProfile,
    core_requirements: list[SkillRequirement],
) -> tuple[float, int, list[str]]:
    if not profile.has_technical_project and not profile.has_backend_evidence:
        return 0.0, 0, []

    core_options = {option for requirement in core_requirements for option in requirement.options}
    extension_groups = [
        ("容器化", ("docker", "kubernetes")),
        ("版本控制", ("git",)),
        ("Linux", ("linux",)),
        ("自动化测试", ("testing",)),
        ("CI/CD", ("ci/cd",)),
        ("消息队列", ("message queue",)),
        ("监控", ("monitoring",)),
    ]
    notes = []
    matched = 0
    for name, options in extension_groups:
        if any(option in core_options for option in options):
            continue
        if contains_any_alias(profile.text, options):
            matched += 1
            notes.append(name)
    return float(min(matched * 2, 10)), matched, notes


def score_extra_competitiveness(profile: ResumeEvidenceProfile) -> tuple[float, list[ExtraEvidence]]:
    if not profile.has_technical_project and not profile.has_backend_evidence:
        return 0.0, []

    extras: list[ExtraEvidence] = []
    extras.extend(_technical_quant_results(profile))
    extras.extend(_technical_awards(profile))
    extras.extend(_open_source_or_blog(profile))
    extras.extend(_complex_system_design(profile))

    team_doc = _team_or_doc(profile)
    if team_doc:
        extras.append(team_doc)

    total = min(sum(item.points for item in extras), 15.0)
    return round(total, 2), extras


def _score_line(requirement: SkillRequirement, option: str, line: ResumeLine) -> SkillEvidence:
    if requirement.dimension == "data_algorithm" and _contains_any_text(
        line.text.lower(), ("leetcode", "复杂度", "程序设计", "算法竞赛", "acm", "蓝桥杯")
    ):
        return SkillEvidence(requirement, PROJECT_DETAIL_RATIO, "100% (训练/竞赛细节)", line.text[:240], option)
    if _is_learning_line(line.text):
        return SkillEvidence(requirement, LEARNING_RATIO, "60% (学习/demo)", line.text[:240], option)
    if line.section == "skills" or _is_skill_like_line(line.text):
        return SkillEvidence(requirement, SKILL_ONLY_RATIO, "30% (仅技能栏)", line.text[:240], option)
    if _is_project_or_work_line(line):
        if _has_project_detail(requirement, line.text):
            return SkillEvidence(requirement, PROJECT_DETAIL_RATIO, "100% (项目+技术细节)", line.text[:240], option)
        return SkillEvidence(requirement, PROJECT_RATIO, "85% (项目经验)", line.text[:240], option)
    return SkillEvidence(requirement, SKILL_ONLY_RATIO, "30% (仅提到关键词)", line.text[:240], option)


def _parse_lines(text: str) -> list[ResumeLine]:
    raw_lines = [line.strip() for line in re.split(r"[\n。；;.!?！？]", text) if line.strip()]
    lines: list[ResumeLine] = []
    current_section = "general"
    for raw in raw_lines:
        section = _section_for_heading(raw)
        if section:
            current_section = section
        inferred = section or _infer_line_section(raw, current_section)
        lines.append(ResumeLine(raw, inferred))
    return lines


def _section_for_heading(line: str) -> str | None:
    compact = line.lower().replace(" ", "")
    if any(marker.lower() in compact for marker in SKILL_SECTION_MARKERS):
        return "skills"
    if any(marker.lower() in compact for marker in PROJECT_SECTION_MARKERS):
        return "project"
    if any(marker.lower() in compact for marker in WORK_SECTION_MARKERS):
        return "work"
    if any(marker.lower() in compact for marker in AWARD_SECTION_MARKERS):
        return "award"
    return None


def _infer_line_section(line: str, current_section: str) -> str:
    lower_line = line.lower()
    if _contains_any_text(lower_line, PROJECT_SECTION_MARKERS) or _is_technical_project_text(lower_line):
        return "project"
    if _contains_any_text(lower_line, AWARD_SECTION_MARKERS):
        return "award"
    if _contains_any_text(lower_line, WORK_SECTION_MARKERS):
        return "work"
    if _is_skill_like_line(line):
        return "skills"
    return current_section


def _looks_non_technical(text: str) -> bool:
    nontech_hits = sum(1 for term in NON_TECH_TERMS if term.lower() in text.lower())
    tech_hits = sum(1 for term in DEV_TERMS if term.lower() in text.lower())
    return nontech_hits >= 2 and tech_hits == 0 and not contains_any_alias(text, ("python", "java", "fastapi", "flask", "django", "mysql", "sql"))


def _is_non_dev_work_line(text: str) -> bool:
    lower_text = text.lower()
    return any(term in lower_text for term in NON_TECH_TERMS) and not any(term in lower_text for term in DEV_TERMS)


def _is_technical_project_line(line: ResumeLine) -> bool:
    return _is_project_or_work_line(line) and _is_technical_project_text(line.text.lower())


def _is_backend_line(line: ResumeLine) -> bool:
    lower = line.text.lower()
    if _contains_any_text(lower, ("no backend", "无后端", "没有后端")):
        return False
    if _contains_any_text(lower, ("test", "测试", "qa")) and not _contains_any_text(
        lower, ("开发", "实现", "设计", "built", "implemented", "restful", "crud")
    ):
        return False
    return _contains_any_text(lower, ("后端", "接口", "api", "restful", "fastapi", "flask", "django", "spring boot", "服务端"))


def _is_project_or_work_line(line: ResumeLine) -> bool:
    return line.section in {"project", "work"} or _is_technical_project_text(line.text.lower())


def _is_technical_project_text(text: str) -> bool:
    return _contains_any_text(text, DEV_TERMS) and (
        contains_any_alias(text, ("python", "java", "fastapi", "flask", "django", "spring boot", "mysql", "sql", "redis"))
        or _contains_any_text(text, ("接口", "api", "数据库", "后端", "服务", "系统"))
    )


def _is_learning_line(text: str) -> bool:
    return _contains_any_text(text.lower(), ("课程", "教程", "demo", "练习", "学习", "跟着", "自学", "实验"))


def _is_skill_like_line(text: str) -> bool:
    lower_text = text.lower()
    return _contains_any_text(lower_text, ("技能", "技术栈", "熟悉", "掌握", "了解", "会使用", "skills", "technical skills"))


def _has_project_detail(requirement: SkillRequirement, text: str) -> bool:
    lower_text = text.lower()
    detail_terms = PROJECT_DETAIL_TERMS.get(requirement.dimension, ())
    if _contains_any_text(lower_text, detail_terms):
        return True
    if requirement.dimension == "language_basic":
        return _contains_any_text(lower_text, DEV_TERMS)
    if requirement.dimension == "api_development":
        return _contains_any_text(lower_text, ("接口", "api", "restful", "crud"))
    return False


def _technical_quant_results(profile: ResumeEvidenceProfile) -> list[ExtraEvidence]:
    results = []
    pattern = r"\d+(\.\d+)?\s*(%|ms|s|qps|万|千|人|次|条|个|\+)"
    for line in profile.lines:
        if len(results) >= 3:
            break
        if _is_technical_project_line(line) and re.search(pattern, line.text, flags=re.I):
            results.append(ExtraEvidence("技术量化成果", 1.0, line.text[:240]))
    return results


def _technical_awards(profile: ResumeEvidenceProfile) -> list[ExtraEvidence]:
    results = []
    technical_award_terms = ("程序设计", "算法", "acm", "蓝桥杯", "数学建模", "软件", "开发", "编程", "计算机")
    for line in profile.lines:
        if len(results) >= 2:
            break
        lower = line.text.lower()
        has_award = _contains_any_text(lower, ("奖", "竞赛", "比赛", "荣誉", "奖学金"))
        if has_award and _contains_any_text(lower, technical_award_terms):
            results.append(ExtraEvidence("技术竞赛/奖项", 2.0, line.text[:240]))
    return results


def _open_source_or_blog(profile: ResumeEvidenceProfile) -> list[ExtraEvidence]:
    results = []
    for line in profile.lines:
        if len(results) >= 2:
            break
        lower = line.text.lower()
        if _contains_any_text(lower, ("开源", "github", "pull request", "技术博客")) and not _contains_any_text(lower, ("营销", "运营")):
            results.append(ExtraEvidence("开源贡献/技术博客", 2.0, line.text[:240]))
    return results


def _complex_system_design(profile: ResumeEvidenceProfile) -> list[ExtraEvidence]:
    results = []
    terms = ("微服务", "分布式", "高并发", "消息队列", "kafka", "rabbitmq", "限流", "熔断", "缓存一致性")
    for line in profile.lines:
        if len(results) >= 3:
            break
        if _is_technical_project_line(line) and _contains_any_text(line.text.lower(), terms):
            results.append(ExtraEvidence("复杂系统设计", 2.0, line.text[:240]))
    return results


def _team_or_doc(profile: ResumeEvidenceProfile) -> ExtraEvidence | None:
    for line in profile.lines:
        if _is_technical_project_line(line) and _contains_any_text(line.text.lower(), ("团队", "协作", "文档", "接口文档", "code review")):
            return ExtraEvidence("团队协作/文档规范", 2.0, line.text[:240])
    return None


def _contains_any_text(text: str, terms: tuple[str, ...]) -> bool:
    lower_text = text.lower()
    return any(term.lower() in lower_text for term in terms)


def _alias_in_text(alias: str, text: str) -> bool:
    alias_lower = alias.lower()
    text_lower = text.lower()
    if not alias_lower:
        return False
    if re.search(r"[a-zA-Z]", alias_lower):
        pattern = rf"(?<![a-zA-Z0-9]){re.escape(alias_lower)}(?![a-zA-Z0-9])"
        return re.search(pattern, text_lower) is not None
    return alias_lower in text_lower
