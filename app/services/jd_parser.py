import re
from dataclasses import dataclass

CORE_POINTS_TOTAL = 70
BONUS_POINTS_TOTAL = 15
EXTRA_POINTS_TOTAL = 15

SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "python": ("python",),
    "java": ("java",),
    "fastapi": ("fastapi",),
    "flask": ("flask",),
    "django": ("django",),
    "spring boot": ("spring boot", "springboot"),
    "mybatis": ("mybatis",),
    "sql": ("sql", "sql语句", "sql 语句"),
    "mysql": ("mysql",),
    "postgresql": ("postgresql", "postgres"),
    "sqlalchemy": ("sqlalchemy",),
    "database": ("数据库", "database", "db"),
    "sql crud": (
        "crud",
        "增删改查",
        "查询",
        "插入",
        "更新",
        "删除",
        "select",
        "insert",
        "update",
        "delete",
    ),
    "redis": ("redis",),
    "cache": ("缓存", "cache"),
    "docker": ("docker",),
    "kubernetes": ("kubernetes", "k8s"),
    "linux": ("linux",),
    "git": ("git", "github", "gitlab"),
    "testing": ("测试", "单元测试", "接口测试", "pytest", "unittest", "test"),
    "ci/cd": ("ci/cd", "cicd", "ci", "持续集成", "持续部署"),
    "message queue": ("消息队列", "rabbitmq", "kafka", "mq"),
    "monitoring": ("监控", "prometheus", "grafana"),
    "data structures": (
        "数据结构",
        "列表",
        "字典",
        "元组",
        "数组",
        "链表",
        "栈",
        "队列",
        "哈希",
        "list",
        "dict",
        "tuple",
    ),
    "algorithms": ("算法", "排序", "查找", "leetcode", "algorithm", "algorithms"),
    "api development": (
        "api",
        "接口",
        "接口开发",
        "接口设计",
        "restful",
        "rest api",
        "后端开发",
        "业务功能开发",
    ),
    "jwt": ("jwt",),
    "rag": ("rag",),
    "llm": ("llm", "大模型"),
    "agent": ("agent", "智能体"),
    "langchain": ("langchain",),
    "embedding": ("embedding", "向量化", "向量"),
    "retrieval": ("retrieval", "检索", "向量检索", "召回"),
    "prompt engineering": ("prompt engineering", "prompt", "提示词"),
}

BONUS_MARKERS = ("优先", "加分", "bonus", "preferred", "nice to have")
ALTERNATIVE_MARKERS = ("任一", "任选", "至少一种", "至少一项", "之一", "one of", "either")


@dataclass(frozen=True)
class SkillRequirement:
    name: str
    options: tuple[str, ...]
    clause: str
    priority: str
    weight: float = 1.0
    requirement_type: str = "single"
    required_count: int = 1
    dimension: str = "general"

    @property
    def label(self) -> str:
        return self.name

    @property
    def is_bonus(self) -> bool:
        return self.priority == "bonus"

    @property
    def category(self) -> str:
        prefix = "bonus" if self.is_bonus else "core"
        return f"{prefix}_{self.requirement_type}"


@dataclass(frozen=True)
class ParsedJD:
    role_type: str
    core_requirements: list[SkillRequirement]
    bonus_requirements: list[SkillRequirement]
    has_explicit_bonus: bool
    clauses: list[str]

    @property
    def requirements(self) -> list[SkillRequirement]:
        return [*self.core_requirements, *self.bonus_requirements]


def parse_jd(jd_text: str) -> ParsedJD:
    normalized = clean_text(jd_text)
    clauses = split_clauses(normalized)
    role_type = infer_role_type(normalized)
    explicit_bonus_clauses = [clause for clause in clauses if is_bonus_clause(clause)]
    core_requirements = _extract_core_requirements(normalized, clauses, role_type)
    bonus_requirements = _extract_bonus_requirements(
        normalized,
        clauses,
        explicit_bonus_clauses,
        core_requirements,
        role_type,
    )
    return ParsedJD(
        role_type=role_type,
        core_requirements=core_requirements,
        bonus_requirements=bonus_requirements,
        has_explicit_bonus=bool(explicit_bonus_clauses),
        clauses=clauses,
    )


def extract_skill_requirements(jd_text: str) -> list[SkillRequirement]:
    return parse_jd(jd_text).requirements


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\r\t]+", "\n", text)
    text = re.sub(r"[•·●◆■]+", " ", text)
    text = re.sub(r"(?m)^\s*[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_clauses(text: str) -> list[str]:
    cleaned = clean_text(text)
    normalized = re.sub(r"([。；;.!?！？])", r"\1\n", cleaned)
    normalized = re.sub(r"(?<![A-Za-z0-9])/了解", "\n了解", normalized)
    normalized = re.sub(r"(?<![A-Za-z0-9])-了解", "\n了解", normalized)
    normalized = re.sub(r"\s+[0-9]+[.、)]\s*", "\n", normalized)
    clauses = []
    for line in normalized.splitlines():
        line = line.strip(" -：:")
        if line:
            clauses.append(line)
    return clauses or [cleaned]


def infer_role_type(text: str) -> str:
    lower_text = text.lower()
    if _contains_any_text(lower_text, ("运营", "市场", "销售", "投放", "用户增长")):
        return "business"
    if _contains_any_text(lower_text, ("测试开发", "自动化测试", "测试工程师")):
        return "testing"
    if _contains_any_text(lower_text, ("产品经理", "产品实习", "需求分析")):
        return "product"
    if _contains_any_text(lower_text, ("java后端", "java 后端", "spring boot", "springboot")) and not _alias_in_text("python", text):
        return "java_backend"
    if _contains_any_text(lower_text, ("后端", "backend", "api", "接口", "fastapi", "flask", "django", "数据库")):
        return "backend"
    if _contains_any_text(lower_text, ("前端", "frontend", "react", "vue")):
        return "frontend"
    return "general"


def is_bonus_clause(clause: str) -> bool:
    lower_clause = clause.lower()
    return any(marker in lower_clause for marker in BONUS_MARKERS) or bool(
        re.search(r"\b(is|as|a)\s+a?\s*plus\b", lower_clause)
    )


def contains_alias(key: str, text: str) -> bool:
    return any(_alias_in_text(alias, text) for alias in SKILL_ALIASES.get(key, (key,)))


def contains_any_alias(text: str, keys: tuple[str, ...]) -> bool:
    return any(contains_alias(key, text) for key in keys)


def aliases_for(options: tuple[str, ...]) -> list[str]:
    aliases: list[str] = []
    for option in options:
        aliases.extend(SKILL_ALIASES.get(option, (option,)))
    return aliases


def _extract_core_requirements(
    text: str,
    clauses: list[str],
    role_type: str,
) -> list[SkillRequirement]:
    requirements: list[SkillRequirement] = []
    seen_dimensions: set[str] = set()

    def add(requirement: SkillRequirement | None) -> None:
        if requirement is None or requirement.dimension in seen_dimensions:
            return
        seen_dimensions.add(requirement.dimension)
        requirements.append(requirement)

    add(_language_requirement(text, clauses))
    add(_web_framework_requirement(clauses, role_type))
    add(_api_requirement(clauses))
    add(_database_requirement(clauses))
    add(_cache_requirement(clauses))
    add(_data_algorithm_requirement(text, clauses))

    requirements.sort(key=_requirement_sort_key)
    return requirements


def _extract_bonus_requirements(
    text: str,
    clauses: list[str],
    explicit_bonus_clauses: list[str],
    core_requirements: list[SkillRequirement],
    role_type: str,
) -> list[SkillRequirement]:
    requirements: list[SkillRequirement] = []
    seen_dimensions = {item.dimension for item in core_requirements}

    def add(requirement: SkillRequirement | None) -> None:
        if requirement is None or requirement.dimension in seen_dimensions:
            return
        seen_dimensions.add(requirement.dimension)
        requirements.append(requirement)

    bonus_source = explicit_bonus_clauses
    for clause in bonus_source:
        options = _extract_known_options(clause)
        if not options:
            continue
        dimension = _dimension_for_options(options)
        add(
            SkillRequirement(
                name=_bonus_name_for_options(options),
                options=options,
                clause=clause,
                priority="bonus",
                requirement_type="competency" if len(options) > 1 else "single",
                dimension=dimension,
            )
        )

    spring_clause = _find_clause_for_terms(clauses, ("spring boot", "springboot", "mybatis"))
    if spring_clause and ("backend" in role_type or role_type == "general") and not explicit_bonus_clauses:
        add(
            SkillRequirement(
                name="Spring Boot / MyBatis",
                options=tuple(
                    option
                    for option in ("spring boot", "mybatis", "java")
                    if contains_alias(option, spring_clause)
                )
                or ("java",),
                clause=spring_clause,
                priority="bonus",
                requirement_type="competency",
                dimension="java_ecosystem",
            )
        )

    ai_clause = _find_clause_for_terms(clauses, ("rag", "llm", "大模型", "agent", "langchain", "prompt", "向量检索"))
    if ai_clause and "ai_application" not in seen_dimensions and (is_bonus_clause(ai_clause) or "backend" in role_type):
        add(
            SkillRequirement(
                name="AI应用/RAG能力",
                options=("rag", "llm", "agent", "retrieval", "prompt engineering", "langchain"),
                clause=ai_clause,
                priority="bonus",
                requirement_type="competency",
                dimension="ai_application",
            )
        )

    requirements.sort(key=_requirement_sort_key)
    return requirements


def _language_requirement(text: str, clauses: list[str]) -> SkillRequirement | None:
    clause = _find_clause_for_terms(clauses, ("python", "java", "编程语言", "基础语法"))
    if not clause:
        return None
    options = tuple(option for option in ("python", "java") if contains_alias(option, clause))
    if not options:
        options = tuple(option for option in ("python", "java") if contains_alias(option, text))
    if not options:
        return None
    return SkillRequirement(
        name="Python/Java 基础" if len(options) > 1 else f"{options[0].title()} 基础",
        options=options,
        clause=clause,
        priority="core",
        requirement_type="alternative" if len(options) > 1 else "single",
        dimension="language_basic",
    )


def _web_framework_requirement(clauses: list[str], role_type: str) -> SkillRequirement | None:
    clause = _find_clause_for_terms(
        clauses,
        ("fastapi", "flask", "django", "spring boot", "springboot", "web框架", "web 框架", "web framework"),
    )
    if not clause or is_bonus_clause(clause):
        return None
    python_options = tuple(
        option for option in ("django", "flask", "fastapi") if contains_alias(option, clause)
    )
    java_options = tuple(option for option in ("spring boot",) if contains_alias(option, clause))
    if python_options:
        return SkillRequirement(
            name="Python Web框架",
            options=python_options,
            clause=clause,
            priority="core",
            requirement_type="alternative",
            dimension="web_framework",
        )
    if java_options and role_type == "java_backend":
        return SkillRequirement(
            name="Java Web框架",
            options=java_options,
            clause=clause,
            priority="core",
            requirement_type="single",
            dimension="web_framework",
        )
    if "web" in clause.lower() or "框架" in clause:
        return SkillRequirement(
            name="Web框架",
            options=("fastapi", "flask", "django", "spring boot"),
            clause=clause,
            priority="core",
            requirement_type="alternative",
            dimension="web_framework",
        )
    return None


def _api_requirement(clauses: list[str]) -> SkillRequirement | None:
    clause = _find_clause_for_terms(
        clauses,
        ("api", "接口", "restful", "后端开发", "业务功能开发", "接口开发"),
    )
    if not clause or is_bonus_clause(clause):
        return None
    return SkillRequirement(
        name="接口开发能力",
        options=("api development",),
        clause=clause,
        priority="core",
        requirement_type="competency",
        dimension="api_development",
    )


def _database_requirement(clauses: list[str]) -> SkillRequirement | None:
    clause = _find_clause_for_terms(
        clauses,
        ("mysql", "postgresql", "sql", "数据库", "增删改查", "查询", "插入", "更新", "删除"),
    )
    if not clause or is_bonus_clause(clause):
        return None
    options = tuple(
        option
        for option in ("mysql", "postgresql", "sql", "database", "sql crud")
        if contains_alias(option, clause)
    ) or ("sql",)
    return SkillRequirement(
        name="MySQL/SQL",
        options=options,
        clause=clause,
        priority="core",
        requirement_type="competency",
        dimension="database_sql",
    )


def _cache_requirement(clauses: list[str]) -> SkillRequirement | None:
    clause = _find_clause_for_terms(clauses, ("redis", "缓存", "cache"))
    if not clause or is_bonus_clause(clause):
        return None
    return SkillRequirement(
        name="缓存/消息",
        options=("redis", "cache"),
        clause=clause,
        priority="core",
        requirement_type="competency",
        dimension="cache_message",
    )


def _data_algorithm_requirement(text: str, clauses: list[str]) -> SkillRequirement | None:
    clause = _find_clause_for_terms(
        clauses,
        ("数据结构", "算法", "列表", "字典", "元组", "数组", "链表", "leetcode", "algorithm"),
    )
    if not clause and not contains_any_alias(text, ("data structures", "algorithms")):
        return None
    return SkillRequirement(
        name="数据结构/算法基础",
        options=("data structures", "algorithms"),
        clause=clause or "数据结构/算法基础",
        priority="core",
        requirement_type="competency",
        dimension="data_algorithm",
    )


def _extract_known_options(text: str) -> tuple[str, ...]:
    ignored = {"database", "sql crud", "cache"}
    options = []
    for key in SKILL_ALIASES:
        if key in ignored:
            continue
        if contains_alias(key, text):
            options.append(key)
    return _ordered_unique(options)


def _dimension_for_options(options: tuple[str, ...]) -> str:
    if any(option in {"spring boot", "mybatis", "java"} for option in options):
        return "java_ecosystem"
    if any(option in {"rag", "llm", "agent", "langchain"} for option in options):
        return "ai_application"
    if any(option in {"docker", "kubernetes", "linux", "git", "testing", "ci/cd", "message queue", "monitoring"} for option in options):
        return "technical_extension"
    return "bonus_skill"


def _bonus_name_for_options(options: tuple[str, ...]) -> str:
    if any(option in {"spring boot", "mybatis"} for option in options):
        return "Spring Boot / MyBatis"
    if "java" in options:
        return "Java生态协作经验"
    if any(option in {"rag", "llm", "agent", "langchain"} for option in options):
        return "AI应用/RAG能力"
    if any(option in {"docker", "kubernetes", "linux", "git", "testing", "ci/cd", "message queue", "monitoring"} for option in options):
        return "工程化扩展能力"
    return "其他岗位相关技术"


def _find_clause_for_terms(clauses: list[str], terms: tuple[str, ...]) -> str | None:
    for clause in clauses:
        for term in terms:
            if re.search(r"[a-zA-Z]", term):
                if _alias_in_text(term, clause):
                    return clause
            elif term in clause:
                return clause
    return None


def _requirement_sort_key(requirement: SkillRequirement) -> tuple[int, int, str]:
    bucket_rank = 1 if requirement.is_bonus else 0
    dimension_rank = {
        "language_basic": 0,
        "web_framework": 1,
        "database_sql": 2,
        "cache_message": 3,
        "api_development": 4,
        "data_algorithm": 5,
        "java_ecosystem": 6,
        "ai_application": 7,
        "technical_extension": 8,
        "bonus_skill": 9,
    }.get(requirement.dimension, 99)
    return (bucket_rank, dimension_rank, requirement.name)


def _contains_any_text(text: str, terms: tuple[str, ...]) -> bool:
    lower_text = text.lower()
    return any(term.lower() in lower_text for term in terms)


def _ordered_unique(values) -> tuple[str, ...]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _alias_in_text(alias: str, text: str) -> bool:
    alias_lower = alias.lower()
    text_lower = text.lower()
    if not alias_lower:
        return False
    if re.search(r"[a-zA-Z]", alias_lower):
        pattern = rf"(?<![a-zA-Z0-9]){re.escape(alias_lower)}(?![a-zA-Z0-9])"
        return re.search(pattern, text_lower) is not None
    return alias_lower in text_lower
