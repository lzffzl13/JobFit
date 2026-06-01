from app.services.llm import extract_skill_requirements, local_fallback_analysis


def test_local_fallback_analysis_scores_known_skills():
    analysis = local_fallback_analysis(
        resume_text="I built a Python FastAPI backend project and used Redis cache.",
        jd_text="Need Python, FastAPI, Redis, Docker. Basic RAG knowledge is preferred.",
    )

    assert analysis.match_score > 50
    assert analysis.fallback_used is True
    assert analysis.gaps


def test_extract_skill_requirements_treats_slash_as_alternative_group():
    requirements = extract_skill_requirements(
        "Need one of FastAPI/Flask/Django for backend work, plus MySQL and Redis."
    )

    web_requirement = next(item for item in requirements if item.label == "Python Web框架")
    labels = [requirement.label for requirement in requirements]
    assert web_requirement.requirement_type == "alternative"
    assert set(web_requirement.options) == {"fastapi", "flask", "django"}
    assert "MySQL/SQL" in labels
    assert "缓存/消息" in labels


def test_local_fallback_counts_alternative_group_once():
    analysis = local_fallback_analysis(
        resume_text="I built a Python FastAPI project and used Redis cache with Docker deployment.",
        jd_text="Need one of FastAPI/Flask/Django, plus Redis and Docker.",
    )

    requirement_names = [item.requirement for item in analysis.matched_strengths]
    assert "Python Web框架" in requirement_names
    assert analysis.score_breakdown.core_points == 70
    assert analysis.match_score >= 70


def test_bonus_items_do_not_drag_score_too_low():
    analysis = local_fallback_analysis(
        resume_text="I built a Python FastAPI project and used Redis with Docker.",
        jd_text="Need Python, FastAPI, Redis, Docker. LangChain experience is a bonus. Prompt engineering is a plus.",
    )

    assert analysis.match_score >= 70
    assert any(item.requirement == "AI应用/RAG能力" for item in analysis.gaps)


def test_chinese_requirement_parsing_covers_backend_jd():
    requirements = extract_skill_requirements(
        "熟练掌握Python/JAVA基础语法，了解常用数据结构和算法。了解至少一种Python Web框架(Django、Flask、FastAPI等)。"
        "了解Spring Boot、MyBatis等主流框架。熟悉MySQL并能编写基本SQL语句。"
    )

    labels = [item.label for item in requirements]
    assert "Python/Java 基础" in labels
    assert "数据结构/算法基础" in labels
    assert "Python Web框架" in labels
    assert "Spring Boot / MyBatis" in labels
    assert "MySQL/SQL" in labels
    assert all(label not in labels for label in ["列表", "字典", "元组", "查询", "插入", "更新", "删除"])


def test_targeted_backend_resume_can_score_above_90():
    jd_text = (
        "熟练掌握Python/JAVA基础语法，了解常用数据结构和算法。"
        "了解至少一种Python Web框架(Django、Flask、FastAPI等)。"
        "参与后端接口开发，熟悉MySQL并能编写基本SQL语句。"
        "了解Spring Boot、MyBatis等主流框架，有Java生态协作经验优先。"
    )
    resume_text = (
        "技能：Python、FastAPI、MySQL、SQLAlchemy、Docker、Git、pytest，了解 Spring Boot 和 MyBatis 基础。\n"
        "数据结构与算法：熟悉列表、字典、哈希表，完成 80+ 道 LeetCode 练习。\n"
        "项目：FastAPI 博客后端系统，完成用户、文章、评论 RESTful API，使用 SQLAlchemy 设计 MySQL 表并实现 CRUD。\n"
        "工程化：使用 Docker 部署，pytest 编写接口测试，Git 协作维护接口文档。\n"
        "成果：接口响应时间降低 30%，获得校级程序设计竞赛二等奖和奖学金。"
    )

    analysis = local_fallback_analysis(resume_text=resume_text, jd_text=jd_text)

    assert analysis.match_score >= 90
    assert analysis.score_breakdown.core_points == 70
    assert analysis.score_breakdown.extra_points >= 5
    assert analysis.score_breakdown.bonus_points >= 10


def test_python_fastapi_mysql_without_spring_stays_above_80():
    analysis = local_fallback_analysis(
        resume_text=(
            "我主要使用 Python、FastAPI、MySQL 和 SQLAlchemy 开发博客后端项目，"
            "实现用户、文章、评论 CRUD 接口，并用 Docker 部署和 pytest 测试。"
        ),
        jd_text=(
            "掌握Python/JAVA基础语法，了解至少一种Python Web框架(Django、Flask、FastAPI等)，"
            "熟悉MySQL和基本SQL语句。了解Spring Boot、MyBatis等主流框架。"
        ),
    )

    assert analysis.match_score >= 70
    assert "Spring Boot / MyBatis" in [item.requirement for item in analysis.gaps]
    assert "Java" not in " ".join(analysis.risk_items)


def test_examples_are_grouped_as_competencies_not_atomic_skills():
    requirements = extract_skill_requirements(
        "了解列表、字典、元组等基础数据结构；能完成查询、插入、更新、删除等 SQL CRUD 操作。"
    )

    labels = [item.label for item in requirements]
    assert labels.count("数据结构/算法基础") == 1
    assert labels.count("MySQL/SQL") == 1
    assert not {"列表", "字典", "元组", "查询", "插入", "更新", "删除"} & set(labels)
