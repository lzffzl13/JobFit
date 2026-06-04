# JobFit Agent 开发进度

> 记录项目开发进度、关键决策和后续方向。

## 当前状态

- 项目定位：AI-first 简历-JD 匹配分析系统
- 架构：LLM 做所有判断，代码只做管道编排和校验
- 核心流程：chunk → RAG 检索 → LLM 分析 → validate → 输出
- LLM 失败直接 503，无 fallback/规则引擎

## 已完成

### 架构重构（2026-06-03）

从规则引擎架构重构为 AI-first 架构：

- 删除 fallback.py（1000+ 行规则引擎）、jd_parser.py、resume_evidence.py、scoring.py、scoring_policy.py、jd_strategies/
- LLM 直接输出 match_score / bonus_score / extra_score
- validator.py 只做 clamp + confidence 警告 + 命中计数
- normalize_llm_payload 防御性解析 LLM 输出（字段别名、类型容错、缺失处理）
- llm_clients/ 可插拔设计（factory + deepseek 实现 + openai 占位）

### 基础功能

- FastAPI 后端 + `/jobfit/analyze` 接口
- 简历文本粘贴 + PDF/DOCX/TXT/Markdown 文件上传
- RAG 检索：BGE Embedding + ChromaDB 向量检索
- DeepSeek Chat API 集成
- 响应式前端（单页报告）

### 测试

- test_validator.py（9 个）— clamp、计数、confidence 警告、边界
- test_llm_normalizer.py（10 个）— 字段别名、缺失、类型容错
- test_retriever.py（2 个）— 分块与检索
- test_api_text.py — 集成测试 + mock LLM 测试

## 当前运行

```bash
uvicorn app.main:app --reload --port 7897
```

- Web：http://127.0.0.1:7897
- API 文档：http://127.0.0.1:7897/docs

## 后续优化

见 README.md「🗺️ 后续优化」。
