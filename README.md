<div align="center">

# 🎯 JobFit Agent

**AI-first 简历-JD 匹配分析系统**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-BGE_+_ChromaDB-FF6F00?logo=openai&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-Chat-4D6BFE?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

上传简历 + 粘贴 JD → AI 生成结构化匹配分析报告

</div>

---

## 📖 项目简介

JobFit Agent 是一个 **AI-first** 的简历-JD 匹配分析系统。核心设计原则：**LLM 做所有判断，代码只做管道和校验**。

用户上传简历（PDF/DOCX/TXT/Markdown）或直接粘贴文本，配合岗位 JD，系统输出：

- 🎯 **匹配度评分**（LLM 综合判断 0-100）
- 📋 **结构化 JD 要求**（核心要求 + 加分项，LLM 提取）
- 🔍 **逐项证据评估**（evidence_ratio 0.0~1.0 连续值 + confidence）
- ⚠️ **能力缺口与风险项**（低匹配的核心需求）
- 💡 **简历优化建议**（针对性改写）
- ❓ **高频面试问题**（基于简历和 JD 生成）
- 📎 **引用证据溯源**（向量检索的原始片段）

## 🏗️ 系统架构

```
┌──────────────┐     ┌──────────────┐
│   简历上传    │     │   JD 粘贴    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌─────────┐           ┌─────────┐
│ chunk() │           │ chunk() │   ← 按 600 字切块
└────┬────┘           └────┬────┘
     │                     │
     ▼                     ▼
┌─────────────────────────────────────┐
│  RAG 检索（BGE Embedding +          │
│  ChromaDB 向量检索）                 │
│  简历 top-8 ←→ JD top-5 交叉检索    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LLM (DeepSeek)                    │
│  • 从 JD 提取需求（任意岗位）       │
│  • 逐项评估 evidence_ratio + conf   │
│  • 输出 match / bonus / extra score │
│  • 输出 gaps / rewrites / questions │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  validator.py                       │
│  • clamp 分数到合法范围             │
│  • 低 confidence 发 warning         │
│  • 计算命中数（只数数，不算分）      │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ JobFitAnalysis │  ← 最终响应
       └───────────────┘

LLM 失败 → 503，不兜底
```

**核心设计原则：** LLM 负责所有判断（提取需求、评估匹配、给出分数），代码只做管道编排、防御性解析和分数 clamp。没有规则引擎、没有 fallback、没有硬编码策略。

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 多格式简历解析 | 支持 PDF、DOCX、TXT、Markdown，自动编码检测 |
| AI 需求提取 | LLM 从 JD 提取需求，支持任意岗位类型，无预设策略 |
| RAG 语义检索 | BGE Embedding + ChromaDB 向量检索，语义匹配替代词频匹配 |
| 连续证据评估 | evidence_ratio 0.0~1.0 连续浮点，LLM 自主判断 |
| LLM 综合评分 | match_score + bonus_score + extra_score，LLM 直接输出 |
| 防御性解析 | 容错 LLM 输出格式不一致（字段别名、类型强制、缺失处理） |
| 响应式前端 | 单页报告，支持桌面/平板/手机 |
| Docker 部署 | 一键启动，国内 PyPI 镜像加速 |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- DeepSeek API Key

### 本地运行

```bash
# 1. 克隆项目
git clone https://github.com/lzffzl13/JobFit.git
cd JobFit

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 5. 启动服务
uvicorn app.main:app --reload --port 9000
```

访问：

- 🌐 **Web 页面：** http://127.0.0.1:9000
- 📚 **API 文档：** http://127.0.0.1:9000/docs

### Docker 部署

```bash
cp .env.example .env
docker compose up --build
```

## 📐 评分机制

LLM 直接输出三个分数，validator 只做 clamp（不重算）：

| 分数 | 范围 | 含义 |
|------|------|------|
| `match_score` | 0-100 | LLM 综合匹配判断 |
| `bonus_score` | 0-15 | JD 加分项匹配 |
| `extra_score` | 0-15 | 简历额外竞争力（量化成果、开源、奖项等） |

每个需求由 LLM 评估：

| 字段 | 类型 | 含义 |
|------|------|------|
| `evidence_ratio` | 0.0~1.0 | 证据强度（连续值，非离散） |
| `confidence` | 0.0~1.0 | LLM 对该判断的置信度 |
| `priority` | core/bonus | 需求优先级 |

## 🗂️ 项目结构

```
JobFit/
├── app/
│   ├── main.py                  # FastAPI 应用入口
│   ├── api/
│   │   └── jobfit.py            # POST /jobfit/analyze 端点
│   ├── core/
│   │   ├── config.py            # 环境变量配置（pydantic-settings）
│   │   └── interfaces.py        # ILLMClient 协议
│   ├── schemas/
│   │   └── jobfit.py            # Pydantic 数据模型
│   └── services/
│       ├── document_parser.py   # 文档解析（PDF/DOCX/TXT/MD）
│       ├── jobfit.py            # 编排器：chunk → retrieve → LLM → validate
│       ├── llm.py               # prompt 构建 + JSON 防御性解析
│       ├── retriever.py         # 文本分块 + BGE Embedding + ChromaDB
│       ├── validator.py         # clamp + confidence 警告 + 命中计数
│       └── llm_clients/
│           ├── base.py          # BaseLLMClient 抽象基类
│           ├── deepseek.py      # DeepSeek API 实现
│           ├── openai.py        # OpenAI 占位（未实现）
│           └── factory.py       # get_llm_client() 工厂
├── static/
│   └── index.html               # 单页前端（原生 HTML/CSS/JS）
├── samples/                     # 示例数据
│   └── resume.txt
├── tests/                       # 测试用例
│   ├── test_api.py              # 集成测试（依赖真实 API）
│   ├── test_api_text.py         # 文本输入集成测试 + mock 测试
│   ├── test_llm_normalizer.py   # normalize_llm_payload 单元测试
│   ├── test_validator.py        # validator 单元测试
│   └── test_retriever.py        # 分块与检索测试
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## 🔌 API 接口

### `POST /jobfit/analyze`

**请求参数（form-data）：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `resume` | file | 否 | 简历文件（.pdf/.docx/.txt/.md） |
| `resume_text` | string | 否 | 直接粘贴的简历文本（优先级高于文件） |
| `jd_text` | string | ✅ | 岗位 JD 文本（最少 20 字） |

**响应字段：**

| 字段 | 说明 |
|------|------|
| `match_score` | 匹配度评分（0-100） |
| `summary` | 分析总结 |
| `score_breakdown` | 评分拆解（命中数 + LLM 三项分数 + 逐项详情） |
| `jd_requirements` | 结构化 JD 要求 |
| `matched_strengths` | 匹配优势 |
| `gaps` | 能力缺口 |
| `resume_rewrites` | 简历优化建议 |
| `interview_questions` | 面试问题 |
| `evidence` | 引用证据片段 |
| `risk_items` | 低匹配核心需求（ratio < 0.85） |

## 🧪 测试

```bash
# 运行单元测试（不需要 API）
pytest tests/test_validator.py tests/test_llm_normalizer.py -v

# 运行全部测试（集成测试需要 DeepSeek API）
pytest -v

# 代码检查
ruff check app tests
```

**测试覆盖：**

- ✅ validator clamp 逻辑（分数边界、命中计数、confidence 警告）
- ✅ normalize_llm_payload 边界情况（字段别名、缺失字段、类型容错）
- ✅ 文本分块与向量检索
- ✅ 集成测试（文本输入 + mock LLM）

## ⚙️ 环境变量

在 `.env` 中配置：

```env
LLM_PROVIDER=deepseek           # LLM 提供商
DEEPSEEK_API_KEY=               # DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_TIMEOUT_SECONDS=45          # LLM 请求超时时间
MAX_CONTEXT_CHARS=9000          # 发送给 LLM 的最大 context 长度
```

## 📌 设计决策

### 为什么 LLM 做所有判断？

早期版本使用规则引擎计算分数（70/15/15 公式 + 方向硬上限），LLM 只负责生成解释。问题：

- 规则引擎无法处理任意岗位类型（需要为每种岗位预设策略）
- evidence_ratio 离散分级（0/30/60/85/100）丢失精度
- 代码量膨胀（fallback.py 1000+ 行，jd_strategies/ 整个目录）

重构后：LLM 直接输出 match_score，validator 只 clamp 不重算。代码量减少 60%+，支持任意岗位。

### 为什么用 BGE Embedding + ChromaDB？

相比词频匹配，向量检索能捕捉语义相似性——"后端开发"能匹配"服务端开发"，即使字面完全不同。BGE-small-zh-v1.5 是中文场景下效果最好的轻量 Embedding 模型之一，ChromaDB 提供零配置的向量存储和检索。

## 🗺️ 后续优化

### 高优先级

- [ ] **检索质量优化**
  - 当前按固定 600 字符硬切，可能切断关键信息
  - 优化：滑动窗口分块、按段落/句子边界切分、重叠窗口（overlap）
  - 检索质量直接影响 LLM 判断准确性，这是上游
- [ ] **Prompt 迭代**
  - SYSTEM_PROMPT 和 build_user_prompt 需要 A/B 验证
  - 优化：Few-shot 示例、Chain-of-Thought 引导、输出格式约束更严格
  - 成本最低，效果提升可能最明显
- [ ] **结构化输出**
  - 当前靠 LLM 自觉输出 JSON，normalize_llm_payload 大量防御性解析
  - 优化：用 DeepSeek JSON mode / function calling / Pydantic output parser
  - 减少解析失败率，减少 normalize 代码
- [ ] **单元测试补全**
  - 当前集成测试依赖真实 API，需要 mock LLM client
  - 已完成：validator 测试 + normalizer 测试（19 个）
  - 待补：jobfit.py 编排器 mock 测试、document_parser 测试

### 中优先级

- [ ] **LLM 超时/重试机制** — 提升稳定性，当前超时直接 503
- [ ] **请求日志和性能追踪** — 方便调试和优化（记录每次请求的耗时、token 用量、检索命中率）
- [ ] **Reranker 重排序** — bge-reranker-base 二阶段检索，提升检索精度
- [ ] **缓存机制** — 相同 resume+JD 组合缓存结果，减少 API 调用
- [ ] **流式输出** — SSE 流式返回，前端实时显示分析进度
- [ ] **多模型支持** — 补全 OpenAI client，支持 GPT-4o / Claude 对比测试

### 低优先级

- [ ] **导出功能** — 分析报告导出为 Markdown/PDF
- [ ] **历史记录** — SQLite 存储分析结果，支持查看和对比
- [ ] **批量分析** — 一次投多个 JD 或多个简历
- [ ] **评估体系** — 人工标注数据集，量化 LLM 输出质量
- [ ] **CI/CD 流水线** — 自动测试 + 部署

## 📄 License

MIT

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！**

</div>
