<div align="center">

# 🎯 JobFit Agent

**面向实习求职场景的 RAG 简历匹配分析系统**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-Chat-4D6BFE?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

上传简历 + 粘贴 JD → 结构化匹配分析报告

</div>

---

## 📖 项目简介

JobFit Agent 是一个面向**实习求职场景**的 RAG（Retrieval-Augmented Generation）系统。用户上传简历（PDF/DOCX/TXT/Markdown）或直接粘贴简历文本，配合岗位 JD，系统会输出：

- 🎯 **岗位匹配度评分**（确定性评分，不由 LLM 决定）
- 📋 **结构化 JD 要求**（核心要求 + 加分项）
- 🔍 **证据强度分析**（简历中哪些内容匹配了哪些要求）
- ⚠️ **能力缺口**（简历中缺失的关键技能）
- 💡 **简历优化建议**（针对性的改写建议）
- ❓ **高频面试问题**（基于简历和 JD 生成）
- 📎 **引用证据溯源**（每个结论都有证据支撑）

## 🏗️ 系统架构

```
┌──────────────┐     ┌──────────────┐
│   简历上传    │     │   JD 粘贴    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│ 文档解析      │     │ JD 结构化解析 │
│ PDF/DOCX/TXT │     │ 技能/角色/加分│
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────┐
│     RAG 检索（余弦相似度）        │
│   简历分块 ←→ JD 分块 交叉检索   │
└──────────────┬───────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────────┐
│  DeepSeek   │  │  确定性评分引擎  │
│  生成解释    │  │  公式 + 硬上限   │
└──────┬──────┘  └──────┬──────────┘
       │                │
       └────────┬───────┘
                ▼
       ┌─────────────────┐
       │  合并 → 最终报告  │
       └─────────────────┘
```

**核心设计原则：** LLM 只负责解释和建议，最终评分由确定性规则引擎计算，确保分数可解释、可复现、不受模型波动影响。

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 多格式简历解析 | 支持 PDF、DOCX、TXT、Markdown，自动编码检测 |
| JD 结构化分析 | 提取核心技能、加分项、推断岗位类型（后端/前端/测试/产品等） |
| 轻量 RAG 检索 | 基于 token 的余弦相似度，召回简历和 JD 中的相关证据片段 |
| DeepSeek 集成 | 结构化 JSON 输出，生成解释、建议和面试问题 |
| 确定性评分 | `总分 = min(核心70 + 加分15 + 竞争力15, 方向上限)` |
| 方向性硬上限 | 防止非技术简历虚高分、关键词堆砌拿高分 |
| 本地 Fallback | 未配置 API Key 也能跑通演示 |
| 响应式前端 | 单页报告，支持桌面/平板/手机 |
| Docker 部署 | 一键启动，国内 PyPI 镜像加速 |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- （可选）DeepSeek API Key

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
# 编辑 .env 填入 DEEPSEEK_API_KEY（可选，不填则使用本地规则）

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

服务启动后同样访问 http://127.0.0.1:9000

## 📐 评分引擎

### 评分公式

```
总分 = min(JD 核心匹配(70) + JD 加分项(15) + 简历额外竞争力(15), 岗位方向上限)
```

### 核心技能证据强度

| 强度 | 含义 | 得分比例 |
|------|------|---------|
| 0% | 简历中完全没有提及 | 0 |
| 30% | 仅技能栏/自评提到，无项目支撑 | 0.3 |
| 60% | 学习/课程/demo 级别描述 | 0.6 |
| 85% | 有项目经验，但缺少具体技术细节 | 0.85 |
| 100% | 有项目并说明具体实现细节 | 1.0 |

### 方向性硬上限

防止分数虚高的保护机制：

| 条件 | 分数上限 |
|------|---------|
| 无编程语言证据 | 45 |
| 无后端/Web/接口开发证据 | 55 |
| 无数据库/SQL 使用证据 | 70 |
| ≥50% 核心技能为 0% | 75 |
| 非技术简历且无技术项目 | 55 |
| 技能栏堆关键词但无项目证据 | 70 |

## 🗂️ 项目结构

```
JobFit/
├── app/
│   ├── main.py                  # FastAPI 应用入口
│   ├── api/
│   │   └── jobfit.py            # POST /jobfit/analyze 端点
│   ├── core/
│   │   └── config.py            # 环境变量配置（pydantic-settings）
│   ├── schemas/
│   │   └── jobfit.py            # Pydantic 数据模型
│   └── services/
│       ├── document_parser.py   # 文档解析（PDF/DOCX/TXT/MD）
│       ├── jd_parser.py         # JD 结构化解析、技能提取、角色推断
│       ├── jobfit.py            # 编排器：RAG + LLM + 本地评分
│       ├── llm.py               # DeepSeek API 客户端、输出归一化
│       ├── resume_evidence.py   # 简历证据分析、技能强度评分
│       ├── retriever.py         # 文本分块 + 余弦相似度检索
│       ├── scoring.py           # 评分门面，构建 EvaluationResult
│       └── scoring_policy.py    # 确定性评分引擎、方向上限
├── static/
│   └── index.html               # 单页前端（原生 HTML/CSS/JS）
├── samples/                     # 示例数据
│   ├── resume.txt
│   ├── jd_ai_app.txt
│   └── jd_backend.txt
├── tests/                       # 测试用例
├── Dockerfile                   # Docker 镜像（Alpine + 清华镜像）
├── docker-compose.yml           # 容器编排
├── pyproject.toml               # 项目配置
└── requirements.txt             # 生产依赖
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
| `score_breakdown` | 评分拆解（核心分/加分/竞争力/上限原因） |
| `summary` | 分析总结 |
| `jd_requirements` | 结构化 JD 要求 |
| `matched_strengths` | 匹配优势 |
| `gaps` | 能力缺口 |
| `extra_strengths` | 额外竞争力 |
| `resume_rewrites` | 简历优化建议 |
| `interview_questions` | 面试问题 |
| `evidence` | 引用证据片段 |

## 🧪 测试

```bash
# 运行测试
pytest -q

# 代码检查
ruff check app tests
```

**测试覆盖：**

- ✅ 文件上传集成测试
- ✅ 文本粘贴集成测试
- ✅ LLM 分数覆盖保护测试
- ✅ 本地 Fallback 评分测试
- ✅ LLM 输出归一化测试
- ✅ 文本分块与检索测试
- ✅ 评分基准测试（高/中/低/运营简历、关键词堆砌）

## ⚙️ 环境变量

在 `.env` 中配置：

```env
DEEPSEEK_API_KEY=              # DeepSeek API Key（可选，不填则使用本地规则）
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_TIMEOUT_SECONDS=45         # LLM 请求超时时间
MAX_CONTEXT_CHARS=9000         # 发送给 LLM 的最大 context 长度
```

> 💡 **不配置 API Key 也能运行**：系统会自动降级为本地规则生成演示结果，适合开发调试和功能演示。

## 📌 设计决策

### 为什么 LLM 不决定最终分数？

LLM 的输出存在不确定性——同样的输入可能得到不同的分数，且难以解释评分依据。本项目将 LLM 的职责限定为**生成解释和建议**，评分完全由确定性规则引擎计算，确保：

- 分数可复现
- 评分逻辑可解释
- 不受模型版本/参数变化影响
- 可通过测试用例验证

### 为什么用词频检索而不是向量检索？

当前版本（MVP）使用基于 token 的余弦相似度进行证据检索，实现简单、无需额外模型依赖。后续计划升级为 Embedding + 向量数据库的语义检索方案。

## 🗺️ 后续规划

- [ ] 升级 RAG：接入 Embedding 模型 + 向量数据库（ChromaDB）
- [ ] 引入 LangChain 框架，构建标准 RAG Pipeline
- [ ] 报告导出（Markdown/PDF）
- [ ] 更多岗位模板（Java 后端、LLM Agent、前端等）
- [ ] CI/CD 流水线

## 📄 License

MIT

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！**

</div>
